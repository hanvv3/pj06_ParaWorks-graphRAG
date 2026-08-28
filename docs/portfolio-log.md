# ParaWorks Portfolio Log

Last updated: 2026-08-28

This document records ParaWorks work in a portfolio-friendly format. Keep adding
short entries here whenever the product, architecture, UX, verification, or
demo story changes.

## 2026-08-28 Auto-Review Trust Promotion finalized plan and frozen profile

- Recorded user approval of the C.5 design and converted it into the sixteen
  ordered TDD/commit checkpoints in
  `docs/superpowers/plans/2026-08-28-auto-review-trust-promotion.md`.
- Mapped the current V2.0 schemas, graph, lifecycle service, Review transition,
  promotion, indexing, API, and same-screen frontend surfaces before assigning
  implementation ownership. V2.0 remains an immutable compatibility surface;
  V2.1 uses separate state, graph, service, and response models.
- Split work into contracts/persistence, immutable evidence refs, resolution
  actors, exact provenance/reaffirmation, revoke/tombstones, deterministic
  eligibility/policy, real LangChain Terra validation, validation leases/cache,
  audit/rollout breaker, signed one-click launch, dual-version LangGraph/API,
  existing-screen UX, and PostgreSQL/golden release gates.
- The finalized plan freezes two implementation details: one approval-effect
  row with many canonical evidence-link rows, and
  a bounded public audit-state projection so `감사 필요` and `조치 필요` remain
  truthful after reload. No hidden provenance identity/count or raw reason is
  exposed.
- Froze the exact five-route extraction profile on OpenAI
  `gpt-5.4-mini-2026-03-17`, reasoning `none`, at 10,000 input/2,048 total
  output tokens and USD 0.75/M input plus USD 4.50/M output. Each selected agent
  returns zero or one candidate; all five routes reserve at most USD 0.083580.
- Froze validation on OpenAI `gpt-5.6-terra`, reasoning `medium`, at 6,000
  input/3,072 total output tokens, four candidates per batch and at most two
  batches/five candidates per workflow. Two batches reserve USD 0.097728, so
  the maximum profile is USD 0.181308 with USD 0.018692 headroom below the
  immutable USD 0.20 limit. Deployment values may only confirm exact registry
  equality.
- Three independent read-only plan audits found and closed pre-implementation
  gaps: V2.0 human provenance, workflow-owner permission drift, provider-call
  transaction boundaries, exact generator identity, hidden-collision storage,
  batch cost multiplication, revoke/reindex MVCC races, audit crash recovery,
  rollout auto-promotion, and post-completion revoke reconciliation.
- The revised plan uses one authoritative no-retry validation-call ledger with
  atomic cost reservation, a complete keyed trusted-fingerprint projection,
  common per-document PostgreSQL transaction locks plus active tombstone search
  filtering, and a persistent `0 -> 10 -> 100` operator authorization latch.
  Two-percent audits begin only for newly authorized full-enforce workflows.
- Persisted Assistant messages now have a planned complete evidence-dependency
  contract; incomplete, stale, revoked, quarantined, permission-incompatible,
  or unavailable bindings fail closed across list, context, summary, email, and
  RAG projections without leaking partial citations.
- Source authority is planned around the server-owned content signature, exact
  parser policy/run, and relational current-document-version pointer. Connector
  signatures, display labels, timestamps guessed with `now()`, and `MAX(id)`
  repairs cannot authorize C.5 serving.
- Provider and rollout changes use append-only control events with atomic
  aggregate backpointers. Quality revoke uses an immutable assessment and
  audit-or-correction, commits breaker/quarantine first, and never rewrites a
  confirmed audit; a correction permanently requires a new reviewed policy.
- New V2.0 and V2.1 candidates both receive immutable evidence provenance, but
  only V2.1 is auto-review eligible. Stored V2.1 threads remain resumable with
  global disabled or a missing provider key and fall back safely to the human
  queue rather than changing graph version.
- Deterministic CI does not claim model quality. A separately authorized,
  sanitized paid Terra aggregate benchmark is required before shadow rollout;
  otherwise the feature stays disabled.
- This remains planning only. No product code, migration, model/provider call,
  paid-mode enablement, push, merge, or PR was performed. Planning/specification
  is finalized. The next unapproved action is execution-mode selection plus
  explicit product-code authorization, and that next action begins actual
  implementation.

## 2026-08-28 Auto-Review Trust Promotion design

- Wrote the proposed Deliverable C.5 design between Review Queue HITL V2 and
  the RAG retriever migration. This is planning/specification only; no product
  code, database schema, provider call, feature enablement, push, or merge was
  performed.
- Reframed trust as three layers: canonical source evidence, pending AI
  knowledge, and trusted knowledge. Raw evidence may support answers later but
  does not become an official Decision, Timeline, History, or Todo by itself.
- Kept AI candidates pending first. Only public/internal direct-fact Timeline
  and narrowly extractive History candidates can enter the initial auto-review
  allowlist; Decision, Todo, restricted, inferred, conflicting, and uncertain
  candidates remain human-reviewed.
- Separated an independent OpenAI `gpt-5.6-terra` medium-reasoning structured
  validator from the final deterministic policy authority. Validator failure,
  malformed output, source drift, permission drift, or budget overflow falls
  back to human review and never silently changes models.
- Preserved the existing locked exactly-once Review transition and promotion
  boundary. Auto-policy approval cannot insert trusted knowledge directly, and
  tests continue to use fake models rather than live provider APIs.
- Designed post-migration approval provenance so exact duplicate
  reaffirmations reuse a canonical knowledge row without letting one revoke
  remove another active human/auto approval. The last active provenance alone
  can revoke the shared knowledge, companion Timeline, and exact vector state.
- Kept UX depth unchanged: existing Review surfaces gain bounded counts,
  auto-validation badges, filtering, audit details, and an authorized revoke
  action instead of a new page or wizard.
- Proposed disabled/shadow/enforce rollout with default disabled, 10% stable
  canary, zero hard-negative/permission/version violations, at least 500 shadow
  comparisons, and at least 99% precision before enforce.
- A read-only architecture audit caught durable-graph, shared-revoke, raw-index,
  and post-audit gaps before implementation. The revised design keeps V2.0
  immutable, adds V2.1-only state, blocks inaccessible collision buckets without
  leaking them, and persists sampled human audits plus an enforce-to-shadow
  breaker.
- The written design was subsequently approved and converted into the separate
  TDD implementation plan above. Sequence remains C.5 -> D Retriever/RAG
  Answer Graph V2 -> E Neo4j GraphRAG -> Slack recovery last.

## 2026-08-27 Review Queue HITL V2 release verification

- Completed the two-screen user journey without adding navigation depth:
  `Integrations -> 검토 후보 만들기 -> Review -> 검토 완료`. Review actions
  update authoritative database state but never auto-resume; the explicit
  completion action resumes the same workflow thread.
- Production extraction continues through the existing LangChain structured
  adapters, while the review workflow uses actual LangGraph `interrupt()` and
  same-thread `Command(resume=...)`. Tests and smoke runs used deterministic
  models/fake connector clients and made no live provider calls.
- Verified PostgreSQL restart/recovery, commit/checkpoint reconciliation,
  shared-scope exact-batch launch, exactly-once decision/history/todo promotion
  and companion Timeline provenance, concurrent resume, terminal races, root
  checkpoint namespace, privacy scanning, and generated-id-only cleanup:
  `14 passed, 0 skipped` on a disposable PostgreSQL 16 + pgvector test target.
- A real RED privacy check found that checkpoint state still serialized an
  unused, always-empty `review_item_ids` key. The minimal correction removed
  that checkpoint-only field without changing ReviewItem rows, graph nodes,
  API payloads, permissions, or promotion behavior; its focused RED was
  `3 failed, 4 passed` and GREEN was `7 passed`.
- Final review caught that a normal approval changed the live ReviewItem
  distribution and was incorrectly classified as checkpoint corruption. A
  focused RED (`3 failed`) and real PostgreSQL RED (`2 failed`) proved the UI's
  explicit completion path was rotating/repairing the checkpoint and consuming
  five state versions. The corrected validator treats current PostgreSQL
  ReviewItem resolution as authoritative while the paused tuple must retain
  exact identity, schema, and total count. Normal completion now preserves the
  checkpoint thread, never invokes repair/rotation, and uses the contractual
  `+2` state-version path with no duplicate AgentRun, ReviewItem, knowledge, or
  Timeline effects.
- Strengthened restart evidence so application A and B own different
  SQLAlchemy engines/pools/sessionmakers in addition to different checkpoint
  runtimes/pools/savers. A is fully closed and disposed before B is created.
- Fresh release evidence: touched-module regression `199 passed`; Deliverable C
  `333 passed` with its one known deferred Slack failure still visible;
  Deliverable B `229 passed`; non-Slack backend `989 passed, 1 skipped,
  10 deselected`; full backend `989 passed, 1 skipped` with exactly the ten
  user-deferred Slack failures and no new failure.
- Cleared the whole-tree legacy Ruff baseline from 32 findings using 23 safe
  automatic fixes and nine minimal behavior-preserving B008/N806/F841 edits.
  Slack files received import/mode cleanup only. Whole-tree Ruff, lock, and
  `git diff --check` are green.
- Frontend lint/build passed with 18 generated pages. Deterministic Playwright
  passed desktop V2 `48`, mobile V2 `44`, and adjacent UX `9`; deterministic
  backend two-screen smoke passed `9`.
- V2 remains disabled by default and Slack recovery remains last. This
  historical next boundary was superseded on 2026-08-28 by the separately
  proposed Deliverable C.5 trust-promotion design before Deliverable D.

## 2026-08-27 Review Queue HITL V2 implementation plan

- Converted the approved Deliverable C design into eleven independently
  reviewable TDD commits covering truthful V1 metadata, canonical source refs,
  V1/V2 waterlines, locked Review transitions, real LangChain adapters, actual
  LangGraph interrupt/resume, lifecycle APIs, the two-screen UX, and PostgreSQL
  recovery/concurrency evidence.
- Proposed exact additive diagnostic, dry-run, lifecycle-status, error-code,
  source-ref, and Review replay/promotion contracts for human approval so later
  implementation cannot silently change a gated output or trust boundary.
- Kept the no-migration decision: scoped batch ownership uses the existing
  workflow/request/evidence tables plus a PostgreSQL advisory transaction lock,
  while async sync recovery and cutover waterlines use `Source.raw_metadata`.
- Required configured production model paths to use the existing LangChain
  structured-output adapters and the workflow to use LangGraph 1.2
  `interrupt()` / same-thread `Command(resume=...)`; deterministic models remain
  limited to dry-run, local/demo, and tests.
- Added explicit verification gates for exact-once Review promotion, checkpoint
  mode matching, restart/resume, frontend desktop/mobile behavior, the full
  non-Slack suite, and an unchanged visible ten-test Slack deferral baseline.
- No product code was changed. The next step is human review of the plan and an
  explicit choice between subagent-driven or inline implementation.

## 2026-08-27 Review Queue HITL V2 product design

- Approved a focused Deliverable C design that connects the completed
  LangGraph checkpoint primitives to the real Review Queue without adding RAG,
  Neo4j, or Slack work.
- Kept the user journey within the existing Integrations and Review screens:
  sync evidence, explicitly create review candidates, inspect/resolve them in
  Review, then explicitly complete the review workflow.
- Defined a real `interrupt()` / same-thread `Command(resume=...)` graph whose
  resume acknowledgement never carries approval data; current PostgreSQL
  ReviewItem state and permission checks remain authoritative.
- Defined one locked Review transition service for single, bulk, and
  agent-candidate actions, with `source_review_item_id` provenance preventing
  duplicate knowledge and companion Timeline records after retries or races.
- Chose company/workspace-level ownership for an exact canonical source-version
  batch, so authorized users converge on one workflow instead of creating
  owner-specific duplicate candidates and knowledge effects.
- Added a V2-mode sync waterline so historical source versions already handled
  by the legacy path are not regenerated during the V1/V2 transition.
- Preserved disabled-by-default V2 routes and legacy rollback compatibility,
  while requiring legacy status to identify its checkpoint as metadata-only.
- Deferred CDC, transactional outbox, brokers, and streaming projection until
  measured scale or latency bottlenecks justify a separate design. The next
  step is an implementation plan and review, not product-code implementation.

## 2026-08-26 LangGraph runtime and checkpoint primitives

- Added JSON-safe, HMAC-keyed runtime contracts and application-owned workflow
  schema without exposing a new public workflow route.
- Separated explicit PostgreSQL checkpoint bootstrap from startup readiness;
  application startup never calls checkpointer setup.
- Defined SQLite/demo checkpoints as process-local memory mode and PostgreSQL
  checkpoints as durable mode, with fail-closed production behavior instead
  of memory fallback.
- Verified a real PostgreSQL interrupt, pool shutdown, independent pool
  restart, and same-thread `Command(resume=...)` at root checkpoint namespace
  on an isolated disposable test database.
- Hardened the proof with a fail-before-mutation test database identity guard,
  direct inspection of every stored/decoded checkpoint payload shape for
  relationship paths, LLM prompts, and raw connector payloads, and independent
  best-effort cleanup of the generated thread and every database resource.
- Locked the Review graph contract at `company-memory-review-v2.0` and the
  application migration at `2f6a8b9c0d1e`.
- Hardened saver confirmation to compare the exact ordered returned and
  persisted interrupt ids/JSON-safe values, while translating resume saver
  connection or deserialization failures to the bounded
  `checkpoint_unavailable` category.
- Bounded cyclic and excessively deep interrupt normalization to the opaque
  `checkpoint interrupt state mismatch` category, preventing raw recursion
  failures from escaping the runtime boundary.
- Verification recorded zero new non-Slack backend failures. The focused suite
  passed 224 tests, the non-Slack gate passed 717 tests with one existing
  optional pgvector skip, and the full suite retained exactly the ten visible
  user-deferred Slack failures.
- Deliverable C remains a separate, unimplemented Review Queue HITL V2 slice
  that requires review of its own plan before coding begins.

## 2026-08-26 LangChain·LangGraph dependency compatibility

- Upgraded and locked the approved dependency lines: LangChain 1.3.17,
  LangGraph 1.2.11, langchain-openai 1.6.0,
  langchain-google-genai 4.3.5, and
  langgraph-checkpoint-postgres 3.1.2.
- Added no-network compatibility coverage for the current OpenAI/Gemini
  constructors, LangChain structured output and `create_agent`, typed LangGraph
  runtime context, `interrupt()` / `Command(resume=...)`, and PostgreSQL saver
  imports.
- The focused AI integration suite passed. Complete backend comparison recorded
  old lock `506 passed, 11 failed, 1 skipped` and accepted lock `518 passed,
  11 failed, 1 skipped`; the 11 failures are identical pre-existing failures,
  so this deliverable introduced zero new backend failures. This deliverable
  changes no production route, graph topology, Review Queue behavior, or RAG
  behavior.

## 2026-08-26 LangChain·LangGraph runtime foundation design

- Audited the installed architecture and confirmed that ParaWorks calls real
  LangChain and LangGraph APIs, while identifying that its current graph is
  still a linear wrapper without durable checkpointing or true HITL resume.
- Chose a focused runtime refactor before Neo4j GraphRAG so keyword, pgvector,
  and the future Neo4j retriever share typed LangGraph routing and the same
  permission/evidence contracts.
- Defined real Review Queue interruption with `interrupt()`, PostgreSQL-backed
  checkpoints, `Command(resume=...)`, and fail-closed permission/citation
  validation.
- Kept Neo4j, graph projection, and `neo4j-graphrag` implementation out of this
  first slice so the runtime foundation remains independently testable and
  reversible.
- Re-reviewed the design against current LangGraph persistence semantics and
  the ParaWorks trust boundary, then split delivery into four independently
  verifiable stages instead of one broad refactor.
- Added explicit thread ownership, DB-enforced idempotency, checkpoint failure
  reconciliation, strict checkpoint serialization, graph versioning, legacy
  route compatibility, bounded hidden-match semantics, and server-derived
  citation rules.
- A final adversarial review corrected the checkpointer contract: application
  graph versions no longer misuse `checkpoint_ns`, initial/resume invokes use
  synchronous durability, and application status changes only after the saved
  tuple is confirmed.
- Hardened the trust boundary with canonical source/version-only Review inputs,
  keyed fingerprints, short claim leases around out-of-transaction model work,
  payload-safe idempotency keys, and DB-unique promotion provenance on every
  knowledge table including generated Timeline rows.
- Replaced "cite the whole evidence window" with structured answer blocks that
  reference server-issued evidence slots. Only selected, revalidated canonical
  records become public citations, and RAG V2 remains shadow-only until its
  faithfulness evaluation matches or exceeds the legacy baseline.
- After user approval of the revised trust-boundary contract, converted the
  first green slice into a separate execution plan for dependency
  compatibility. It fixes exact supported minor lines, a targeted uv lock
  refresh, no-network provider/structured-output/agent tests, LangGraph
  runtime-context and interrupt smoke coverage, and a hard stop before any
  runtime behavior change.

## 2026-05-16 Docker Postgres port fallback

- Fixed the production-like Docker helper so a non-ParaWorks listener on
  `127.0.0.1:5432` now falls back to the next available host port starting at
  `5433` instead of retrying the same occupied port.
- Aligned the pgvector dev helper and runbook with the same available-port
  behavior, preserving the compose `PARAWORKS_POSTGRES_PORT` override path.
- Repeated helper runs now reuse an existing ParaWorks Postgres host port
  instead of drifting from `5433` to higher ports.
- Verification: regression checks failed before the script fix, then passed
  through direct static test execution; PowerShell parser checks passed for both
  helper scripts; the Docker database path and full backend/frontend startup
  were verified locally.

## 2026-05-16 Dashboard SaaS responsive polish

- Reworked the Dashboard hero into a compact AI workspace card with separate
  text and right-side collaboration mock illustration areas, preserving the
  existing Korean copy and dashboard data flow.
- Changed the Dashboard content layout so the right utility column sits beside
  the main content only on wide screens, then flows below the main stack on
  laptop/tablet widths without squeezing the hero or KPI cards.
- Hardened the Calendar card hover popover and selected-date behavior so
  current date and event-bearing selected date can diverge safely.
- Verification: Dashboard Playwright workflow passed, frontend lint passed with
  existing timeline warnings only, frontend production build passed, and
  viewport measurements passed for 2560, 1920, 1440, 1366, 1024, and 768 widths.

## 2026-05-16 Dashboard calendar week order polish

- Changed the Dashboard calendar from Monday-start weeks to Sunday-start weeks
  so the grid reads Sunday through Saturday.
- Split today's visual state from the selected date state: today stays visible
  with a softer highlight when another date is selected, while the selected date
  keeps the stronger gradient emphasis.
- Verification: Dashboard Playwright workflow passed, frontend lint passed with
  existing timeline warnings only, and frontend production build passed.

## 2026-05-16 Dashboard calendar sync and Review bulk actions

- Extended the Dashboard API with `calendar_events` so synced Google Calendar
  events outside today can appear in the calendar grid, while `today_events`
  continues to drive the "today schedule" KPI.
- Made the Dashboard listen to the shared Review Queue update event so the
  review badge count shown in dashboard content refreshes with the sidebar.
- Hardened connector sync so duplicate source events with unchanged content
  signatures are not passed into ingestion or downstream review extraction.
- Added Review Queue bulk selection with a Gmail-style top checkbox, project
  selection before bulk processing, duplicate/similar bulk actions, right-click
  approve/reject actions, and an in-app confirmation modal.
- Verification: backend dashboard/ingestion/review tests passed with 33 tests,
  ruff passed for touched backend files, Review/Dashboard Playwright tests
  passed with 3 tests, frontend lint passed with existing timeline warnings
  only, and frontend production build passed.

## 2026-05-16 Review bulk action UX follow-up

- Moved group-level selection into the former expand-chevron position and
  removed the visible chevron affordance from review group headers.
- Scoped duplicate/similar approve and reject actions to each duplicate group,
  placing those actions beside the group confidence summary instead of in the
  global toolbar.
- Rendered review context menus and bulk confirmation dialogs through a body
  portal so modal backdrops cover the full viewport.
- Fixed the bulk approval failure warning copy so skipped project-unclassified
  items show readable Korean guidance.
- Verification: Review bulk Playwright coverage passed with 2 tests, frontend
  lint passed with existing timeline warnings only, and frontend production
  build passed.

## 2026-05-16 Timeline calendar status and filter cleanup

- Calendar-backed timeline items whose event time is already in the past now
  render as completed in the Timeline page, matching user expectations for
  historical schedule entries.
- Removed the unavailable `reviewing` status filter and the generic `Source`
  source filter option from the Timeline filter controls.
- Cleaned up unused Timeline icon imports so frontend lint is quiet.
- Verification: Timeline Playwright coverage passed with 3 tests, frontend
  lint passed with no warnings, and frontend production build passed.

## 2026-05-16 Review Mail Docs Calendar source labels

- Improved `/review` so Mail/Docs agent candidates show source-family badges
  (`Mail`, `Docs`, `Calendar`, or combined labels such as `Mail + Docs`) in
  the card header instead of always showing `Mail/Docs Agent`.
- Hardened Review API source-evidence fallback so Mail/Docs/Calendar rows keep
  indexed `source_types` and source-id evidence summary metadata instead of
  defaulting missing source type data to Slack.
- Verification: added backend regression coverage for indexed source-type
  fallback and Playwright coverage for the card-level source badges.

## 2026-05-15 Google Calendar updatedMin fallback

- Fixed Google Calendar sync recovery when Google rejects an old per-calendar
  `updatedMin` cursor with `The requested minimum modification time lies too far
  in the past`.
- The connector now refetches only the affected calendar through the existing
  initial window instead of failing the whole Calendar sync, preserving
  evidence-first ingestion and duplicate-skip cost controls.
- Verification: Calendar connector regression test was observed failing before
  the fix, then `test_google_connector.py` passed with 30 tests; connector
  ingestion plus Google connector tests passed with 41 tests; ruff passed on the
  touched connector/test files.

## 2026-05-15 Dashboard Calendar today events visibility

- Connected synced Google Calendar `Source` rows to the Dashboard `today_events`
  API response so today's events appear in the existing "today schedule" panel
  without waiting for Review Queue promotion.
- Kept the trust boundary intact: raw Calendar events are shown only as schedule
  visibility, while Calendar-derived todos still appear in today's work list
  only after approval into trusted `Todo` rows.
- Updated the dashboard UI and API type contract so the top schedule metric and
  right-side schedule panel use real Calendar event data.
- Verification: dashboard backend tests passed with 4 tests, ruff passed for
  touched backend files, dashboard Playwright passed, frontend lint completed
  with existing timeline warnings only, and frontend production build passed.

## 2026-05-15 대시보드 오늘 할 일 및 담당 프로젝트 개선

- 대시보드의 `오늘 해야 할 업무`가 검토 대기 todo 후보가 아니라, 승인된 todo ReviewItem 중 오늘(Asia/Seoul 기준) 이후 마감인 항목을 가까운 마감일 순으로 표시하도록 수정했다.
- 완료 버튼은 서버의 trusted knowledge를 변경하지 않고, 현재 대시보드 화면에서만 해당 항목을 숨긴다.
- `내 담당 프로젝트`가 빈 배열로 고정되어 있던 문제를 고쳐, 등록 프로젝트의 근거 수, 활동 수, 검토 대기 수가 대시보드에 표시되도록 연결했다.
- 검증: 대시보드 backend 테스트 3개 통과, ruff 통과, frontend lint/build 통과, Playwright 대시보드 업무 흐름 1개 통과.

포트폴리오 관점:

- Review에서 승인된 업무가 오늘의 실행 목록으로 연결되고, 프로젝트별 활동 상태가 첫 화면에 드러나도록 하여 “검토된 회사 기억이 실제 업무 홈으로 이어지는” 흐름을 강화했다.

## 2026-05-15 Google Calendar all-calendars project memory path

- Google Calendar sync now covers every accessible calendar instead of only the
  primary calendar, with per-calendar cursors and collision-safe Calendar source
  ids.
- Calendar evidence now flows through the same evidence-first path as
  Slack/Gmail/Drive: sync, Source evidence, AgentRun, ReviewItem, approval,
  Projects/Timeline, and approval-gated RAG indexing.
- Approved Calendar timeline entries now use the actual event start time for
  project activity and timeline grouping.

Portfolio angle:

- Shows the company-memory platform handling schedules as first-class reviewed
  evidence across all user calendars, while keeping Calendar inside the
  Mail/Docs ownership lane and preserving the human review boundary.

## 2026-05-15 프로젝트 근거 기본 선택 및 Slack 원문 시각 보강

- 프로젝트/타임라인 탭이 최신 생성 프로젝트를 무조건 기본 선택해, 승인 근거가 있는 프로젝트가 있어도 빈 프로젝트가 먼저 보이던 문제를 수정했다.
- 프로젝트 탭은 승인된 원본 근거, 활동, 타임라인이 있는 첫 프로젝트를 기본 선택하고, 타임라인 탭은 승인된 타임라인 항목이 있는 첫 프로젝트를 기본 선택한다.
- Slack source URL이 Source에 매칭되더라도 `raw_metadata.ts`가 비어 있으면 `Source.created_at`보다 Slack permalink의 `p...` timestamp를 먼저 사용한다.
- 실제 Docker DB에서 `project-paraworks-mvp`가 원본 근거 12건과 타임라인 6건을 계산하고, 승인 시각(`created_at`)과 원문 시각(`occurred_at`)이 분리되는 것을 확인했다.
- 검증: 프로젝트 메모리/Review backend 47개 통과, ruff 통과, frontend lint/build 통과, 핵심 Playwright 6개 통과.

포트폴리오 관점:

- 사용자가 만든 빈 프로젝트와 승인 데이터가 쌓인 프로젝트가 함께 있어도, 데모 첫 화면에서 실제 가치가 있는 프로젝트 근거와 활동이 바로 보이도록 개선했다.

## 2026-05-15 Gmail/Drive 프로젝트 라우팅 승인 연결

- `backend/app/agent_runtime/project_routing.py`에 Gmail, Drive, Slack이 함께 쓸 수 있는 공용 프로젝트 라우팅 계약을 추가했다.
- Mail/Document Agent가 만든 `llm_tool` 기반 ReviewItem은 프로젝트가 확정되지 않았거나 사용자 선택이 필요한 상태이면 승인할 수 없도록 Review 승인 정책을 강화했다.
- Review 화면에서 프로젝트 미선택 Gmail/Drive 후보는 "프로젝트 선택 후 승인 가능" 안내와 함께 승인 버튼이 비활성화되고, 등록 프로젝트를 선택하면 같은 ReviewItem을 승인할 수 있다.
- 승인된 Gmail/Drive 후보는 기존 Review Queue 신뢰 경계를 거쳐 Timeline/Projects에 프로젝트별 활동으로 표시되고, 승인된 source chunk와 approved knowledge는 RAG indexing 대상에 포함된다.
- approved knowledge의 벡터 문서 메타데이터에 `project_key`를 보존해 프로젝트 기반 검색/분석으로 이어질 수 있게 했다.
- 검증: 공용 라우팅/Review/Project/RAG 백엔드 테스트 74개 통과, ruff 통과, 프론트 lint/build 통과, Gmail/Drive Review -> Timeline -> Projects Playwright desktop/mobile 2개 통과, 기존 Review project routing Playwright desktop/mobile 2개 통과.

Portfolio angle:

- Gmail/Drive 증거가 AI 후보에서 끝나지 않고, 프로젝트 선택과 사람 승인 후 회사 기억, 프로젝트 활동, 타임라인, RAG 색인까지 이어지는 제품 루프를 보여준다.
## 2026-05-15 Slack 프로젝트 Tool Routing 승인 경계 완성

- Slack 신규 sync에서 규칙 기반 `project_assignment` 생성을 중단하고, Slack Agent의 LangChain tool routing 결과만 프로젝트 지정 근거로 사용하게 했다.
- `topic_tag` fallback으로 프로젝트가 자동 지정되던 경로를 제거해, LLM router가 프로젝트를 확정하지 못한 후보는 사용자가 직접 프로젝트를 선택해야 승인할 수 있게 했다.
- Review Queue의 promotion preview와 approve API가 Slack Agent `llm_tool` 후보의 `project_key` 누락을 승인 불가로 반환한다.
- Review 화면에 `프로젝트 선택 후 승인 가능` 안내를 추가하고, Timeline은 날짜 단위 그룹으로, Projects metric은 모바일 겹침 없이 표시되도록 보강했다.
- 검증: backend targeted 65 passed, ruff 통과, frontend TypeScript/lint/build 통과, Playwright 4개 시나리오 desktop/mobile 총 8 passed.

포트폴리오 관점:

- Slack 원본 수집, LLM 기반 프로젝트 판단, Human Review 승인 경계, 프로젝트별 타임라인 표시가 하나의 신뢰 가능한 제품 흐름으로 연결되었다.

## 2026-05-15 Slack 프로젝트 Router Tool Agent 추가

- 기존 `agent_slack` LangGraph 흐름에 `project_route` 노드를 추가했다.
- Slack Agent는 업무 후보를 추출한 뒤 LangChain tool-calling 기반 프로젝트 router를 실행해, 등록 프로젝트 중 연결 후보를 고르고 프로젝트 활동 요약과 근거를 생성한다.
- Router는 `list_registered_projects`, `score_project_candidates` tool을 사용하며, 결과는 승인 전 trusted knowledge가 아니라 `ReviewItem.payload`에 `project_assignment_method=llm_tool`, 요약, 근거, 확신도, 대체 후보로 보존된다.
- Slack LLM project routing으로 후보가 생성된 sync에서는 기존 deterministic `project_assignment` 중복 생성을 건너뛰되, Agent 후보가 새로 생성되지 않은 no-op sync에서는 fallback 분류를 막지 않도록 조건을 좁혔다.
- Review 화면은 LLM 프로젝트 분류 요약과 연결 근거를 표시하고, 사용자가 등록 프로젝트 select에서 프로젝트를 바꾼 뒤 승인할 수 있는 흐름을 유지한다.
- 현재 Slack Agent LangGraph 문서를 `agent_slack/slack_agent_langgraph.md`에 추가했다.
- 검증: 관련 백엔드 회귀 테스트 60개 통과, ruff 통과, 프론트엔드 TypeScript/lint/build 통과, Playwright review project routing desktop/mobile 2개 통과.

포트폴리오 관점:

- Slack 원본 대화가 “업무 후보 추출”에서 끝나지 않고, 등록 프로젝트 맥락에 맞게 요약/근거와 함께 human-in-the-loop 검토로 넘어가는 흐름을 보여준다.
- 규칙 기반 매칭의 오탐을 줄이면서도 LLM 제안을 사용자가 검토하고 수정할 수 있어, 프로젝트별 활동 타임라인의 신뢰도를 높인다.

## 2026-05-15 Slack 장시간 동기화 실패 오인 수정

- 실제 Playwright로 `/integrations` Slack 동기화를 눌러 네트워크 응답과 runtime-status를 확인했다.
- 확인 결과 최신 job은 `complete`, `last_error=null`이었고, 과거 대량 sync는 약 153초가 걸려 프론트 polling 한도 135초를 넘길 수 있었다.
- 프론트가 이 timeout을 실제 backend failure와 같은 `동기화 실패` 모달로 표시하던 문제를 수정했다.
- 120초 이상 진행 중이거나 polling 한도를 넘긴 정상 running job은 `백그라운드에서 계속 진행 중입니다` 안내로 표시하고, 실패 상태로 오인하지 않게 했다.
- 검증: Playwright 동기화 모달 회귀 desktop/mobile 6개 통과, TypeScript/lint/build 통과.

## 2026-05-14 사용자 정의 프로젝트 동기화 검토사항 수정

- 사용자가 프로젝트를 생성하면 기존 Slack/Gmail/Drive/Calendar source를 즉시
  해당 프로젝트 기준으로 분류하고, 근거가 매칭될 때 `project_assignment`
  ReviewItem을 `pending_review` 상태로 생성하도록 수정했다.
- connector sync 이후에도 같은 프로젝트 분류기를 실행해 새로 동기화된 source와
  이미 동기화되어 skipped 처리된 기존 source가 별도 재분류 호출 없이 프로젝트
  연결 검토사항으로 들어오도록 했다.
- 프로젝트 요약, 근거 사유, 승인 타임라인 사유, source 라벨에 남아 있던 깨진
  한글 fallback 문구를 읽을 수 있는 한국어 문장으로 교체했다.
- 검증: 프로젝트/동기화/검토 관련 백엔드 테스트 37개 통과, ruff 통과,
  프론트엔드 TypeScript 검사와 프로덕션 빌드 통과.

포트폴리오 관점:

- 사용자 생성 프로젝트, connector 근거, Agent 제안, 수동 승인이 Review Queue에서
  만나는 흐름을 구현해 프로젝트 탭이 실제 human-in-the-loop 라우팅 화면으로
  작동하도록 만들었다.

## 2026-05-14 Approved Project Timeline and RAG Visibility Fix

- Fixed the approved Review Queue to Project/Timeline display path so promoted
  Decision, History, Timeline, and Todo records preserve `project_key` through
  the shared `/api/v1/projects` response.
- Updated the project workspace to show approved workflow items alongside
  connector assignment evidence, while keeping Review Queue approval as the
  trust boundary.
- Repaired the approval-based RAG source chunk path by preserving Mail/Document
  `source_ids`, `source_types`, `source_urls`, and `source_authors` in ReviewItem
  payloads.
- Aligned RAG indexing tests with the approval-only policy: approved knowledge
  records are indexed, and original email/document chunks enter RAG only when
  their external `Source.source_id` appears in an approved ReviewItem payload.
- Verification: targeted project/review/mail-document/RAG backend tests passed
  with 44 passed; frontend TypeScript check and production build passed.

Portfolio angle:

- Shows the human-review loop becoming product-visible and retrieval-ready:
  approved evidence-backed work history now appears in project workflow,
  timeline views, and RAG indexing without indexing unapproved synced content.

## 2026-05-14 Project Recognition and Timeline Workflow Boundary

- Replaced loose `/projects` source grouping with two canonical company
  projects: `K테크 파일럿` and `시드 투자 IR`.
- Added deterministic, zero-token project classification that creates
  `project_assignment` Review Queue candidates from Slack, Gmail, Drive, and
  Calendar source evidence.
- Changed `/projects` to show only approved project assignments, so legacy
  smoke/demo labels such as `Project Newbiegenie`, `프로젝트 결과`, and
  `미분류 프로젝트` no longer appear as business projects.
- Updated `/timeline` to consume project-scoped workflow data from `/projects`
  instead of showing one generic `Company Memory` timeline.
- Stopped using raw connector titles like `Slack message in C0AUJDZUKA8` as
  task titles and moved project evidence reasons into the UI data model.
- Removed the deterministic RAG Redis canned-answer branch so local fallback
  answers now summarize retrieved evidence snippets instead of returning a
  fixed Redis/PostgreSQL response.
- Added a local/dev reset utility for connector-derived data that preserves
  auth users and integration connections while clearing sources, review items,
  approved knowledge, vector state, AgentRuns, and assistant conversations.

Portfolio angle:

- Shows ParaWorks moving from connector-data display toward evidence-backed
  project understanding and evidence-grounded assistant answers with human
  review as the trust boundary.
  project understanding with human review as the trust boundary.

## 2026-05-12 Smoke-Only Demo Data Boundary

- Changed the default runtime posture so demo/mock content is not seeded unless
  `PARAWORKS_SEED_DEMO_DATA=true`.
- Kept `scripts/start-smoke.ps1` as the smoke/demo entrypoint that explicitly
  enables demo mode and demo seed data.
- Updated the pgvector Docker-backed dev startup path to run with
  `PARAWORKS_DEMO_MODE=false` and `PARAWORKS_SEED_DEMO_DATA=false`, so an empty
  database remains empty until Slack or Google connectors are installed and
  synced.
- Stopped the Review page from showing local fallback review items when the API
  cannot return real items.
- Changed Dashboard, Projects, and Timeline sample-only surfaces to render empty
  states in production-like mode instead of hard-coded ORION/Nova/Atlas data.
- Verification: `uv run pytest backend/tests -v` passed with 297 passed and 1
  skipped; `npm run build` passed from `frontend`.

## 2026-05-12 AI Assistant Conversation Memory

- Added database-backed, per-user AI assistant conversations for the `AI 비서`
  surface.
- Persisted user and assistant messages with citations, source snippets,
  permission notices, hidden-source counts, and linked AgentRun ids.
- Kept token, cache, and cost details out of the user-facing assistant flow so
  cost observability remains in `Agent Runs`.
- Added regression coverage for user-scoped assistant conversations and the
  `/search` assistant UX.

Portfolio angle:

- Shows ParaWorks evolving from one-shot RAG search into a product-like
  evidence-backed AI assistant with memory, permission safety, and operator
  observability.

## 2026-05-12 AI Assistant Chat UX and Sincere-mode RAG LLM

- Reworked `/search` into a more natural chat surface with compact conversation
  history, duplicate empty-chat prevention, bottom composer, and folded
  evidence/source panels inside each assistant response.
- Continued the chat polish on 2026-05-13 by keeping history ordered by latest
  updated conversation rather than selected conversation, constraining scrolling
  to the transcript pane, removing message badges, rendering assistant markdown,
  adding copy actions, and adding one-click suggested prompts.
- Shortened chat history titles from the first user message so the history list
  behaves like a conversation list instead of a document summary list.
- Added a RAG LLM adapter for non-demo 진심모드:
  - OpenAI primary model: `gpt-5.4-mini`;
  - `.env` `AGENT_LLM_OPENAI_MODEL` as the OpenAI fallback model;
  - provider fallback through the configured provider order.
- Kept demo mode deterministic so tests and cheap demos do not call live LLMs.
- Updated the session handoff runbook with the active branch and local
  continuation notes.

Portfolio angle:

- Shows ParaWorks moving from a technical RAG answer page toward a credible
  AI assistant product surface while preserving evidence, permissions, and
  operator cost observability.

## 2026-05-11 Sidebar and Workspace Navigation Update

- Simplified the main sidebar by removing separate Decision, History, and
  Knowledge Map entries.
- Added a Project page where users can switch between assigned projects from a
  top project menu and review progress, risk, pending review count, Gantt-style
  planning, calendar scheduling, board status, and task lists.
- Kept the Timeline entry as `타임라인` and made the page project-scoped, with
  history summaries tied back to source Slack/Gmail/Drive/Calendar evidence.
- Updated the global top search so the left search icon acts as the submit
  button and routes to `AI 비서` with the query prefilled.
- Renamed the assistant surface to `AI 비서` in navigation and page copy.
- Restored Dashboard review visibility with a `검토사항` section and renamed the
  Review page heading from `검토 큐` to `검토사항`; when the backend is not
  reachable, demo review items remain visible instead of leaving the page in a
  loading state.
- Updated Timeline history interactions so the history icon opens a source-
  specific history panel only on demand, then collapses back to a full-width
  timeline when closed.
- Rebuilt the Dashboard as a personalized work home focused on today's assigned
  tasks, review items, meetings, mentions, assigned projects, and suggested Ask
  prompts instead of workspace-wide ingestion metrics.
- Moved source collection status into Integrations, where connector operations
  and source health belong.
- Verification: `npm run build` passed from `frontend`.

Portfolio angle:

- Shows ParaWorks moving from many knowledge-category pages toward an
  operator-friendly workspace organized around projects, timelines, source
  history, and evidence-backed Ask.

## Portfolio Positioning

ParaWorks is a Korean-first, multi-agentic Slack-style collaboration and
knowledge review workspace for business users. The MVP demonstrates how work
messages, SaaS connector events, and review workflows can become searchable,
permission-aware organizational knowledge.

## Current Narrative

The project started as an Adapter-First Demo Harness and evolved into a more
product-shaped MVP:

- A Korean-first business UX with English switching for international-ready use.
- A Slack-like Messenger surface for team conversation and collaboration.
- A review queue that turns selected messages and connector evidence into
  human-confirmed knowledge candidates.
- A Docker-free SQLite smoke mode so the product can be run and demonstrated
  quickly even when Postgres, Redis, and MinIO are unavailable.
- A Slack connector boundary prepared for future real Slack Web API ingestion.

## Work Completed

### Korean-First UX and Messenger MVP

- Added Korean default shell copy and a Korean/English language switch.
- Added `/messages` as a Slack-like messenger screen with channels, timeline,
  and message composer.
- Added backend message APIs for listing channels, listing messages, and
  posting messages.
- Fixed SSE completion behavior so successful sync completion is not shown as
  a stream error.

Portfolio angle:

- Shows product localization for the first target market: Korean business users.
- Demonstrates moving beyond a technical harness into a familiar collaboration
  experience.

### SQLite Smoke Mode

- Added `scripts/start-smoke.ps1` for Docker-free local demos.
- Added a smoke runbook and updated local development and verification docs.
- Smoke mode starts FastAPI and Next.js against a temporary SQLite database.

Portfolio angle:

- Shows practical engineering for demo reliability and onboarding speed.
- Reduces environment friction, which matters for stakeholder demos and hiring
  portfolio walkthroughs.

### Messenger Persistence

- Added SQLAlchemy models for `message_channels` and `messages`.
- Changed message service behavior from process memory to database-backed
  persistence.
- Seeded demo channels and messages on first use against an empty database.

Portfolio angle:

- Shows the transition from UI prototype to stateful MVP infrastructure.
- Establishes a foundation for analytics, search, review, and audit history.

### Messenger to Review Queue

- Added `POST /api/v1/messages/messages/{message_id}/send-to-review`.
- Added UI action to send a message into the review workflow.
- Created review items with message snippets and `paraworks://messages/...`
  source links.

Portfolio angle:

- Connects the Slack-like messenger directly to ParaWorks' knowledge workflow.
- Shows a concrete "conversation to organizational knowledge" product loop.

### Slack Connector Preparation

- Added `backend/app/connectors/slack.py` with a testable `SlackApiClient`
  protocol boundary.
- Added tests for Slack message payload mapping into ParaWorks `SourceEvent`
  records.
- Added Slack integration runbook and environment placeholders:
  `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_IDS`, `SLACK_WORKSPACE_URL`.

Portfolio angle:

- Shows adapter-first architecture: real SaaS APIs can be connected without
  coupling product logic to vendor SDK details.
- Sets up future real Slack ingestion with pagination, rate limits, OAuth, and
  permission mapping.

## Verification Evidence

Latest known verified state:

- Backend tests: `uv run pytest backend/tests -v` -> 27 passed.
- Frontend build: `npm.cmd run build` -> passed.
- Smoke runtime:
  - `http://127.0.0.1:3000/messages`
  - `http://127.0.0.1:3000/dashboard`
  - `http://127.0.0.1:3000/review`
- Browser smoke covered message posting, language switching, sending a message
  to review, and confirming the created review item on `/review`.

## Portfolio Demo Script

1. Start the SQLite smoke environment with `.\scripts\start-smoke.ps1`.
2. Open `/dashboard` to show the product overview.
3. Open `/messages` to show Korean-first Slack-like collaboration.
4. Post or select a message and send it to the review queue.
5. Open `/review` to show the message as a knowledge review candidate.
6. Explain that connector data and messenger data share the same review/search
   architecture.
7. Point to the Slack connector boundary as the next real-world integration
   step.

## Next Portfolio-Worthy Milestones

- Implement `RealSlackApiClient` with Slack Web API cursor pagination and
  rate-limit handling.
- Add Slack OAuth install flow and token storage decisions.
- Map Slack channel/private-message permissions into ParaWorks review/search
  access rules.
- Add message-to-knowledge actions such as "create decision record" and
  "create todo".
- Add focused frontend regression tests for Messenger and review actions.

## Product North Star Update: Multi-Agent Knowledge Automation

Recorded on 2026-05-01.

ParaWorks' final target is an AI-agentic company memory platform. The agent
system will use multi-agent orchestration with LangChain 1.x and LangGraph 1.x
to automate:

- Email summarization, review, timeline creation, and history creation.
- Slack channel/message summarization, review, timeline creation, and history
  creation.
- RAG over generated timelines/history and internal company documents.
- Permission-aware answers for Korean business users.

The first recommended vertical slice is:

Slack channel/messages -> agentic summary and review -> timeline/history
candidates -> Review Queue.

Portfolio angle:

- Shows the transition from MVP collaboration surface to AI orchestration
  platform.
- Makes token cost optimization a product and architecture requirement from the
  start, not a late performance cleanup.
- Supports a three-developer split across agent runtime, data/RAG, and product
  UX.

## Collaboration Guide Update: Equal Agent Ownership

Recorded on 2026-05-01.

Added `AGENTS.md` as the repo-level collaboration guide for human developers
and coding assistants. The guide changes the team split from technical layers
to equal agent ownership:

- Slack Agent.
- Mail and Document Agent.
- RAG and Orchestrator Agent.

It also codifies evidence-first AI output, Review Queue as the trust boundary,
permission propagation, token-cost accounting, fake-LLM testing, and assistant
behavior rules for Codex, Claude Code, Gemini, and similar tools.

Portfolio angle:

- Shows that ParaWorks is being developed as a serious multi-agent system with
  team-scale engineering discipline.
- Makes AI safety, cost optimization, permissions, and human review explicit
  development compliance requirements.

## Integration Pipeline Update: Assistant-Safe Merging

Recorded on 2026-05-01.

Updated `AGENTS.md` with a development pipeline for three independent agent
tracks and coding assistants:

- shared contract branch first;
- feature branch per agent;
- integration branch for frequent green merges;
- contract tests as the merge gate;
- human decision points for schema, permission, cost, trust-boundary, and
  duplicate-resolution policy changes.

Portfolio angle:

- Demonstrates that ParaWorks is designed for AI-assisted team development, not
  only AI-powered product features.
- Shows awareness that Codex can help resolve conflicts, but stable contracts,
  registry-based integration, and verification gates must exist before the
  merge.

## UX Direction Update: Slack-Like Workspace

Recorded on 2026-05-01.

Started a frontend UX pass to make ParaWorks feel like a Korean-first
Slack-like business workspace rather than a plain demo harness.

Planned scope:

- darker workspace navigation rail;
- top command/search entry for future AI/RAG usage;
- richer Messages channel surface;
- Tools/Apps-style Integrations page;
- cleaner Korean-first operational copy.

Implemented scope:

- redesigned the shared app shell into a Slack-like workspace rail;
- added a top command/search bar and agent readiness affordance;
- upgraded Messages with denser channel navigation, timeline styling, review
  actions, and an anchored composer;
- upgraded Integrations into a Tools/Apps surface with connector readiness and
  sync activity panel.

Verification evidence:

- `npm.cmd run build` from `frontend` passed without warnings.
- HTTP smoke returned 200 for `/integrations`, `/messages`, and `/dashboard`
  on `http://127.0.0.1:3000`.

Portfolio angle:

- Shows product sense and UX architecture, not only backend AI engineering.
- Prepares the interface for Slack Agent, Review Inbox, and RAG Orchestrator
  experiences without a later full layout rewrite.

## Agent Development Update: Slack Agent Skeleton

Recorded on 2026-05-01.

Started the first functional agent track after the shared runtime and registry
contracts. The Slack Agent skeleton is scoped to:

- accept shared `EvidencePacket` input;
- use a fake/model client boundary instead of live LLM calls;
- return `AgentRunResult`;
- preserve source links, snippets, and strictest permission level;
- record token/cost metadata.

Portfolio angle:

- Shows the project moving from architecture contracts into actual agent
  implementation.
- Keeps the work merge-friendly because Slack Agent lives in its own owned
  package and integrates only through shared runtime contracts.

## Agent Development Update: Slack Agent Review Bridge

Recorded on 2026-05-01.

Started the bridge that turns persisted Slack source chunks into shared
`EvidencePacket` input, runs the Slack Agent, and persists agent output as
`ReviewItem(status="pending_review")`.

Portfolio angle:

- Shows the first real product loop for agentic Slack knowledge extraction:
  source evidence -> agent runtime -> human review.
- Preserves the project compliance story by carrying source links, snippets,
  permission level, prompt version, cache key, and token/cost metadata into the
  Review Queue.

## Agent Development Update: Slack Agent API and UI

Recorded on 2026-05-01.

Next milestone is exposing the Slack Agent Review bridge through the product:

- backend endpoint for Slack Agent Review generation;
- deterministic demo model instead of live LLM calls;
- frontend Tools action to run the Slack Agent after mock Slack sync;
- activity panel result showing how many Review Queue items were created.

Portfolio angle:

- Turns agent architecture into a user-visible workflow.
- Demonstrates cost-safe AI development by using a deterministic local model
  boundary before enabling paid LLM provider calls.

## UX Update: Agent-Aware Review Inbox

Recorded on 2026-05-01.

Next UI milestone is upgrading `/review` from a generic review list into an
agent-aware activity inbox:

- AI Agent-generated candidates are visibly labeled.
- Prompt version, token usage, estimated cost, cache key, permission, and
  confidence become inspectable from the Review UI.
- Source Evidence Drawer gets clean Korean copy and clearer evidence links.

Portfolio angle:

- Makes the human-review trust boundary visible to users and interviewers.
- Connects the token-cost optimization requirement to a concrete product
  surface instead of keeping it hidden in backend metadata.

Implemented scope:

- Reworked `/review` into an Activity Inbox-style review queue.
- Replaced corrupted Korean UI copy.
- Added AI Agent badges for agent-generated Review Items.
- Exposed prompt version, token count, estimated cost, cache key, confidence,
  permission, and source evidence in the UI.
- Cleaned up the Source Evidence Drawer copy and layout.

Verification evidence:

- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted after build to clear stale Next.js cache.
- HTTP smoke returned 200 for `/review`, `/integrations`, and `/health`.

## Agent Runtime Update: AgentRun Cost Audit Model

Recorded on 2026-05-01.

Next backend milestone is persisting every agent execution as an `AgentRun` row
so token usage, estimated cost, prompt version, cache key, permission level, and
run status can be audited beyond the ReviewItem payload.

Portfolio angle:

- Shows that token-cost optimization is backed by durable observability, not
  only UI labels.
- Creates the shared audit foundation needed by Slack Agent, Mail/Document
  Agent, and RAG/Orchestrator Agent.

Implemented scope:

- Added the `agent_runs` table and `AgentRun` model.
- Persisted one `AgentRun` row for each Slack Agent Review execution.
- Linked generated Review Items back to the originating agent run through
  `payload.agent_run_id`.
- Stored prompt version, cache key, model name, token usage, estimated cost,
  source window, permission level, and run metadata.

Verification evidence:

- `uv run pytest backend/tests/test_agent_run_model.py backend/tests/test_db_init.py -v`
  passed.
- `uv run pytest backend/tests -v` passed with 40 backend tests.

## Agent Development Update: Mail/Document Agent Slice

Recorded on 2026-05-01.

Next agent-track milestone is giving Developer B an independently owned agent
slice for Gmail and Drive evidence while preserving the same shared runtime
contract used by Slack Agent.

Portfolio angle:

- Demonstrates that ParaWorks is not a single hard-coded Slack demo; it now has
  a repeatable multi-agent backend pattern across communication and document
  sources.
- Shows practical 3-person division of labor: Slack Agent, Mail/Document Agent,
  and RAG/Orchestrator Agent can evolve with the same `EvidencePacket`,
  `AgentRunResult`, `AgentRun`, and Review Queue boundaries.

Implemented scope:

- Added `mail_document_agent` with manifest, model protocol, deterministic
  local model, and `MailDocumentAgent`.
- Added a bridge that builds evidence packets from Gmail and Drive chunks,
  excludes Slack chunks, persists `AgentRun`, and links Review Items through
  `payload.agent_run_id`.
- Added `POST /api/v1/integrations/mail-docs/agent-review` for deterministic
  MVP smoke testing without paid LLM calls.

Verification evidence:

- `uv run pytest backend/tests/test_mail_document_agent.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_mail_document_agent_api.py -v`
  passed.
- `uv run pytest backend/tests -v` passed with 44 backend tests.

## UX Update: Integrations Multi-Agent Actions

Recorded on 2026-05-01.

Next product milestone is making the second backend agent visible from the
same Integrations surface users already use for mock connector smoke testing.

Portfolio angle:

- Shows that the product can expose multiple independently owned agents without
  duplicating UI state or endpoint-specific response types.
- Makes the 3-person agent split tangible in the app: Slack Agent and
  Mail/Docs Agent can both be run from the Korean business-user workflow.

Implemented scope:

- Generalized the frontend agent-review response type to `AgentReviewResponse`.
- Replaced Slack-only action state with reusable agent action descriptors.
- Added Mail/Docs Agent buttons to Gmail and Drive cards.
- Kept Korean UX copy intact and displayed completed agent names in friendly
  labels.

Verification evidence:

- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-mail-docs-ui.db`.
- HTTP smoke returned 200 for `/health`, `/integrations`, and `/dashboard`.
- Gmail sync, Drive sync, and `POST /api/v1/integrations/mail-docs/agent-review`
  returned `agentName=mail_document_agent` and `created=1`.

## Agent Development Update: RAG Orchestrator Agent

Recorded on 2026-05-01.

Next core-product milestone is giving users a question-answering endpoint over
the company memory evidence that Slack, Gmail, Drive, and review workflows have
already collected.

Portfolio angle:

- Completes the three-track agent split: Slack Agent, Mail/Document Agent, and
  RAG/Orchestrator Agent now each have an independently testable backend slice.
- Shows a cost-safe RAG migration path: deterministic keyword retrieval now,
  vector DB and LangGraph orchestration later without changing the public answer
  contract.
- Demonstrates permission-aware RAG behavior by hiding restricted sources for
  viewer users while reporting hidden matches.

Implemented scope:

- Added `rag_orchestrator_agent` with manifest, deterministic model, answer
  dataclasses, and cost metadata.
- Added permission-aware retrieval over existing `DocumentChunk` evidence.
- Added `POST /api/v1/ask` returning answer text, source links, snippets,
  permission notices, cache key, model name, token usage, and estimated cost.

Verification evidence:

- `uv run pytest backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_ask_api.py -v`
  passed.
- `uv run pytest backend/tests -v` passed with 50 backend tests.

## UX Update: Company Memory Ask Workbench

Recorded on 2026-05-01.

Next product milestone is making the RAG Orchestrator visible to Korean
business users through the existing Search surface.

Portfolio angle:

- Turns the backend `/api/v1/ask` contract into an inspectable product workflow:
  question, AI answer, citations, raw matching evidence, permission notice, and
  cost metadata are visible together.
- Shows that ParaWorks treats RAG answers as auditable outputs, not opaque chat
  bubbles.
- Keeps the demo cost-safe by using the deterministic orchestrator while still
  exposing token and estimated-cost fields.

Implemented scope:

- Added frontend `AskResponse` type.
- Reworked `/search` into a Company Memory workbench.
- One query now calls both `/api/v1/ask` and `/api/v1/search` using viewer
  permissions.
- Rendered answer text, source links, token count, estimated cost, hidden
  match count, permission notice, cache key, model name, and raw evidence.

Verification evidence:

- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-ask-ui.db`.
- HTTP smoke returned 200 for `/health`, `/search`, and `/dashboard`.
- Gmail sync, Drive sync, and `POST /api/v1/ask` returned
  `agentName=rag_orchestrator_agent`, `sources=2`, `hidden=0`, and `tokens=100`.

## Observability Update: Agent Run Cost Dashboard

Recorded on 2026-05-01.

Next operations milestone is making AI execution cost and token usage visible
from the product, not only stored in the database.

Portfolio angle:

- Shows AI cost governance as a first-class product feature.
- Gives the three-agent split a shared observability surface: Slack Agent,
  Mail/Docs Agent, and future RAG runs can be compared through one audit table.
- Demonstrates a production-minded pattern where every agent run has prompt,
  model, token, cost, permission, and cache metadata.

Implemented scope:

- Added read-only `GET /api/v1/agent-runs`.
- Returned aggregate run count, total tokens, estimated total cost, and recent
  run details.
- Added frontend `AgentRunsResponse` and `AgentRunSummaryItem` types.
- Reworked `/dashboard` with Agent execution count, estimated cost, token total,
  and recent Agent Runs panel.

Verification evidence:

- `uv run pytest backend/tests/test_agent_runs_api.py -v` passed.
- `uv run pytest backend/tests -v` passed with 51 backend tests.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-agent-runs.db`.
- Slack Agent and Mail/Docs Agent smoke run produced `totalRuns=2`,
  `totalTokens=226`, and `estimatedCost=0.000063`.
- HTTP smoke returned 200 for `/health`, `/dashboard`, and `/search`.

## Observability Update: RAG AgentRun Persistence

Recorded on 2026-05-01.

Next observability milestone is ensuring the RAG Orchestrator participates in
the same AgentRun audit trail as Slack Agent and Mail/Docs Agent.

Portfolio angle:

- Completes the shared three-agent audit story: Slack extraction, Mail/Docs
  extraction, and RAG question answering all create durable cost records.
- Shows that every user-facing AI answer can be traced to prompt version, model,
  token usage, estimated cost, cache key, permission level, and source count.
- Strengthens the token-cost optimization requirement by making RAG asks visible
  in the same dashboard totals.

Implemented scope:

- Persisted one `AgentRun` for each `answer_question_with_rag` execution.
- Stored question text, source count, hidden match count, source type, and cache
  hit metadata.
- Kept the public `/api/v1/ask` response shape unchanged while allowing
  `/api/v1/agent-runs` and `/dashboard` to include RAG ask runs.

Verification evidence:

- `uv run pytest backend/tests/test_rag_orchestrator_service.py -v` passed.
- `uv run pytest backend/tests -v` passed with 52 backend tests.
- Smoke server restarted with `.tmp/paraworks-rag-agent-run.db`.
- Gmail sync, Drive sync, and `POST /api/v1/ask` produced
  `askAgent=rag_orchestrator_agent`, `askTokens=100`, `totalRuns=1`,
  `totalTokens=100`, and `latestQuestion=Redis job state`.
- HTTP smoke returned 200 for `/health`, `/dashboard`, and `/search`.

## Knowledge Update: Review Approval Promotion

Recorded on 2026-05-01.

Next product milestone is closing the human-review loop so approved agent
candidates become durable company memory records.

Portfolio angle:

- Completes the source evidence -> agent candidate -> human approval -> company
  memory loop.
- Shows that ParaWorks keeps human approval as the trust boundary before
  writing durable history, decision, and task records.
- Preserves the audit story by carrying source links, source snippets,
  confidence, permission level, and approved review status into knowledge
  tables.

Implemented scope:

- Added `promote_review_item` in `backend/app/knowledge/promotion.py`.
- Mapped `decision_record` Review Items into `DecisionRecord`.
- Mapped `history_event` Review Items into `HistoryEvent`.
- Mapped `todo` Review Items into `Todo`.
- Called promotion from the existing Review approve endpoint.

Verification evidence:

- `uv run pytest backend/tests/test_review_knowledge_promotion.py -v` passed.
- `uv run pytest backend/tests -v` passed with 55 backend tests.
- Smoke server restarted with `.tmp/paraworks-review-promotion.db`.
- Slack sync produced 3 pending Review Items, approving one returned
  `approvedStatus=approved` and `approvedType=todo`.
- HTTP smoke returned 200 for `/health`, `/review`, and `/dashboard`.

## Product Update: Knowledge Library

Recorded on 2026-05-01.

Next user-facing milestone is making approved company memory visible after
Review Queue approval.

Portfolio angle:

- Turns durable knowledge rows into an inspectable product surface.
- Shows the completed workflow from Slack evidence to Review approval to
  approved decisions, history, and todos.
- Provides a natural next step toward vectorizing approved company memory for
  production RAG.

Implemented scope:

- Added read-only `GET /api/v1/knowledge`.
- Returned approved decisions, history events, todos, counts, source evidence,
  confidence, permission, and review status.
- Added frontend `KnowledgeResponse` and `KnowledgeItem` types.
- Added `/knowledge` page with summary cards and evidence-preserving records.
- Added Knowledge navigation labels in Korean and English.

Verification evidence:

- `uv run pytest backend/tests/test_knowledge_api.py -v` passed.
- `uv run pytest backend/tests -v` passed with 56 backend tests.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-knowledge-library.db`.
- Slack sync and approving all 3 Review Items produced `decisions=1`,
  `history=1`, and `todos=1` from `/api/v1/knowledge`.
- HTTP smoke returned 200 for `/health`, `/knowledge`, `/review`, and
  `/dashboard`.

## RAG Update: Approved Knowledge Retrieval

Recorded on 2026-05-01.

Next retrieval milestone is allowing the RAG Orchestrator to answer from
human-approved company memory, not only raw source chunks.

Portfolio angle:

- Connects Knowledge Library records back into the user-facing Ask workflow.
- Shows the intended learning loop: raw evidence is reviewed, promoted into
  company memory, then reused as trusted RAG context.
- Keeps the permission story intact by applying hidden-match behavior to
  approved knowledge records as well as raw document chunks.

Implemented scope:

- Added `RagEvidenceCandidate` as a common retrieval candidate for raw chunks
  and approved knowledge.
- Added approved `DecisionRecord`, `HistoryEvent`, and `Todo` retrieval to the
  RAG Orchestrator service.
- Preserved source links and source snippets from approved knowledge records in
  `EvidencePacket`.
- Kept `/api/v1/ask` response shape unchanged.

Verification evidence:

- `uv run pytest backend/tests/test_rag_orchestrator_service.py backend/tests/test_ask_api.py -v`
  passed.
- `uv run pytest backend/tests -v` passed with 59 backend tests.
- Smoke server restarted with `.tmp/paraworks-knowledge-rag.db`.
- Slack sync and approving all 3 Review Items followed by `POST /api/v1/ask`
  for `Redis queues` returned `askAgent=rag_orchestrator_agent`,
  `sourceCount=2`, and `hidden=0`.
- HTTP smoke returned 200 for `/health`, `/search`, `/knowledge`, and
  `/dashboard`.

## Observability Update: AgentRun Detail View

Recorded on 2026-05-01.

Next observability milestone is inspecting one AI execution from dashboard
summary to prompt, model, token, cost, cache, permission, and metadata detail.

Portfolio angle:

- Makes AI orchestration cost and behavior auditable at the individual run
  level.
- Gives reviewers a concrete UI for explaining prompt versions, token usage,
  cache keys, permission level, and runtime metadata.
- Connects the executive dashboard to an engineer-facing trace view without
  changing the agent execution contract.

Implemented scope:

- Added `GET /api/v1/agent-runs/{id}` with a shared AgentRun serializer.
- Added `token_usage` to AgentRun API payloads while preserving flat token
  fields for existing UI code.
- Added `/agent-runs/[id]` frontend detail page.
- Linked recent dashboard AgentRun rows to their detail pages.

Verification evidence:

- `uv run pytest backend/tests/test_agent_runs_api.py -v` passed.
- `uv run pytest backend/tests -v` passed with 61 backend tests.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-agent-run-detail.db`.
- Gmail and Drive sync followed by `POST /api/v1/ask` produced AgentRun `8`
  with `agent=rag_orchestrator_agent`, `tokens=136`, and
  `question=Redis job state` from `/api/v1/agent-runs/8`.
- HTTP smoke returned 200 for `/health`, `/dashboard`, `/agent-runs/8`, and
  `/review`.
- Browser smoke opened `/agent-runs/8`, rendered the Rag Orchestrator run
  details, and reported no console errors.

## Harness Reliability: Isolated Frontend Smoke Cache

Recorded on 2026-05-01.

During browser retesting, the AgentRun detail page rendered without Tailwind
styles because the running Next dev server and `npm run build` shared the same
`.next` directory.

Portfolio angle:

- Shows debugging across browser rendering, CSS asset serving, Next build
  artifacts, and local smoke scripts.
- Turns a flaky local-demo failure into a repeatable regression test.
- Protects future AI-assisted workflows where test/build commands may run
  while the smoke UI remains open.

Implemented scope:

- Added `NEXT_DIST_DIR` support to `frontend/next.config.ts`.
- Updated `scripts/start-smoke.ps1` so smoke dev uses `.next-smoke` instead of
  the production build `.next` directory.
- Added `backend/tests/test_smoke_frontend_cache.py` to guard the cache
  isolation contract.

Verification evidence:

- Reproduced the broken page as a CSS 404 for
  `/_next/static/css/app/layout.css`.
- `uv run pytest backend/tests/test_smoke_frontend_cache.py -v` failed before
  the fix and passed after the fix.
- Restarted smoke with `.tmp/paraworks-agent-run-detail.db`.
- Confirmed `/agent-runs/8` and its CSS file returned 200 before and after
  `npm.cmd run build` while the smoke dev server stayed open.
- Browser smoke reloaded `/agent-runs/8` and rendered the styled AgentRun cards.

## Observability Update: AgentRun Operations Summary

Recorded on 2026-05-01.

Next operations milestone is moving from single-run inspection to an overview
that compares cost, token usage, cache behavior, and status across all agent
tracks.

Portfolio angle:

- Shows AI cost governance at both detail and aggregate levels.
- Gives the three-developer agent split a shared operational dashboard:
  Slack Agent, Mail/Docs Agent, and RAG Orchestrator can be compared without
  coupling their internals.
- Turns token-cost optimization into a visible product workflow instead of a
  hidden backend concern.

Implemented scope:

- Added `GET /api/v1/agent-runs/summary`.
- Returned total runs, token totals, estimated cost, average cost, average
  tokens per run, cache hits, cache hit rate, status counts, and per-agent
  cost/token breakdowns.
- Added frontend `AgentRunSummaryResponse` and `AgentRunAgentSummary` types.
- Added `/agent-runs` as an operations summary page with cards, per-agent
  table, status distribution, and links to run detail pages.
- Added `AI 실행` / `AI Runs` navigation labels and linked the dashboard
  AgentRun panel to the full operations page.

Verification evidence:

- `uv run pytest backend/tests/test_agent_runs_api.py -v` passed.
- `npm.cmd run build` from `frontend` passed and included `/agent-runs`.
- Smoke server restarted with `.tmp/paraworks-agent-run-detail.db`.
- HTTP smoke returned 200 for `/health`, `/agent-runs`, `/dashboard`, and
  `/api/v1/agent-runs/summary`.
- Summary smoke returned `totalRuns=8`, `totalTokens=666`,
  `cacheHitRate=0.0`, and `agents=2`.
- Browser smoke opened `/agent-runs` and confirmed the `AI 실행 관측`,
  `Agent별 비용과 토큰`, `상태 분포`, and `최근 실행 로그` sections.

## Agent Platform Update: Review, Vector, and Orchestration Foundations

Recorded on 2026-05-01.

Next platform milestone is preparing the product loop for real multi-agent
implementation: stricter human review, vector-ready retrieval, and a
LangGraph-ready workflow contract.

Portfolio angle:

- Shows the core AI safety boundary: generated candidates cannot be approved
  into company memory until required fields and evidence are present.
- Introduces a vector-store abstraction without forcing paid embeddings or a
  production Vector DB during MVP development.
- Makes the future LangGraph migration concrete by fixing state and node names
  before adding the dependency.

Implemented scope:

- Added Review promotion preview and approval validation for decision,
  history, and todo review item types.
- Added frontend Review Queue preview cards showing the exact normalized record
  shape that will be promoted on approval.
- Added a permission-aware `InMemoryVectorStore` with hidden-match counting and
  exportable document shape for future pgvector, Chroma, or Qdrant adapters.
- Added a RAG candidate to `VectorDocument` projection bridge.
- Added a local company-memory workflow skeleton with append-only audit state
  and LangGraph-ready node order: collect evidence, draft review candidates,
  retrieve company memory, answer with RAG.

Verification evidence:

- Review preview and promotion tests passed.
- Vector store and existing RAG service tests passed.
- Agent orchestration skeleton tests passed.
- Frontend build passed after Review Queue preview UI changes.
- Full backend suite passed with 70 tests.
- Smoke server restarted with `.tmp/paraworks-review-vector-langgraph.db`.
- Slack sync created 3 pending review items; promotion preview returned
  `canApprove=true` and `target=todo`.
- HTTP smoke returned 200 for `/health`, `/review`, `/search`, and
  `/agent-runs`.
- Browser smoke opened `/review` and confirmed approval preview cards for
  todo, history, and decision records.

## RAG Infrastructure Update: PostgreSQL + pgvector Adapter

Recorded on 2026-05-01.

Confirmed PostgreSQL + pgvector as the production RAG storage direction while
preserving SQLite smoke mode for fast demos.

Portfolio angle:

- Shows a practical RAG infrastructure choice instead of leaving vector storage
  vague.
- Keeps company memory, permissions, source evidence, and vector search close
  to the same transactional Postgres boundary.
- Avoids extra operational complexity from a separate vector database during
  MVP development.

Implemented scope:

- Added `PgVectorStore` with schema SQL, upsert SQL, permission-filtered search
  SQL, and hidden-match accounting.
- Added `PgVectorConfig` with table-name and embedding-dimension validation.
- Added Docker init SQL for `rag_vector_documents`, `embedding vector(1536)`,
  ivfflat cosine index, and permission index.
- Documented PostgreSQL + pgvector as the default RAG storage path in
  `AGENTS.md` and `README.md`.

Verification evidence:

- `uv run pytest backend/tests/test_pgvector_store.py -v` passed.
- `uv run pytest backend/tests -v` passed with 74 backend tests.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-pgvector-adapter.db`.
- HTTP smoke returned 200 for `/health`, `/dashboard`, `/review`, `/search`,
  and `/agent-runs`.
- Browser smoke opened `/search` and confirmed the Company Memory/Search
  surface still rendered under SQLite smoke mode.

## RAG Infrastructure Update: Vector Indexing Pipeline

Recorded on 2026-05-01.

Added the first indexing pipeline that turns current company memory into
embeddable vector documents while keeping local MVP smoke mode independent from
live Postgres.

Portfolio angle:

- Shows how ParaWorks bridges Slack/Gmail/Drive evidence and approved company
  knowledge into a single RAG serving corpus.
- Demonstrates production-minded design: deterministic test embeddings locally,
  a writer protocol for pgvector, and permission metadata carried through every
  indexed document.
- Keeps token cost under control by making indexing explicit and testable
  before introducing paid embedding providers.

Implemented scope:

- Added `DeterministicHashEmbeddingModel` for stable local embedding tests and
  smoke previews.
- Added `index_vector_documents` and `VectorIndexWriter` so the same pipeline
  can target the existing `PgVectorStore` adapter.
- Added `build_rag_index_documents` to collect all source chunks plus approved
  decision, history, and todo records.
- Added `POST /api/v1/rag/reindex` dry-run preview for validating indexing
  coverage without requiring live PostgreSQL in SQLite smoke mode.

Verification evidence:

- `uv run pytest backend/tests/test_rag_indexing.py -v` passed with 4 tests.
- `uv run pytest backend/tests -v` passed with 78 backend tests.
- `uv run ruff check backend/app/rag/embeddings.py backend/app/rag/indexing.py backend/app/api/v1/rag.py backend/tests/test_rag_indexing.py` passed.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-rag-vector-indexing.db`.
- Slack and Gmail mock sync created 3 source chunks; `POST /api/v1/rag/reindex`
  returned `dry_run=true`, `indexed_count=3`, `embedding_dimensions=16`, and
  `storage_backend=preview`.
- HTTP smoke returned 200 for `/dashboard`, `/review`, and `/search`.

## RAG Cost Optimization Update: Incremental Vector Indexing

Recorded on 2026-05-01.

Added the first explicit cost-control layer for paid embedding providers before
connecting OpenAI embeddings.

Portfolio angle:

- Shows product-aware AI engineering: the system avoids repeated embedding
  calls when Slack/Gmail/Drive sync runs over unchanged content.
- Makes cost savings observable through `skipped_count` and
  `saved_embedding_calls`, not just an internal implementation detail.
- Keeps future provider integration safer because the expensive boundary is
  already guarded by content hashing and index state.

Implemented scope:

- Added `VectorIndexState` and the `vector_index_states` table to track
  `document_id + embedding_model + content_hash`.
- Added stable `VectorDocument` content hashing.
- Added `index_changed_vector_documents` to skip unchanged documents, reindex
  changed documents, and persist successful index state.
- Extended `POST /api/v1/rag/reindex` dry-run responses with incremental cost
  signals: `skipped_count`, `skipped_document_ids`, and
  `saved_embedding_calls`.
- Documented that full-corpus re-embedding must not be the default path.

Verification evidence:

- `uv run pytest backend/tests/test_rag_indexing.py backend/tests/test_db_init.py -v`
  passed with 9 focused tests.
- `uv run pytest backend/tests -v` passed with 82 backend tests.
- `uv run ruff check backend/app/models/vector_index.py backend/app/rag/indexing.py backend/app/api/v1/rag.py backend/tests/test_rag_indexing.py`
  passed after Ruff import cleanup.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-incremental-vector-indexing.db`.
- Slack and Gmail mock sync created 3 source chunks; `POST /api/v1/rag/reindex`
  returned `incremental=true`, `indexed_count=3`, `skipped_count=0`, and
  `saved_embedding_calls=0` on a fresh index.
- HTTP smoke returned 200 for `/dashboard`, `/review`, and `/search`.

## RAG Infrastructure Update: Embedding Provider, pgvector Writes, Jobs, and Vector Retrieval

Recorded on 2026-05-01.

Completed the next RAG slice in the agreed order: provider boundary, pgvector
write mode, indexing job contract, and vector-capable retrieval.

Portfolio angle:

- Shows the expensive OpenAI embedding boundary is isolated, batch-oriented,
  usage-aware, and tested without live API calls.
- Demonstrates production safety: SQLite smoke mode cannot accidentally perform
  pgvector writes, while PostgreSQL mode requires an API key and explicit
  `dry_run=false`.
- Adds an operator-friendly job contract so indexing can move to Celery/Redis
  later without changing the product API.
- Makes the RAG answer path vector-ready while keeping local demos stable.

Implemented scope:

- Added `OpenAIEmbeddingModel` and `OpenAIEmbeddingConfig` using batched
  `/v1/embeddings` requests, `encoding_format=float`, optional dimensions, and
  usage tracking.
- Updated incremental indexing to batch only changed documents after content
  hash skip checks.
- Added OpenAI embedding settings and pgvector production write mode for
  `/api/v1/rag/reindex?dry_run=false`.
- Added `POST /api/v1/rag/reindex/jobs` backed by `SyncJob` for indexing job
  status and cost counters.
- Added optional vector-store retrieval in `answer_question_with_rag` and a
  guarded pgvector search adapter for Ask API.

Verification evidence:

- `uv run pytest backend/tests/test_embedding_provider.py backend/tests/test_rag_indexing.py -v`
  passed with provider and batch indexing tests.
- `uv run pytest backend/tests/test_rag_indexing.py::test_reindex_job_endpoint_records_indexing_job -v`
  passed.
- `uv run pytest backend/tests/test_rag_orchestrator_service.py::test_rag_service_can_answer_from_vector_store_matches -v`
  passed.
- `uv run pytest backend/tests -v` passed with 87 backend tests.
- `uv run ruff check backend/app/rag/embeddings.py backend/app/rag/indexing.py backend/app/api/v1/rag.py backend/app/api/v1/ask.py backend/app/agents/rag_orchestrator_agent/service.py backend/tests/test_embedding_provider.py backend/tests/test_rag_indexing.py backend/tests/test_rag_orchestrator_service.py`
  passed after Ruff import cleanup.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with
  `.tmp/paraworks-embedding-pgvector-job-retrieval.db`.
- Slack and Gmail mock sync created 3 source chunks; `POST /api/v1/rag/reindex/jobs`
  returned a `rag-index-*` job with `status=complete`, `indexed_count=3`,
  `embedding_request_count=1`, and `storage_backend=preview`.
- HTTP smoke returned 200 for `/dashboard`, `/review`, and `/search`.

## Product Observability Update: RAG Indexing Admin Panel

Recorded on 2026-05-01.

Moved RAG indexing cost-control signals into the Agent Operations/Admin surface
instead of the end-user Search screen.

Portfolio angle:

- Shows the cost optimization work in a demo-friendly way without polluting the
  final business-user product flow.
- Demonstrates product judgment: technical counters belong in admin
  observability, while Search remains focused on retrieval and evidence.
- Makes `indexed`, `skipped`, and `saved embedding calls` visible for operators
  so the team can prove incremental indexing is reducing provider calls.

Implemented scope:

- Added `GET /api/v1/rag/indexing/summary` with vector index state counts and
  latest `rag-index` jobs.
- Added RAG indexing types to the frontend API contract.
- Added a RAG indexing operations panel to `/agent-runs` with admin-only
  positioning and latest job counters.

Verification evidence:

- `uv run pytest backend/tests/test_rag_indexing.py::test_rag_indexing_summary_returns_latest_jobs_and_state_counts -v`
  passed.
- `uv run ruff check backend/app/api/v1/rag.py backend/tests/test_rag_indexing.py`
  passed.
- `uv run pytest backend/tests -v` passed with 88 backend tests.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-rag-indexing-observability.db`.
- Slack and Gmail mock sync created source chunks; `POST /api/v1/rag/reindex/jobs`
  returned `indexed_count=3`, `embedding_request_count=1`, and
  `status=complete`.
- `GET /api/v1/rag/indexing/summary` returned the latest `rag-index` job.
- HTTP smoke returned 200 for `/agent-runs`, `/search`, and `/dashboard`.

## RAG Infrastructure Update: pgvector Dev Path and Fake Embedding Integration Test

Recorded on 2026-05-01.

Added a safe developer path for validating real PostgreSQL + pgvector behavior
without putting live OpenAI calls in automated tests.

Portfolio angle:

- Shows production-readiness work beyond app code: runbooks, scripts,
  environment boundaries, and integration-test gates.
- Keeps provider cost and secret safety explicit by separating live manual
  checks from automated fake-embedding tests.
- Documents a real local blocker found during validation: Docker Postgres could
  not bind `127.0.0.1:5432` on this machine, and cleanup was handled with
  `docker compose down`.

Implemented scope:

- Added `docs/superpowers/runbooks/pgvector-dev.md` with startup, env,
  `dry_run=false`, fake integration test, port-conflict, and cost-policy notes.
- Added `scripts/start-pgvector-dev.ps1` for Postgres/Redis-backed local app
  startup without embedding secrets in the script.
- Added OpenAI embedding and pgvector search settings to `.env.example`.
- Added runbook/script tests and a skipped-by-default real pgvector integration
  test using `DeterministicHashEmbeddingModel`.

Verification evidence:

- `uv run pytest backend/tests/test_pgvector_dev_runbook.py backend/tests/test_pgvector_integration.py -v`
  passed with 2 tests and skipped the real pgvector integration when
  `PARAWORKS_PGVECTOR_TEST_DATABASE_URL` was unset.
- `uv run ruff check backend/tests/test_pgvector_dev_runbook.py backend/tests/test_pgvector_integration.py`
  passed.
- `uv run pytest backend/tests -v` passed with 90 backend tests and 1 skipped
  opt-in pgvector integration test.
- `npm.cmd run build` from `frontend` passed.
- `docker compose up -d postgres redis` pulled required images but failed to
  bind `127.0.0.1:5432`; partial containers were cleaned up with
  `docker compose down`.

## RAG Operations Update: Celery/Redis Indexing Job Contract

Recorded on 2026-05-01.

Moved RAG reindex jobs behind a Celery/Redis worker contract while preserving
deterministic eager execution for local smoke and tests.

Portfolio angle:

- Shows the difference between an API that does work synchronously and an
  operational job pipeline with queue, polling, and worker boundaries.
- Keeps cost controls intact: the worker executes the same incremental
  hash-skip pipeline before any embedding provider call.
- Demonstrates pragmatic local development: eager mode keeps SQLite smoke fast,
  while `CELERY_TASK_ALWAYS_EAGER=false` enables real Redis worker validation.

Implemented scope:

- Added Celery app construction with Redis broker/result backend and eager-mode
  settings.
- Added `rag.reindex` task plus `execute_rag_reindex_job` for testable job
  status transitions.
- Moved reindex execution logic out of the API route into
  `backend/app/rag/reindexing.py`.
- Updated `POST /api/v1/rag/reindex/jobs` to create a queued job first, then
  execute eagerly in local/test mode or enqueue for Celery in worker mode.
- Added `GET /api/v1/rag/reindex/jobs/{job_id}` for polling.
- Added `scripts/start-celery-worker.ps1` and documented worker mode in the
  pgvector runbook.

Verification evidence:

- `uv run pytest backend/tests/test_rag_indexing_tasks.py backend/tests/test_rag_indexing.py -v`
  passed with 17 focused tests.
- `uv run ruff check backend/app/tasks/celery_app.py backend/app/tasks/rag_indexing.py backend/app/rag/reindexing.py backend/app/api/v1/rag.py backend/tests/test_rag_indexing_tasks.py backend/tests/test_rag_indexing.py`
  passed.
- `uv run pytest backend/tests -v` passed with 95 backend tests and 1 skipped
  opt-in pgvector integration test.
- `npm.cmd run build` from `frontend` passed.
- Smoke server restarted with `.tmp/paraworks-celery-rag-indexing.db`.
- Slack and Gmail mock sync followed by `POST /api/v1/rag/reindex/jobs`
  returned `status=complete`; `GET /api/v1/rag/reindex/jobs/{job_id}` returned
  `indexed_count=3`; summary API returned one latest job.
- HTTP smoke returned 200 for `/agent-runs`, `/dashboard`, and `/search`.

## RAG Operations UX And Dev Path Hardening

Recorded on 2026-05-01.

Implemented the next recommended ParaWorks steps: Admin-facing async job UX,
normal-user search freshness UX, Celery queue-mode contract tests, and a
resilient pgvector local development path.

Portfolio angle:

- Shows product judgment around cost visibility: normal business users see
  company-memory freshness and evidence quality, while Admin/Ops users see
  embedding calls avoided, skipped documents, and job status details.
- Demonstrates operational maturity: RAG reindexing now has clearer
  `queued/running/complete/failed` UX, failure reason surfacing, and polling
  contracts.
- Shows practical backend discipline: queue mode is tested separately from eager
  local mode, so the API boundary stays safe when Redis/Celery is enabled.
- Reduces onboarding friction for collaborators by making pgvector host ports
  configurable instead of requiring tracked compose edits when `5432` is busy.

Implemented scope:

- Added `failure_reason` to RAG indexing job summaries for failed jobs.
- Added tests for failed-job detail responses and non-eager queue behavior.
- Updated `/agent-runs` with Korean operations copy, progress/status display,
  failure reason display, latest RAG jobs, and Admin-only cost counters.
- Updated `/search` with a non-technical company-memory freshness panel and
  removed token/cost/cache details from the normal user answer area.
- Added `PARAWORKS_POSTGRES_PORT` and `PARAWORKS_REDIS_PORT` compose defaults.
- Added `-PostgresPort` and `-RedisPort` to `scripts/start-pgvector-dev.ps1`.
- Documented alternate-port pgvector startup in the runbook.

Cost policy reinforced:

- Keep paid embedding and token-cost details in Admin/Ops screens.
- Give end users confidence signals without encouraging them to reason about
  provider internals.
- Preserve incremental indexing as the first cost gate before provider calls.

Verification evidence:

- Focused RAG/Celery tests passed with 20 tests.
- Focused Ruff passed for changed backend files.
- Full backend tests passed with 98 tests and 1 skipped pgvector integration
  test.
- Frontend production build passed.
- HTTP smoke confirmed health, RAG job creation/detail/summary, `/agent-runs`,
  and `/search`.
- HTML smoke confirmed the new Korean titles render and replacement characters
  are absent.
- Real Redis/Celery worker-mode smoke confirmed `queued -> complete` with
  `CELERY_TASK_ALWAYS_EAGER=false`.

## Connector Ingestion Contract

Recorded on 2026-05-01.

Started the real connector ingestion phase by defining the shared contract that
Slack, Gmail, Drive, Calendar, and future internal-document adapters must use.
Mock connectors now follow the same metadata shape expected from live OAuth
adapters.

Portfolio angle:

- Shows integration architecture beyond mock demos: external data sources enter
  through a stable `SourceEvent` + `ConnectorManifest` boundary.
- Supports 3-developer parallel work because Slack, Mail/Docs, and RAG workers
  can rely on one ingestion result shape instead of importing each other's code.
- Adds operational sync accounting with fetched, created, and skipped counts.
- Reinforces cost control before LLM/RAG work: duplicate source events are
  skipped before review extraction and before any downstream embedding.

Implemented scope:

- Added `ConnectorManifest` for connector type, display name, mode, auth type,
  OAuth scopes, sync strategy, and cost policy.
- Added a connector registry for integration metadata.
- Added `sync_connector_events` to centralize `SyncJob` creation, connector
  fetch, ingestion, duplicate skip counts, completion, and failure handling.
- Updated the integrations API to list manifest metadata and use the shared
  sync boundary.
- Updated `/integrations` to show connector manifest metadata, OAuth scope
  summaries, sync strategy, cost policy, fetched counts, and skipped counts in
  Korean.
- Updated `AGENTS.md` with connector ingestion rules for coding assistants.

Verification evidence:

- Focused connector tests passed with 12 tests.
- Focused Ruff passed for changed connector, ingestion, API, and test files.
- Full backend tests passed with 102 tests and 1 skipped pgvector integration
  test.
- Frontend production build passed.
- Smoke confirmed `/api/v1/integrations` returns 4 connector manifests,
  Slack history scopes, successful Slack sync with fetched/created/skipped
  counts, and `/integrations` renders Korean copy without replacement
  characters.

## Playwright Visual Smoke And RAG Permission Audit

Recorded on 2026-05-01.

Made frontend visual checking repeatable with Playwright and started the next
RAG permission/security hardening slice.

Portfolio angle:

- Adds a real visual regression workflow across desktop and mobile, not only
  HTTP smoke checks.
- Turns the previous Korean mojibake issue into an automated guardrail by
  checking key pages for Korean headings and broken replacement text.
- Strengthens RAG auditability without leaking hidden source content:
  end-users can see hidden match counts, while restricted source details remain
  filtered.
- Preserves connector ACL metadata on chunks so downstream RAG, review, and
  portfolio explanations can trace why content was visible or hidden.

Implemented scope:

- Installed `@playwright/test` and Chromium for local visual smoke.
- Added `frontend/playwright.config.ts`, `frontend/e2e/visual-smoke.spec.ts`,
  and `scripts/run-visual-smoke.ps1`.
- Added `npm run test:visual`.
- Rewrote `/dashboard` Korean copy to remove mojibake.
- Added `hidden_match_count` to search responses and `source_id` to visible
  search results.
- Added `source_ids` to Ask/RAG answers so visible answer citations are
  auditable by stable source identifiers.
- Preserved source id, permission level, participants, and connector raw
  metadata in `DocumentChunk.metadata_`.
- Updated `/search` to show hidden match counts without exposing hidden
  snippets or links.

Verification evidence:

- Focused permission/connector/Ask tests passed with 15 tests.
- Focused Ruff passed for changed backend search, ingestion, and tests.
- Full backend tests passed with 102 tests and 1 skipped pgvector integration
  test.
- Frontend production build passed.
- Playwright visual smoke passed with 10 Chromium desktop/mobile tests.
- Playwright initially failed because browser binaries were missing; installing
  Chromium made the check executable for future runs.

## Slack Live API Client Boundary

Recorded on 2026-05-01.

Started the real OAuth connector phase with a Slack Web API client boundary
while keeping mock mode as the default for demos and tests.

Portfolio angle:

- Shows the transition from mock connector harness to a live API-ready
  integration without leaking or requiring real workspace tokens.
- Keeps the connector architecture testable: Slack API behavior is verified
  with `httpx.MockTransport` and fake clients, never by calling Slack in tests.
- Preserves the ingestion contract: live Slack payloads still become
  `SourceEvent` records and flow through the same `sync_connector_events`
  pipeline as mock data.
- Reinforces cost and security discipline before LLM work: source deltas are
  fetched first, duplicates are skipped, and review/RAG boundaries remain
  evidence-driven.

Implemented scope:

- Added `SlackWebApiClient` for `conversations.history` bearer-token calls.
- Added cursor pagination and clear `SlackApiError` handling.
- Added `get_configured_connector` so Slack settings build a live connector
  only when token and channel ids are present.
- Updated `/api/v1/integrations/{connector_type}/sync` to use the configured
  connector factory while preserving mock fallback.
- Updated the Slack integration runbook with live env settings, scope
  requirements, no-secret policy, fake-client test policy, and cost/security
  notes.

Verification evidence:

- Focused Slack connector/factory/mock sync tests passed with 7 tests.
- Focused Ruff passed for the touched Slack connector, factory, integration
  endpoint, and tests.
- Full backend tests passed with 106 tests and 1 skipped pgvector integration
  test.
- Frontend production build passed.
- Playwright visual smoke passed with 10 Chromium desktop/mobile tests.

## Slack OAuth Installation Boundary

Recorded on 2026-05-01.

Added the first OAuth installation boundary for Slack while keeping real
workspace access opt-in and mock/demo behavior safe by default.

Portfolio angle:

- Demonstrates secure integration design beyond mock data: install URLs use
  signed state, OAuth code exchange is isolated behind a client boundary, and
  database records never store raw bot tokens.
- Shows production-minded defaults: `PARAWORKS_DEMO_MODE=true` keeps mock sync
  active even when local Slack credentials exist, preventing accidental API
  usage, private data ingestion, and surprise downstream indexing costs.
- Keeps the implementation testable without external services through
  `httpx.MockTransport`, fake access payloads, and a local token vault boundary.
- Creates a clean handoff point for the next developer slice: replacing the
  local vault with a managed secret store and wiring installed connections into
  sync.

Implemented scope:

- Added Slack OAuth settings and `.env.example` placeholders.
- Added `IntegrationConnection` to persist workspace metadata, scopes,
  `token_ref`, masked token, status, and non-sensitive metadata.
- Added `SlackOAuthStateSigner`, `SlackOAuthClient`, `LocalTokenVault`, install
  URL builder, and callback completion service.
- Added `/api/v1/integrations/slack/oauth/install-url` and callback endpoint.
- Updated connector factory so live Slack sync requires demo mode to be
  disabled as well as token/channel configuration.
- Updated Slack runbook with OAuth env, testing, cost, and security rules.

Verification evidence:

- RED test first: `backend/tests/test_slack_oauth.py` initially failed because
  `backend.app.connectors.slack_oauth` did not exist.
- Focused OAuth tests passed with 5 tests.
- Focused connector factory/mock/review/OAuth regression tests passed with 14
  tests.
- Focused Ruff passed for touched backend files and tests.
- Full backend tests passed with 112 tests and 1 skipped pgvector integration
  test.
- Frontend production build passed.
- Playwright visual smoke passed with 10 Chromium desktop/mobile tests.

## Slack OAuth UI Status

Recorded on 2026-05-01.

Wired the Slack OAuth installation boundary into the Integrations experience so
users can see whether Slack is connected, installable, or still waiting for
environment configuration.

Portfolio angle:

- Shows full-stack integration maturity: backend exposes sanitized connection
  state, and the frontend renders status/CTA without leaking raw tokens or
  `token_ref` values.
- Keeps the portfolio demo safe and cost-aware: mock sync remains usable when
  OAuth is not configured, and the UI clearly separates setup readiness from
  actual data ingestion.
- Adds visual smoke coverage so future UI work catches broken OAuth status
  cards on both desktop and mobile.

Implemented scope:

- Added `/api/v1/integrations/connections` to return connection status,
  workspace metadata, scopes, and masked tokens only.
- Added frontend API types for Slack OAuth install URLs and integration
  connections.
- Updated `/integrations` Slack card with connection status, setup guidance,
  and a safe Slack install CTA.
- Added Playwright coverage that the Slack OAuth status renders and does not
  expose common secret markers.

Verification evidence:

- RED backend test first: `/api/v1/integrations/connections` returned 404
  before implementation.
- RED visual smoke first: `[data-testid="slack-oauth-status"]` was missing
  before UI implementation.
- Focused backend connection API test passed.
- Python Ruff passed for touched backend files and tests.
- Full backend tests passed with 113 tests and 1 skipped pgvector integration
  test.
- Frontend production build passed.
- Playwright visual smoke passed with 12 Chromium desktop/mobile tests on fresh
  alternate ports.

## Installed Slack Sync Token Boundary

Recorded on 2026-05-01.

Connected installed Slack OAuth records to the sync connector factory without
putting raw tokens in the database or API responses.

Portfolio angle:

- Shows the handoff from OAuth installation to live ingestion readiness: the
  sync path can now build a `SlackConnector` from a stored connection record and
  a vault-resolved bot token.
- Preserves the cost guardrail: live Slack sync still requires
  `PARAWORKS_DEMO_MODE=false`, configured channel ids, and a resolvable vault
  token. Missing vault state falls back to mock instead of making unexpected
  external calls.
- Keeps the security story crisp for interviews: DB stores `token_ref` only,
  the vault resolves the secret at runtime, and sync responses never expose raw
  tokens or token references.

Implemented scope:

- Added `get_sync_connector` as the sync-time factory that can use installed
  Slack connections.
- Kept `get_configured_connector` as the legacy env-token path for local live
  experiments.
- Updated `/api/v1/integrations/{connector_type}/sync` to use the sync-time
  factory with DB context.
- Added tests for installed connection token resolution, missing-vault fallback,
  and sync endpoint secret non-exposure.
- Updated Slack runbook with installed sync selection and cost/security notes.

Verification evidence:

- RED factory test first failed because `get_sync_connector` did not exist.
- Focused connector/OAuth/mock sync tests passed with 13 tests.
- Python Ruff passed for touched backend files and tests.
- Full backend tests passed with 116 tests and 1 skipped pgvector integration
  test.
- Frontend production build passed.
- Playwright visual smoke passed with 12 Chromium desktop/mobile tests on fresh
  alternate ports.

## 2026-05-01 - Route Audit, Integrations Resilience, And Next 16 Upgrade

Audited the frontend after Slack OAuth status UI caused the integrations page to
degrade when optional status endpoints were unavailable on a stale backend.

Portfolio angle:

- Shows production-minded frontend hardening: core connector manifests now render
  independently from optional Slack OAuth connection metadata.
- Demonstrates end-to-end QA ownership: route coverage expanded from a narrow
  dashboard check to desktop/mobile smoke checks across dashboard, messages,
  review, knowledge, integrations, agent runs, and search.
- Keeps the cost story explicit: missing optional integration status APIs fail
  locally in UI state instead of triggering extra live connector, Slack, or
  embedding calls.

Implemented scope:

- Upgraded frontend dependencies to Next.js 16.2.4 and aligned ESLint with the
  ESLint 9 flat-config path.
- Fixed `/integrations` loading so Gmail, Google Drive, and Google Calendar
  modules stay visible when Slack OAuth status endpoints are not available.
- Removed the disabled Slack setup button from the normal demo path so the
  `Slack Agent 실행` action no longer wraps because of an unnecessary control.
- Added Playwright assertions for route-level rendering, mojibake prevention,
  missing application errors, and connector-card presence.

Verification evidence:

- Frontend lint passed with ESLint 9.
- Frontend production build passed on Next.js 16.2.4.
- Full backend tests passed with 116 tests and 1 skipped pgvector integration
  test.
- Playwright visual smoke passed with 22 Chromium desktop/mobile tests across
  all current MVP pages.
- `npm audit --audit-level=moderate` still reports a moderate advisory through
  Next's bundled PostCSS range; the suggested forced fix would downgrade Next and
  should not be applied.

## 2026-05-01 - Google OAuth Boundary For Gmail, Drive, And Calendar

Added a Google OAuth installation boundary for the three Google connector cards
without enabling live Google sync yet.

Portfolio angle:

- Shows disciplined integration sequencing: OAuth security and connection
  metadata land before live data ingestion.
- Demonstrates multi-connector architecture: Gmail, Drive, and Calendar share a
  signed-state OAuth boundary while preserving each connector's own scope set.
- Keeps the cost story visible: OAuth readiness does not trigger Google sync,
  LLM calls, or embedding work; future sync should fetch deltas and hash-check
  content before downstream agent work.

Implemented scope:

- Added `google_oauth.py` with signed state, install URL generation, callback
  completion, Google token exchange boundary, and sanitized persistence.
- Added backend settings for Google client id, client secret, redirect URI, and
  OAuth state secret.
- Extended the local token vault with a generic token kind so Google stores
  `local:<connector>:<account>:oauth` instead of a Slack-specific bot token ref.
- Added generic Google OAuth install/callback API routes under
  `/api/v1/integrations/{gmail|drive|calendar}/oauth/...`.
- Updated the Integrations UI to show OAuth status boxes for Gmail, Drive, and
  Calendar while keeping primary card actions focused on sync/agent execution.
- Added a Google integration runbook and a plan note for the implementation
  sequence.

Verification evidence:

- RED backend test first failed because `backend.app.connectors.google_oauth`
  did not exist.
- RED Playwright test first failed because `gmail-oauth-status` was missing.
- Focused Google/Slack OAuth backend tests passed with 12 tests.
- Frontend lint passed after the OAuth UI update.
- Playwright visual smoke passed with 24 Chromium desktop/mobile tests after
  Google OAuth readiness assertions were added.

## 2026-05-01 - Google Installed Sync Boundary

Connected installed Google OAuth records to the sync connector factory through a
live connector skeleton.

Portfolio angle:

- Shows the integration handoff after OAuth: installed Gmail, Drive, and Calendar
  connections can now become provider-specific sync connectors when demo mode is
  disabled.
- Demonstrates a merge-friendly split for three developers: each Google provider
  can now evolve behind the same `GoogleConnector` and `SourceEvent` contract.
- Keeps cost discipline explicit: demo mode remains mock-first, missing vault
  tokens fall back to mock, and future provider work must add cursor/hash delta
  checks before downstream agent or embedding calls.

Implemented scope:

- Added `backend/app/connectors/google.py` with Google API client skeletons and
  Gmail/Drive/Calendar `SourceEvent` mapping.
- Extended `get_sync_connector` to resolve installed Google connection tokens
  from the local vault when `PARAWORKS_DEMO_MODE=false`.
- Preserved mock fallback for demo mode and missing vault tokens.
- Added connector and factory tests for provider mapping, bearer-token headers,
  installed token resolution, demo fallback, and missing-vault fallback.
- Updated the Google integration runbook and added an implementation plan note.

Verification evidence:

- RED tests first failed because `backend.app.connectors.google` did not exist.
- Focused Google connector/factory tests passed with 13 tests.
- Python Ruff passed for the new Google connector, factory, and tests.
- Full backend tests passed with 129 tests and 1 skipped pgvector integration
  test.
- Frontend lint and production build passed.
- Playwright visual smoke passed with 24 Chromium desktop/mobile tests.
- The in-app browser showed Slack, Gmail, Drive, Calendar, and Google OAuth
  status blocks on `http://127.0.0.1:3000/integrations`.

## 2026-05-01 - Slack OAuth Callback UX And Redirect Audit

Hardened the Slack OAuth install path after a real Slack authorization attempt
failed with `redirect_uri did not match any configured URIs`.

Portfolio angle:

- Shows practical OAuth troubleshooting beyond mock integrations: local app
  routes, backend install URL generation, and third-party console settings must
  align exactly.
- Adds a safer user-facing callback page so OAuth failures are explained in
  Korean instead of surfacing a broken route or raw API response.
- Reinforces the security story: the callback UI shows sanitized workspace
  metadata only and regression tests block raw token, client secret, and
  `token_ref` leakage.

Implemented scope:

- Added `/integrations/slack/callback` frontend route.
- Forwarded Slack `code` and signed `state` to the backend callback endpoint.
- Rendered safe success, loading, and failure states for Korean business users.
- Documented that Slack App Redirect URLs must exactly match
  `SLACK_OAUTH_REDIRECT_URI`, including the `localhost` vs `127.0.0.1`
  distinction.

Cost/security note:

- OAuth installation itself does not sync Slack history, call an LLM, or create
  embeddings. Live sync remains gated by demo mode, channel ids, and vault token
  resolution so accidental installs do not create downstream token or embedding
  costs.

## 2026-05-02 - Slack OAuth Credential Status Guardrail

Added a clearer boundary between stored Slack connection metadata and actual
live-sync credential availability.

Portfolio angle:

- Shows a realistic integration hardening step: OAuth metadata in the database
  is not the same as a usable secret in the runtime vault.
- Prevents a misleading "connected" UI after local backend restarts, where the
  development in-memory vault may no longer hold the bot token.
- Keeps the secret boundary intact by exposing only `credential_status`, never
  raw tokens or `token_ref` values.

Implemented scope:

- Added sanitized `credential_status` to `/api/v1/integrations/connections`.
- Marked credentials as `available` only when the current backend process can
  resolve the local vault token.
- Updated the Integrations UI to show "재연결 필요" when connection metadata
  exists but the local development token is missing.
- Documented the local vault restart limitation in the Slack runbook.

Cost/security note:

- The UI now makes it harder to accidentally assume live Slack sync is ready.
  Real Slack ingestion remains gated by `PARAWORKS_DEMO_MODE=false`, channel
  ids, and a resolvable vault token before any downstream review, LLM, or
  embedding work can run.

## 2026-05-02 - Slack OAuth Reconnect UX

Closed the follow-up UX gap after adding credential availability checks: users
can now recover from local vault token loss directly from the Slack card.

Portfolio angle:

- Shows end-to-end product polish around real integration failure modes, not
  only the happy OAuth path.
- Keeps the primary sync/agent actions stable while placing the reconnect CTA
  inside the OAuth status area where it belongs.
- Adds desktop/mobile visual coverage for the `token missing -> Slack 재연결`
  state so the workspace name remains a single-line title and the recovery
  action stays readable.

Implemented scope:

- Added a `Slack 재연결` CTA when OAuth metadata exists but
  `credential_status` is missing.
- Kept the reconnect CTA out of the primary action row to avoid crowding
  `동기화` and `Slack Agent 실행`.
- Simplified the OAuth status title to the workspace name only.
- Kept the workspace title on one line with truncation; reconnect state is
  carried by the status pill, helper copy, and `Slack 재연결` CTA.

Cost/security note:

- Reconnection only refreshes the local credential boundary. It still does not
  sync Slack history or trigger downstream LLM/embedding work while demo mode is
  enabled.

## 2026-05-02 - Slack Live Sync Error Handling

Started the first real Slack sync verification with `PARAWORKS_DEMO_MODE=false`
and confirmed the connector reaches Slack, but the configured channel is not
readable by the bot yet.

Portfolio angle:

- Shows real integration debugging beyond OAuth success: app installation,
  bot-channel membership, and channel ids are separate operational checks.
- Improves API resilience by turning Slack Web API failures into explicit 502
  responses instead of generic 500 errors.
- Keeps privacy intact during live testing by checking channel access and
  counts without printing Slack message bodies.

Implemented scope:

- Added a regression test for Slack API failure handling on the sync endpoint.
- Mapped `SlackApiError` from sync to an HTTP 502 with a clear detail message.
- Documented `channel_not_found` and `not_in_channel` troubleshooting in the
  Slack runbook.

Verification evidence:

- Live sync reached Slack and returned
  `Slack conversations.history failed: channel_not_found`.
- Follow-up channel access probes returned `not_in_channel` for sampled public
  channels, meaning the bot must be invited to a target channel or
  `SLACK_CHANNEL_IDS` must point to a bot-readable channel.

Cost/security note:

- The failed live sync did not trigger LLM or embedding work. Connector access
  is still the first cost gate; downstream review/RAG processing should only
  run after source access is valid and duplicate checks have completed.

## 2026-05-02 - Slack Live Sync Smoke Success

Completed the first successful live Slack sync path after adding the ParaWorks
bot to the configured Slack channel.

Portfolio angle:

- Demonstrates a real SaaS integration beyond mock data: OAuth, bot channel
  membership, Slack Web API access, ingestion, duplicate skipping, and agent
  review generation now work as one local smoke path.
- Shows privacy-aware verification: live Slack messages were synced into the
  local app, but terminal output only reported counts and status metadata, not
  message bodies.
- Reinforces cost discipline: source duplicate checks skipped unchanged Slack
  events before downstream review/agent work.

Verification evidence:

- Backend ran with `PARAWORKS_DEMO_MODE=false`.
- Slack `conversations.history` access check succeeded for the configured
  channel.
- `POST /api/v1/integrations/slack/sync` returned `status=complete`,
  `fetched_events=194`, `skipped_events=194`, and `created_review_items=0`.
- `POST /api/v1/integrations/slack/agent-review` returned
  `created_review_items=1` with the deterministic local Slack Agent.
- Agent run observability showed `total_runs=3`, `total_tokens=250`, and
  `estimated_cost_usd=0.000081`.

Next product step:

- Add a live sync readiness/status surface so users can see the active mode,
  configured channel id, last sync counts, and Slack API errors without opening
  terminal logs.
- Then continue with Review Queue promotion and RAG indexing over approved
  Slack-derived timeline/history candidates.

## 2026-05-02 - LangGraph Orchestrator Foundation

Moved the company memory orchestration foundation from a local sequential
runner to a real LangGraph `StateGraph` while keeping deterministic tests and
the existing agent contracts intact.

Portfolio angle:

- Shows the core ParaWorks architecture moving toward a true multi-agent
  orchestration layer instead of isolated demo agents.
- Keeps the three-developer split clean: Slack Agent, Mail/Docs Agent, and RAG
  Orchestrator can continue evolving behind shared `EvidencePacket` and
  review/RAG contracts.
- Adds a visible graph topology (`graph_mermaid`) that can later be reused in
  documentation, operations screens, or portfolio diagrams.

Implemented scope:

- Added `langchain>=1.2.0,<2.0.0` and `langgraph>=1.1.6,<2.0.0` to the backend
  dependencies. Local resolution installed `langchain==1.2.17` and
  `langgraph==1.1.10`.
- Replaced the local `AgentWorkflow.run()` loop with a compiled LangGraph
  `StateGraph`.
- Preserved append-only node audit behavior and exposed the graph as Mermaid.
- Added a workflow output marker for the cost policy:
  `delta_sync_hash_skip_evidence_budget`.

Cost/security note:

- This foundation still performs no paid LLM calls in tests. The next LLM
  integration should keep deterministic model doubles for CI, use delta sync
  and source-hash skips before agent calls, and persist `AgentRun` token/cost
  metadata for every production model call.

Verification evidence:

- `uv run pytest backend/tests/test_agent_orchestration.py -v` passed.
- `uv run pytest backend/tests/test_agent_runtime_contracts.py backend/tests/test_agent_registry.py backend/tests/test_agent_orchestration.py backend/tests/test_slack_agent.py backend/tests/test_mail_document_agent.py backend/tests/test_rag_orchestrator_agent.py -v` passed with 17 tests.

## 2026-05-02 - LangGraph Orchestration API

Exposed the company memory LangGraph workflow through backend API endpoints so
the frontend and operations screens can inspect orchestration status without
calling paid models.

Portfolio angle:

- Turns the orchestration foundation into a product-visible capability:
  backend clients can now read the active graph backend, node order, Mermaid
  topology, and cost guardrails.
- Adds a deterministic dry-run endpoint that proves the orchestration path
  executes end-to-end without invoking Slack, embeddings, or paid LLM APIs.
- Makes the architecture easier to explain in interviews: the graph can be
  shown as an API-backed execution contract instead of only code internals.

Implemented scope:

- Added `GET /api/v1/orchestration/company-memory` for workflow status,
  `node_names`, `graph_mermaid`, and cost policy flags.
- Added `POST /api/v1/orchestration/company-memory/dry-run` for deterministic
  execution over the same LangGraph workflow.
- Registered the orchestration router in the v1 API router.
- Added API tests for status and dry-run behavior.

Cost/security note:

- The status and dry-run endpoints report `paid_llm_calls_in_status_api=false`
  and `token_cost_usd=0`. This keeps operational visibility separate from
  model execution cost.

Verification evidence:

- `uv run pytest backend/tests/test_orchestration_api.py backend/tests/test_agent_orchestration.py backend/tests/test_agent_runs_api.py -v` passed with 9 tests.
- `uv run ruff check backend/app/api/v1/orchestration.py backend/app/api/v1/router.py backend/tests/test_orchestration_api.py backend/app/agent_runtime/orchestration.py backend/tests/test_agent_orchestration.py` passed.

## 2026-05-02 - Agent Runs LangGraph Operations Card

Connected the new LangGraph orchestration status API to the Agent Runs
operations page.

Portfolio angle:

- Makes the multi-agent orchestration architecture visible in the product UI:
  users can see the Company Memory graph backend, execution steps, and cost
  guardrails from the same page that tracks agent runs and token cost.
- Shows practical AI cost design in the interface: delta sync, source-hash
  skipping, evidence token budgeting, and blocked paid calls are presented as
  operational controls instead of buried implementation notes.
- Improves interview/demo storytelling by tying backend LangGraph work to a
  browser-verified admin experience.

Implemented scope:

- Added frontend API typing for `/api/v1/orchestration/company-memory`.
- Fetched orchestration status on `/agent-runs`.
- Added a LangGraph operations card with workflow steps and cost guardrails.
- Rechecked the page with Playwright screenshot verification after restarting
  the local smoke backend/frontend.

Cost/security note:

- The Agent Runs page only reads the status endpoint. It does not call the
  dry-run endpoint during render and does not trigger Slack, embeddings, or
  paid LLM calls.

Verification evidence:

- `npx eslint src/app/agent-runs/page.tsx src/lib/api/types.ts` passed.
- `npm run build` passed.
- `npx playwright screenshot --full-page http://127.0.0.1:3000/agent-runs ..\\.tmp\\agent-runs-langgraph-v2.png` completed.
- `npx playwright test e2e/visual-smoke.spec.ts -g "/agent-runs renders" --project=chromium-desktop` passed.

## 2026-05-02 - LangGraph Capture And Dry-Run UX

Captured the Company Memory LangGraph as reusable portfolio documentation and
added a zero-cost dry-run control to the Agent Runs operations page.

Portfolio angle:

- Adds a concrete architecture visual that can be used in the final portfolio:
  `docs/assets/company-memory-langgraph.svg` and
  `docs/assets/company-memory-langgraph.png`.
- Demonstrates that the LangGraph orchestrator is not only backend plumbing:
  the admin UI can now execute a deterministic dry-run and show the result.
- Shows cost discipline in product behavior: dry-run confirms orchestration
  order without Slack sync, embeddings, or paid LLM calls.

Implemented scope:

- Added a saved SVG graph and a Playwright-captured PNG for the Company Memory
  workflow.
- Added `OrchestrationDryRunResponse` frontend typing.
- Added a client-side `OrchestrationDryRun` control on `/agent-runs`.
- Added a Playwright regression test for the zero-cost dry-run UX.

Cost/security note:

- The dry-run calls `/api/v1/orchestration/company-memory/dry-run` and returns
  `token_cost_usd=0`. It does not read Slack message bodies, call embedding
  providers, or invoke external LLM APIs.

Verification evidence:

- `npx playwright screenshot --viewport-size=1280,720 file:///C:/Users/hanvv/Study/potenup3/pj04_ParaWorks/docs/assets/company-memory-langgraph.svg ..\\docs\\assets\\company-memory-langgraph.png` completed.
- `npx eslint src/app/agent-runs/page.tsx src/app/agent-runs/OrchestrationDryRun.tsx src/lib/api/types.ts e2e/orchestration.spec.ts` passed.
- `npm run build` passed.
- `npx playwright test e2e/orchestration.spec.ts --project=chromium-desktop` passed.
- `POST /api/v1/orchestration/company-memory/dry-run` returned four completed
  nodes and `token_cost_usd=0`.

## 2026-05-02 - LangGraph Agent Service Execution

Connected the Company Memory LangGraph workflow to the existing Slack,
Mail/Docs, and RAG agent services.

Portfolio angle:

- Moves the orchestrator from a visible dry-run foundation to a real execution
  path: LangGraph nodes now call agent services that persist `AgentRun`
  records and create review candidates.
- Preserves the three-developer split: Slack Agent and Mail/Docs Agent produce
  human-reviewable timeline/history candidates, while the RAG Orchestrator
  answers from company memory evidence.
- Demonstrates cost-aware orchestration: the real run endpoint is separate from
  status/dry-run and marked with `requires_explicit_run=true` so UI rendering
  never triggers hidden agent costs.

Implemented scope:

- Added `backend.app.agent_runtime.company_memory` for service-level Company
  Memory orchestration.
- Added reusable LangGraph workflow construction for custom node handlers.
- Added `POST /api/v1/orchestration/company-memory/run` as the explicit agent
  execution endpoint.
- Added tests proving Slack/Mail/RAG agent services run through LangGraph and
  persist the expected `AgentRun` and `ReviewItem` records.

Cost/security note:

- This run still uses deterministic local model implementations in tests. It
  creates estimated `AgentRun` token/cost metadata, but does not call external
  LLM providers unless a future production model adapter is explicitly wired.
- The endpoint is an explicit POST action, not part of page render/status
  polling, to avoid accidental token spend.

Verification evidence:

- `uv run pytest backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_agent_orchestration.py backend/tests/test_slack_agent.py backend/tests/test_mail_document_agent.py backend/tests/test_rag_orchestrator_service.py -v` passed with 18 tests.
- `uv run ruff check backend/app/agent_runtime/company_memory.py backend/app/agent_runtime/orchestration.py backend/app/agent_runtime/__init__.py backend/app/api/v1/orchestration.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py` passed.

## 2026-05-02 - Agent Candidate Bulk Approval

Added a safe Review Queue operation for approving agent-generated candidates
into Knowledge records.

Portfolio angle:

- Strengthens the human-in-the-loop company memory workflow: agent outputs do
  not enter durable Knowledge automatically, but reviewers can now approve
  agent candidates as a deliberate batch operation.
- Shows practical orchestration boundary design: Slack/Mail agents draft
  candidates, Review Queue gates them, and approved items become Knowledge that
  RAG can use.
- Demonstrates cost-aware workflow design because approval does not call LLMs
  or embeddings; it only promotes already-reviewed structured records.

Implemented scope:

- Added `POST /api/v1/review/approve-agent-candidates`.
- The endpoint only approves pending items that include an agent marker
  (`payload.agent_name`) and valid source evidence.
- Manual reviewer-created pending items remain pending.
- Added cost policy metadata indicating no paid LLM or embedding calls.

Cost/security note:

- The operation requires the human review state (`pending_review`) and skips
  invalid or manual items. It does not read secrets, call connectors, or trigger
  embedding/indexing work.

Verification evidence:

- `uv run pytest backend/tests/test_review_knowledge_promotion.py backend/tests/test_review.py backend/tests/test_knowledge_api.py backend/tests/test_rag_orchestrator_service.py -v` passed with 17 tests.
- `uv run ruff check backend/app/api/v1/review.py backend/tests/test_review_knowledge_promotion.py` passed.

## 2026-05-02 - Slack Runtime Status Surface

Added a Slack runtime status endpoint and connected it to the Integrations
operations UI.

Portfolio angle:

- Gives operators a direct view of Slack sync readiness: mock/live mode,
  configured channel ids, connection status, credential availability, and the
  latest sync job.
- Turns previous terminal-only Slack troubleshooting into product-visible
  observability.
- Reinforces cost discipline: the status lookup explicitly does not trigger
  sync, embeddings, or LLM calls.

Implemented scope:

- Added `GET /api/v1/integrations/slack/runtime-status`.
- The endpoint returns mode, configured channel ids, connection/credential
  status, latest Slack sync job metadata, and cost-policy flags.
- Added frontend `SlackRuntimeStatus` typing.
- Added a Slack operations status panel to `/integrations`.
- Extended Playwright smoke coverage to assert the runtime status panel is
  visible and still does not expose secrets.

Cost/security note:

- Runtime status is read-only. It reports existing metadata and does not fetch
  Slack messages, expose bot tokens, or invoke model/embedding work.

Verification evidence:

- `uv run pytest backend/tests/test_integration_runtime_status.py backend/tests/test_slack_oauth.py backend/tests/test_connector_factory.py -v` passed with 18 tests.
- `uv run ruff check backend/app/api/v1/integrations.py backend/tests/test_integration_runtime_status.py` passed.
- `npx eslint src/app/integrations/page.tsx src/lib/api/types.ts e2e/visual-smoke.spec.ts` passed.
- `npm run build` passed.
- `npx playwright test e2e/visual-smoke.spec.ts -g "integrations page shows Slack OAuth" --project=chromium-desktop` passed.

## 2026-05-02 - Google Runtime Status Surface

Extended connector runtime observability from Slack to Gmail, Google Drive, and
Google Calendar.

Portfolio angle:

- Makes Google integration readiness inspectable in the product UI before the
  team invests in deeper live connector work.
- Aligns all major connectors around the same operational contract: mode,
  connection state, credential state, account/channel context, latest sync, and
  no-cost status lookup.
- Reduces debugging dependence on terminal logs for OAuth and sync issues.

Implemented scope:

- Added `GET /api/v1/integrations/{gmail|drive|calendar}/runtime-status`.
- Added backend tests for Google runtime status and unknown connector handling.
- Added frontend `GoogleRuntimeStatus` typing.
- Added a Google operations status panel to `/integrations`.
- Extended Playwright smoke coverage to assert the Google runtime panel is
  visible.

Cost/security note:

- Google runtime status is read-only. It does not call Google APIs, fetch mail
  or documents, trigger embeddings, or invoke LLMs. It also avoids exposing raw
  refresh tokens or token references.

Verification evidence:

- `uv run pytest backend/tests/test_integration_runtime_status.py -v` passed.
- `uv run ruff check backend/app/api/v1/integrations.py backend/tests/test_integration_runtime_status.py` passed.
- `npx eslint src/app/integrations/page.tsx src/lib/api/types.ts e2e/visual-smoke.spec.ts` passed.
- `npm run build` passed.
- `npx playwright test e2e/visual-smoke.spec.ts -g "Google connector cards" --project=chromium-desktop` passed.

## 2026-05-02 - Execution Cost Plan And Skip Reasons

Why it matters:

- The company-memory graph should not call every agent just because a user
  pressed run. Slack, mail/document, and RAG agents now receive an execution
  cost plan before the graph enters the expensive service nodes.
- The cost plan records each agent's `run` or `skip` decision, the reason, and
  deterministic input/output token estimates. This keeps the demo portfolio
  honest about API cost instead of hiding cost behind orchestration language.
- Empty Slack evidence, empty mail/document evidence, and empty questions now
  skip their agent calls and avoid creating misleading `AgentRun` records.

Implemented scope:

- Added a company-memory cost plan builder to the LangGraph runtime.
- Threaded `cost_plan` through graph state and orchestration outputs.
- Guarded Slack review drafting, mail/document review drafting, and RAG answer
  generation with per-agent skip decisions.
- Added regression coverage for both run and skip paths.

Cost/security note:

- This is a local deterministic estimate. It does not call an embedding model,
  LLM, Slack, Google, or external API.
- The skip path is intentionally conservative: if there is no evidence or no
  user question, the runtime spends zero model tokens for that agent.

Verification evidence:

- `uv run pytest backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_agent_runs_api.py -v` passed.
- `uv run ruff check backend/app/agent_runtime/company_memory.py backend/tests/test_company_memory_orchestration_service.py` passed.

## 2026-05-02 - Runtime Status Secret Redaction

Why it matters:

- Integration status pages are useful for debugging live Slack and Google
  setup, but sync failure messages can accidentally include access tokens,
  refresh tokens, token references, or OAuth client secrets.
- Runtime status APIs now redact secret-like strings before returning
  `latest_sync.message` to the frontend.
- The original sync record is left intact for server-side diagnosis; redaction
  happens at the API boundary where user-facing exposure risk exists.

Implemented scope:

- Added `redact_secret_text` for Slack token, token reference, refresh token,
  and client secret patterns.
- Applied redaction to integration runtime status sync messages.
- Added regression tests for Slack and Google runtime status secret leakage.

Cost/security note:

- The redaction path is local string processing. It does not call connector
  APIs or LLMs.
- This reduces the risk of leaking sensitive operational values through the
  Korean-first dashboard during live connector testing.

Verification evidence:

- `uv run pytest backend/tests/test_integration_runtime_status.py -v` passed.
- `uv run ruff check backend/app/api/v1/integrations.py backend/app/core/redaction.py backend/tests/test_integration_runtime_status.py` passed.

## 2026-05-02 - Portfolio Case Study Draft

Why it matters:

- The project now has enough architecture and implementation evidence to be
  presented as more than a UI clone or basic RAG demo.
- A dedicated case study helps explain the engineering value: multi-agent
  ownership, LangGraph orchestration, evidence-first review, pgvector-ready RAG,
  cost controls, and connector security boundaries.

Implemented scope:

- Added `docs/portfolio-case-study.md`.
- Structured the story around problem, architecture, agent ownership, cost
  optimization, security/review boundaries, frontend experience, verification,
  and a resume bullet draft.
- Referenced the saved LangGraph graph capture assets.

Verification evidence:

- Documentation-only change reviewed against `AGENTS.md` and the current
  implementation history.

## 2026-05-02 - Playwright Sync Metric Selector Hardening

Why it matters:

- The integration smoke test failed because a broad `Fetched` text lookup also
  matched lower-case `fetched=` text inside recent sync status messages.
- The page itself rendered correctly, but the test selector was too fragile for
  a screen that intentionally shows both metric labels and sync log summaries.

Implemented scope:

- Added `data-testid="sync-result-metrics"` to the integration sync result
  metric grid.
- Scoped Playwright metric assertions to that grid and used exact text
  matching for `Fetched`, `Review items`, and `Skipped`.

Verification evidence:

- `npx eslint src/app/integrations/page.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop`
  passed with 14/14 tests.

## 2026-05-02 - Liquid Glass Frontend Refresh

Why it matters:

- ParaWorks needed a more memorable portfolio-facing visual identity without
  losing its Slack-like business workspace ergonomics.
- The refresh follows Apple Liquid Glass guidance by treating navigation,
  search, and primary controls as a floating functional layer while keeping
  content surfaces readable.

Implemented scope:

- Reworked global visual tokens for translucent panels, glass controls,
  stronger depth shadows, subtle structured background light, and accessibility
  fallbacks for reduced transparency or increased contrast.
- Refreshed `AppShell` with a floating glass sidebar, mobile glass toolbar,
  glass search command surface, and stained-glass primary agent action.
- Applied global surface behavior so existing cards and panels inherit the new
  material without rewriting every page.

Cost/security note:

- This is a frontend-only visual change. It does not trigger connector sync,
  embeddings, or LLM calls.
- Status colors and operational labels remain visible so the design stays
  useful for business users and live connector debugging.

Verification evidence:

- `npm run build` passed.
- `npx eslint src/components/layout/AppShell.tsx` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop`
  passed with 14/14 tests.
- In-app browser screenshot review checked `/dashboard` and `/integrations` at
  the current viewport.

## 2026-05-02 - Liquid Glass Intensity Pass

Why it matters:

- The first Liquid Glass refresh improved the theme, but still read closer to a
  soft translucent dashboard than an iOS-style glass system.
- This pass pushed the material closer to Liquid Glass by adding stronger
  refraction edges, reflective highlights, deeper blur/saturation, and floating
  dock-like navigation surfaces.

Implemented scope:

- Intensified global glass tokens, shadows, background light sheets, and
  refractive edge overlays.
- Added shared pseudo-element highlights to liquid surfaces, dark rails,
  controls, and primary stained-glass actions.
- Upgraded the mobile toolbar into a rounded glass slab and made the desktop
  sidebar/top search feel more like floating system chrome.
- Restored `--workspace-rail-active` to a readable text color after visual QA
  showed page eyebrow labels becoming too faint.

Verification evidence:

- `npx eslint src/components/layout/AppShell.tsx` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop`
  passed with 14/14 tests.
- `npm run build` passed.
- In-app browser screenshot review checked `/dashboard` after the intensity
  pass and contrast fix.

## 2026-05-02 - Dark Liquid Glass Mode

Why it matters:

- Browser QA showed the light Liquid Glass theme was still too bright for
  dense business screens, reducing text readability.
- ParaWorks now defaults to a darker, higher-contrast Liquid Glass experience
  while preserving a light mode toggle for comparison and future demos.

Implemented scope:

- Added `data-theme` based dark/light glass modes with a pre-hydration script
  to avoid a bright first paint.
- Added persistent theme toggles in the sidebar and mobile toolbar.
- Reworked dark-mode glass tokens, page background, panels, controls, status
  surfaces, and hard-coded text/background overrides so existing pages remain
  consistent.
- Added Playwright coverage for switching between dark and light glass modes.

Verification evidence:

- `npx eslint src/app/layout.tsx src/components/layout/AppShell.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop`
  passed with 15/15 tests.
- `npm run build` passed.
- In-app browser screenshot review checked `/dashboard` and `/integrations` in
  dark Liquid Glass mode.

## 2026-05-02 - Gray Purple Dark Glass Palette

Why it matters:

- User feedback clarified that the dark mode should not feel like a navy SaaS
  dashboard. The target palette is charcoal gray first, with white glass glow
  and a Slack-like deep purple accent group.
- This keeps the Liquid Glass look vivid while making the workspace calmer,
  more business-like, and more consistent.

Implemented scope:

- Replaced the dark-mode navy/blue/cyan token group with charcoal gray,
  white-glow, and deep purple glass tokens.
- Updated dark page background, glass controls, panels, primary actions,
  shadows, and hard-coded color overrides to reduce blue cast.
- Verified `/dashboard` and `/integrations` visually in the in-app browser,
  including OAuth/status panel contrast.

Verification evidence:

- `npx eslint src/app/layout.tsx src/components/layout/AppShell.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop`
  passed with 15/15 tests.
- `npm run build` passed.

## 2026-05-02 - Dark Glass Consistency QA Fix

Why it matters:

- Browser QA found that the integrations page still had inconsistent Liquid
  Glass details: cards looked too milky, the language segment active state felt
  flat, some OAuth text used old hard-coded colors, and sync buttons did not
  belong to the same glass system.
- The dark gray and deep purple palette needs consistent contrast and material
  behavior across controls, cards, and status panels.

Implemented scope:

- Added a `liquid-segment-active` material for KO/EN and active mobile/sidebar
  navigation states.
- Added `integration-glass-card` to reduce unnatural white opacity on
  integration cards and keep their glass tone closer to the primary purple
  action.
- Reworked integration sync and agent buttons to use `liquid-primary` and
  `liquid-control` instead of flat dark/white button styles.
- Replaced OAuth status hard-coded text colors with theme token colors so
  contrast stays consistent in dark mode.

Verification evidence:

- In-app browser screenshot review checked `/integrations` in dark mode.
- `npx eslint src/components/layout/AppShell.tsx src/app/integrations/page.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop`
  passed with 15/15 tests.
- `npm run build` passed.

## 2026-05-02 - Integration Runtime Glass Consistency

Implemented scope:

- Unified the `/integrations` task stream panel with the same
  `integration-glass-card` material used by the connector cards.
- Replaced Slack/Google runtime status hard-coded text colors with
  `--ink-strong` so dark-mode contrast follows the Liquid Glass token system.
- Added a reusable `glass-row` surface for runtime rows and sync metrics,
  keeping nested glass elements in the same gray-purple material family.
- Changed runtime mode pills to `liquid-control` so they visually align with
  the top floating controls and primary dark-mode button treatment.

Verification evidence:

- In-app browser screenshot review checked `/integrations` in dark mode.
- `npx eslint src/app/integrations/page.tsx src/components/layout/AppShell.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop`
  passed with 15/15 tests.
- `npm run build` passed.

## 2026-05-02 - Cross-Viewport Theme Token Audit

Implemented scope:

- Compared `/integrations` across desktop/mobile and dark/light modes with
  Playwright computed-style checks.
- Replaced desktop shell hard-coded `text-white/*`, `border-white/*`, and
  white hover states with `--shell-*` theme tokens.
- Added a `shell-rail` glass material so the desktop sidebar becomes a light
  frosted rail in light mode and gray-purple glass in dark mode.
- Aligned mobile language hover states with `--glass-control-strong` instead
  of hard-coded white opacity.
- Added a Playwright regression test that verifies shell chrome changes
  tokens across desktop and mobile theme modes.

Verification evidence:

- Playwright computed-style audit confirmed desktop sidebar changes from
  dark `rgba(18, 17, 21, 0.62)` to light `rgba(255, 255, 255, 0.54)`.
- `npx eslint src/components/layout/AppShell.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile`
  passed with 32/32 tests.
- `npm run build` passed.

## 2026-05-02 - Light Gray Deep Purple Palette

Implemented scope:

- Toned down the light-mode foundation from bright white glass to warm light
  gray glass surfaces.
- Shifted light-mode shell, active accents, primary controls, and outlines
  toward ParaWorks deep purple (`#4a154b`) for stronger brand consistency.
- Reduced mint/blue emphasis in the light-mode background, controls, cards,
  and rows so the UI reads as one coherent gray-purple material system.
- Kept dark-mode tokens unchanged while preserving the shared Liquid Glass
  component structure.

Verification evidence:

- In-app browser screenshot review checked `/integrations` in light mode.
- Playwright computed-style audit confirmed light shell `rgba(228, 225, 235, 0.74)`
  and primary action `rgba(74, 21, 75, 0.9)`.
- `npx eslint src/components/layout/AppShell.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile`
  passed with 32/32 tests.
- `npm run build` passed.

## 2026-05-02 - Light Purple Palette Adjustment

Implemented scope:

- Shifted the light-mode accent system from deep purple to soft lavender and
  light purple while keeping the gray glass foundation.
- Separated active segment behavior so light mode uses dark text on lavender
  glass and dark mode keeps the existing high-contrast deep purple treatment.
- Updated light-mode shell, surface, control, card, and row tint gradients to
  reduce heavy purple saturation and keep the UI calmer.

Verification evidence:

- In-app browser screenshot review checked `/integrations` in light mode.
- Playwright computed-style audit confirmed light shell `rgba(232, 226, 241, 0.76)`
  and primary action `rgba(183, 154, 221, 0.9)`.
- `npx eslint src/components/layout/AppShell.tsx e2e/visual-smoke.spec.ts` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile`
  passed with 32/32 tests.
- `npm run build` passed.

### Agent Cost Budget Guardrails

- Added a reusable agent runtime cost decision that estimates input/output
  token cost before execution and returns `run`, `skip`, or `use_cache`.
- Connected the company memory LangGraph orchestration cost plan to a per-run
  budget limit so large evidence windows can be skipped before calling an LLM.
- Preserved explicit skip reasons such as `no_slack_evidence`,
  `empty_question`, and `budget_exceeded` so the UI/API can explain why an
  agent did or did not run.
- Kept cache hits as a first-class policy outcome so future prompt/result cache
  reuse can avoid paid calls even when the potential token window is large.

Portfolio angle:

- Demonstrates that ParaWorks treats LLM cost as an architecture concern, not a
  post-hoc dashboard metric.
- Gives the three-agent split a shared budget contract, making independently
  developed Slack, Mail/Document, and RAG agents easier to merge safely.
- Supports the final product goal of multi-agent orchestration while protecting
  against expensive repeated sync and re-vectorization patterns.

Verification evidence:

- Added RED tests first for over-budget skip behavior and cache-first budget
  decisions.
- `uv run pytest backend/tests/test_agent_runtime_contracts.py backend/tests/test_company_memory_orchestration_service.py`
  passed with 9/9 tests.
- `uv run ruff check backend/app/agent_runtime backend/tests/test_agent_runtime_contracts.py backend/tests/test_company_memory_orchestration_service.py`
  passed after applying automatic import cleanup.
- `uv run pytest backend/tests/test_agent_runtime_contracts.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_agent_orchestration.py backend/tests/test_agent_runs_api.py`
  passed with 16/16 tests.

### Agent Budget Observability

- Exposed the default per-run agent budget and the supported budget actions
  (`run`, `skip`, `use_cache`) through the company memory orchestration status
  and run APIs.
- Updated the Agent Operations page so operators can see the active per-run
  budget directly beside the LangGraph orchestration and cost guardrail status.
- Added frontend fallback handling so the operations page stays renderable even
  if a running backend still returns the older cost policy shape.
- Kept status API calls free of paid LLM calls while still showing enough budget
  metadata to explain cost behavior before a real run.

Portfolio angle:

- Shows an operator-facing cost control loop: policy, API contract, UI
  visibility, and tests are aligned.
- Makes cost optimization demonstrable during portfolio walkthroughs without
  requiring real paid model calls.

Verification evidence:

- Added API tests first for budget metadata visibility.
- `uv run pytest backend/tests/test_orchestration_api.py` passed with 3/3 tests.
- `uv run ruff check backend/app/api/v1/orchestration.py backend/tests/test_orchestration_api.py`
  passed.
- `npx eslint src/app/agent-runs/page.tsx src/lib/api/types.ts` passed.
- `npm run build` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile`
  passed with 32/32 tests after adding the fallback.

### Global Search Bar Activation

- Converted the desktop sidebar search and floating top search from static
  glass UI into real search forms.
- Both search bars now submit to `/search?q=...`, preserving the Liquid Glass
  visual treatment while making the controls keyboard-friendly.
- Updated the Company Memory search page to read the `q` URL parameter, hydrate
  the input with that query, and immediately run the existing RAG/search flow.
- Added Korean and English placeholders to the shell dictionary so both locales
  show natural search copy.

Portfolio angle:

- Turns visible UX affordances into working product paths without adding a new
  backend surface.
- Demonstrates integration between shell navigation, URL-driven state, and the
  existing RAG/search agent flow.

Verification evidence:

- Added Playwright tests first for sidebar and top search submission.
- Confirmed both new tests failed before implementation because the inputs did
  not exist.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop -g "search submits"`
  passed with 2/2 tests.
- `npx eslint src/components/layout/AppShell.tsx src/app/search/page.tsx src/lib/i18n/dictionary.ts e2e/visual-smoke.spec.ts`
  passed.
- `npm run build` passed after wrapping `useSearchParams` usage in a Suspense
  boundary.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile`
  passed with 34/34 executed tests and 2 expected mobile skips.

### Google OAuth Callback Activation

- Added a generic backend Google OAuth callback route at
  `/api/v1/integrations/google/oauth/callback` that reads the signed state to
  determine whether the returning connection is Gmail, Google Drive, or
  Calendar.
- Added the frontend `/integrations/google/callback` route so Google Cloud's
  shared redirect URI can complete OAuth installs and persist connection
  metadata.
- Kept raw Google access/refresh tokens and internal token refs out of the UI,
  matching the Slack OAuth redaction pattern.
- Added explicit visual coverage for Gmail and Google Drive connect CTAs when
  OAuth is configured, while keeping those CTAs outside the primary sync/action
  row.

Portfolio angle:

- Moves Google integration from a readiness boundary into a real OAuth install
  loop for Gmail and Drive.
- Shows secure connector UX: signed state routing, token redaction, safe local
  error states, and post-callback connection metadata.

Verification evidence:

- Added RED tests first for missing generic Google callback API and missing
  frontend callback route.
- `uv run pytest backend/tests/test_google_oauth.py` passed with 6/6 tests.
- `uv run ruff check backend/app/api/v1/integrations.py backend/tests/test_google_oauth.py`
  passed.
- `npx eslint src/app/integrations/google/callback/page.tsx src/app/integrations/google/callback/GoogleCallbackClient.tsx src/app/integrations/page.tsx e2e/visual-smoke.spec.ts`
  passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop -g "Google"`
  passed with 4/4 tests.
- `npm run build` passed and listed `/integrations/google/callback`.
- `uv run pytest backend/tests/test_google_oauth.py backend/tests/test_integration_runtime_status.py backend/tests/test_google_connector.py`
  passed with 16/16 tests.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile`
  passed with 40/40 executed tests and 2 expected mobile skips.

### Light Deep Purple Theme Tuning

- Replaced the light mode lavender palette with a brighter deep-purple family
  while keeping the base surface tone in light gray.
- Updated light-mode shell, primary action, active segment, glass control, card,
  and row tint gradients so the theme reads closer to Slack-adjacent deep
  purple rather than soft lavender.
- Preserved dark mode tokens and the existing Liquid Glass structure.

Portfolio angle:

- Shows iterative product design judgment: visual direction was adjusted from
  soft lavender to a more confident business-oriented light deep-purple tone
  after browser review.
- Keeps the design system tokenized so future UI changes can be made without
  hardcoding page-by-page color fixes.

Verification evidence:

- `npm run build` passed.
- `npx eslint src/components/layout/AppShell.tsx e2e/visual-smoke.spec.ts`
  passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile`
  passed with 40/40 executed tests and 2 expected mobile skips.

## Demo Login And Admin Permission Foundation

What changed:

- Added a demo auth API with `admin@paraworks.com` plus three employee
  accounts so portfolio demos can switch between admin, internal employee, and
  public-only permission scopes.
- Added `/login` for demo account switching and `/admin` for an admin-only
  user/permission console.
- Aligned the frontend API client with the selected demo user so search, ask,
  and admin APIs use the same permission header instead of hardcoded admin or
  viewer behavior.
- Updated Playwright to use `http://localhost:3000` by default because Next dev
  hydration failed on `127.0.0.1` in this local environment.

Verification evidence:

- `uv run pytest backend/tests/test_auth_api.py backend/tests/test_search_permissions.py backend/tests/test_ask_api.py backend/tests/test_agent_runs_api.py backend/tests/test_knowledge_api.py backend/tests/test_integration_runtime_status.py`
  passed with 20 tests.
- `npx eslint src/app/login/page.tsx src/app/admin/page.tsx src/app/search/page.tsx src/components/layout/AppShell.tsx src/lib/api/client.ts src/lib/api/types.ts src/lib/i18n/dictionary.ts e2e/visual-smoke.spec.ts playwright.config.ts`
  passed.
- `npm run build` passed.
- `npx playwright test e2e/visual-smoke.spec.ts` passed with 50 executed tests
  and 2 expected mobile skips after starting the local smoke backend with a
  SQLite dev DB because the local PostgreSQL password was rejected.

## Workspace Glass Card Consistency

What changed:

- Promoted the integrations page glass-card treatment into shared
  `--card-glass-*` and `--row-glass-*` design tokens.
- Aligned `integration-glass-card`, `liquid-surface`, general `bg-white` cards,
  and `glass-row` helper panels so dark and light modes use the same card
  background, border, highlight, and shadow model.
- Verified computed browser styles for integration, dashboard, and search cards
  in both dark and light modes.

Verification evidence:

- `npm run build` passed.
- `npx playwright test e2e/visual-smoke.spec.ts` passed with 50 executed tests
  and 2 expected mobile skips.

## Pgvector Dev Environment Hardening

What changed:

- Added Postgres and Redis healthchecks to the local Docker stack and made the
  pgvector helper detect an occupied host `5432` before falling back to `5432`.
- Added `scripts/check_pgvector_dev.py` so developers can verify the vector
  extension, vector table, and app indexing state table without guessing whether
  the local DB is ready.
- Added a `-SkipApp` path for DB-only setup, documented the `127.0.0.1` database
  URL convention, and kept frontend dev URLs on `localhost` for stable Next.js
  browser testing.
- Fixed pgvector metadata writes by serializing metadata as JSON before casting
  to `jsonb` in PostgreSQL.
- Made the live pgvector integration test use unique document IDs so incremental
  indexing state does not hide regressions between repeated runs.

Portfolio angle:

- Shows production-minded local infrastructure work: the vector DB path is now
  reproducible, testable, and safer when another local PostgreSQL instance is
  already running.
- Strengthens the RAG story for interviews because ParaWorks can demonstrate
  SQLite smoke mode for quick demos and PostgreSQL + pgvector for the real
  retrieval architecture.
- Keeps future embedding/token costs under control by preserving the incremental
  indexing path while validating that only changed documents need to be written.

Verification evidence:

- `uv run python scripts/check_pgvector_dev.py --database-url postgresql+psycopg://paraworks:paraworks@127.0.0.1:5432/paraworks --expect-app-schema`
  passed against the local pgvector container with vector extension `0.8.2`.
- `uv run pytest backend/tests/test_pgvector_dev_runbook.py backend/tests/test_pgvector_integration.py backend/tests/test_pgvector_store.py backend/tests/test_rag_indexing.py backend/tests/test_rag_indexing_tasks.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_vector_retriever.py -v`
  passed with 37 tests.
- `docker compose config` passed.
- `uv run ruff check backend/app/rag/pgvector_store.py scripts/check_pgvector_dev.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_dev_runbook.py backend/tests/test_pgvector_integration.py backend/tests/test_rag_indexing_tasks.py`
  passed.

## Embedding Cost Preflight Guard

What changed:

- Added a preflight embedding budget gate before paid OpenAI embedding calls in
  the RAG indexing path.
- Added environment-controlled pricing and budget settings:
  `OPENAI_EMBEDDING_INPUT_COST_PER_1M_TOKENS` and
  `RAG_EMBEDDING_MAX_ESTIMATED_COST_USD`.
- Exposed the active RAG indexing cost policy through
  `/api/v1/rag/indexing/summary` and surfaced it in the company memory search
  freshness panel.
- Preserved incremental hash skip behavior so unchanged documents continue to
  avoid embedding requests entirely.

Portfolio angle:

- Shows cost-aware AI engineering: ParaWorks estimates changed-document
  embedding cost before a provider request can spend money.
- Makes the system easier to operate in a three-developer workflow because the
  active budget policy is visible through the API and frontend instead of living
  only in `.env`.
- Strengthens the product story that RAG quality and token-cost discipline are
  designed together, not treated as separate cleanup work.

Verification evidence:

- Added a RED test that failed because the embedding budget exception did not
  exist, then implemented the gate until the test passed.
- Added a RED test for the missing indexing summary `cost_policy`, then exposed
  the API field until the test passed.
- `uv run pytest backend/tests/test_pgvector_dev_runbook.py backend/tests/test_pgvector_integration.py backend/tests/test_pgvector_store.py backend/tests/test_rag_indexing.py backend/tests/test_rag_indexing_tasks.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_vector_retriever.py backend/tests/test_embedding_provider.py -v`
  passed with 39 tests.
- `uv run ruff check backend/app/rag/indexing.py backend/app/rag/reindexing.py backend/app/api/v1/rag.py backend/app/core/config.py backend/tests/test_rag_indexing.py`
  passed.
- `npx eslint src/app/search/page.tsx src/lib/api/types.ts` passed.
- `npm run build` passed.

## RAG Reindex Approval UX

What changed:

- Extended dry-run reindex responses with `embedding_budget`, including changed
  document count, estimated input tokens, estimated cost, budget limit, and the
  resulting budget action.
- Kept dry-run free and non-blocking: over-budget dry-runs return a warning
  preview instead of calling the embedding provider.
- Added a RAG reindex approval panel to `/agent-runs` so operators can run a
  cost preview before approving `dry_run=false` execution.
- Added desktop and mobile Playwright coverage for the preview -> approved run
  interaction.

Portfolio angle:

- Turns backend cost guardrails into an operator-facing workflow, which is more
  compelling than a hidden environment variable.
- Shows responsible AI product design: paid embedding work requires a visible
  estimate and explicit approval.
- Helps a three-developer team integrate safely because RAG indexing behavior is
  observable and test-covered from API to browser.

Verification evidence:

- Added RED tests for missing dry-run `embedding_budget`, then implemented the
  preview response until they passed.
- `uv run pytest backend/tests/test_rag_indexing.py backend/tests/test_rag_indexing_tasks.py -v`
  passed with 22 tests.
- `uv run ruff check backend/app/rag/indexing.py backend/app/rag/reindexing.py backend/tests/test_rag_indexing.py`
  passed.
- `npx eslint src/app/agent-runs/page.tsx src/app/agent-runs/RagReindexControl.tsx src/lib/api/types.ts e2e/visual-smoke.spec.ts`
  passed.
- `npm run build` passed.
- `npx playwright test e2e/visual-smoke.spec.ts --project=chromium-desktop --project=chromium-mobile -g "agent operations previews"`
  passed with 2 tests.

## Google Live Collection Hardening

What changed:

- Upgraded the live Google Web API client so Gmail and Google Drive collection
  can read beyond the first API page.
- Gmail now performs a lightweight list -> detail hydration flow: list message
  ids first, then fetch metadata-only details for `Subject`, `From`, and
  `Date`.
- Google Drive file listing now requests `nextPageToken` and follows it while
  preserving the existing compact fields selection.
- Updated connector tests around bearer-token propagation, Gmail pagination,
  Gmail metadata hydration, Drive pagination, and error handling.

Portfolio angle:

- Moves Gmail and Drive closer to real SaaS evidence ingestion instead of a
  first-page skeleton.
- Preserves the three-developer merge contract because provider pagination is
  hidden behind the same `GoogleConnector` and `SourceEvent` boundary.
- Keeps the cost story explicit: sync fetches only source metadata/content
  needed for review candidates and still does not trigger embeddings or LLM
  calls by itself.

Cost/security note:

- Gmail hydration uses `format=metadata` rather than full message bodies, which
  reduces payload size while keeping timeline author/title/date quality.
- Drive listing keeps a narrow fields projection and does not download file
  contents during connector sync.

Verification evidence:

- Added focused regression tests for paginated Gmail and Drive collection.
- `uv run pytest backend/tests/test_google_connector.py -v` passed with 7
  tests.
- `uv run pytest backend/tests/test_google_connector.py backend/tests/test_connector_factory.py backend/tests/test_google_oauth.py backend/tests/test_integration_runtime_status.py -v`
  passed with 26 tests.
- `uv run ruff check backend/app/connectors/google.py backend/tests/test_google_connector.py`
  passed.

## Slack Incremental Live Sync Cursor

What changed:

- Added a Slack live sync cursor path that derives the latest ingested
  `channel_id` + `ts` per channel from existing source metadata.
- `sync_connector_events` now passes that cursor to connectors that support
  incremental fetching, while older/mock connectors still use `fetch_events()`.
- `SlackConnector` forwards channel cursors to `SlackWebApiClient`, and the web
  client sends Slack `conversations.history` an `oldest` timestamp.
- Updated the Slack runbook with cursor behavior, test policy, and cost notes.

Portfolio angle:

- Shows ParaWorks moving from duplicate-skipping after collection to true
  source-delta collection before downstream work.
- Gives the Slack Agent track a safer merge contract: live Slack sync can evolve
  behind `fetch_events_since(...)` without forcing schema changes or frontend
  churn.
- Strengthens the AI-cost story because fewer repeated source events reach
  review generation, agent drafting, or later RAG indexing.

Cost/security note:

- The cursor lookup is local database metadata only. It does not call Slack,
  LLMs, or embedding providers.
- Slack message bodies still stay out of terminal logs; only channel/timestamp
  metadata is used to narrow the next API window.

Verification evidence:

- Added RED tests for Slack `oldest` handling and ingestion cursor passing,
  then implemented the minimal code until they passed.
- `uv run pytest backend/tests/test_slack_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_connector_factory.py backend/tests/test_integration_runtime_status.py -v`
  passed with 24 tests.
- `uv run ruff check backend/app/connectors/slack.py backend/app/ingestion/sync.py backend/tests/test_slack_connector.py backend/tests/test_connector_ingestion_contract.py`
  passed.

## Slack Live Sync Retry Guardrails

What changed:

- Added bounded retry handling to `SlackWebApiClient` for Slack `429`
  rate-limit responses and transient `5xx` history API failures.
- Honored Slack `Retry-After` headers when present, with a safe default delay
  for transient errors that do not include the header.
- Converted exhausted retry paths into clear `SlackApiError` messages so sync
  endpoints can keep returning controlled failure states.
- Updated Slack runbook guidance and tests for retry behavior.

Portfolio angle:

- Makes the live Slack integration more production-like: API rate limits and
  temporary provider failures are expected operating conditions, not demo-only
  surprises.
- Strengthens the three-developer integration contract because Slack connector
  resilience stays behind the connector boundary and does not leak into agent
  or frontend code.

Cost/security note:

- Retries are intentionally bounded. ParaWorks can recover from transient Slack
  failures without creating unlimited provider calls or cascading into repeated
  review/LLM/embedding work.
- Retry handling does not log message bodies or expose bot tokens.

Verification evidence:

- Added RED tests for Slack rate-limit retry, retry exhaustion, and transient
  server-error recovery.
- Focused retry tests passed after implementation.
- `uv run pytest backend/tests/test_slack_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_connector_factory.py backend/tests/test_integration_runtime_status.py -v`
  passed with 27 tests.
- `uv run ruff check backend/app/connectors/slack.py backend/tests/test_slack_connector.py`
  passed after Ruff applied import/format cleanup.

## LangGraph Evidence Cache Reuse

What changed:

- Added evidence cache planning to the Company Memory LangGraph orchestration
  path.
- The orchestration cost plan now builds Slack, Mail/Docs, and RAG evidence
  packets before execution, computes their evidence cache keys, and checks for
  completed matching `AgentRun` records.
- Unchanged evidence now produces `use_cache` decisions and avoids creating
  duplicate Slack/Mail review candidates or repeated RAG agent runs.
- Exposed `evidence_cache_reuse=true` in the orchestration cost policy API.

Portfolio angle:

- Shows a realistic multi-agent orchestration concern: merging multiple agent
  tracks safely means the orchestrator must decide when not to run agents.
- Strengthens the cost-optimization story because repeated user clicks over
  unchanged Slack/Gmail/Drive/RAG evidence no longer create duplicate agent
  spend or noisy review work.
- Keeps the split between three developers clean: each agent owns its packet
  and cache key contract, while LangGraph owns the run/skip/cache decision.

Cost/security note:

- Cache planning is local database lookup plus deterministic evidence hashing.
  It does not call Slack, Google, embeddings, or paid LLM APIs.
- Cached decisions still preserve the explicit POST execution boundary; status
  and dry-run endpoints remain zero-cost.

Verification evidence:

- Added a RED test proving a second identical Company Memory run should use
  cache and create no new `AgentRun`/`ReviewItem` records.
- Added a RED API test for `evidence_cache_reuse` in orchestration cost policy.
- `uv run pytest backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_agent_runs_api.py -v`
  passed with 11 tests.
- `uv run pytest backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py -v`
  passed with 7 tests after Ruff cleanup.
- `uv run ruff check backend/app/agent_runtime/company_memory.py backend/app/api/v1/orchestration.py backend/app/agents/slack_agent/__init__.py backend/app/agents/rag_orchestrator_agent/__init__.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py`
  passed.

## Admin Audit Log Foundation

What changed:

- Added an `AuditLog` model and admin-only `/api/v1/admin/audit-logs` API.
- Recorded audit events for review approval, bulk agent-candidate approval,
  review reject/more-evidence actions, connector sync, agent review runs,
  Company Memory LangGraph runs, and RAG reindex execution/job creation.
- Added sanitized audit metadata so operational context is visible without
  exposing tokens or secret references.
- Extended `/admin` with a recent audit log panel using the same Liquid Glass
  card system as the rest of the workspace.

Portfolio angle:

- Shows service maturity beyond feature demos: important operational actions
  are now attributable to an actor, target, timestamp, and metadata.
- Strengthens the three-developer workflow because merged AI-generated work can
  be reviewed through a shared audit trail instead of scattered terminal logs.
- Makes permission design more concrete: employees cannot read audit logs,
  while admins can inspect workspace activity from the product UI.

Cost/security note:

- Audit writes are local database operations. They do not call Slack, Google,
  embeddings, or LLM APIs.
- Audit metadata is sanitized before persistence so token-like strings are not
  rendered in the admin console.

Verification evidence:

- Added RED tests first for missing `AuditLog` model/API and key action audit
  records.
- `uv run pytest backend/tests/test_audit_logs.py backend/tests/test_auth_api.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_orchestration_api.py backend/tests/test_integration_runtime_status.py -v`
  passed with 23 tests.
- `uv run pytest backend/tests/test_audit_logs.py -v` passed with 6 tests after
  Ruff cleanup.
- `uv run ruff check backend/app/models/audit.py backend/app/services/audit.py backend/app/api/v1/admin.py backend/app/api/v1/router.py backend/app/api/v1/review.py backend/app/api/v1/integrations.py backend/app/api/v1/orchestration.py backend/app/api/v1/rag.py backend/tests/test_audit_logs.py`
  passed.
- `npx eslint src/app/admin/page.tsx src/lib/api/types.ts` passed.
- `npm run build` passed and included `/admin`.

## RAG Ranked Citation Quality

What changed:

- Reworked keyword retrieval from exact full-query substring matching to
  term-based candidate scoring.
- Search and Ask responses now include ranked citations with source id, source
  URL, source type, permission level, snippet, relevance score, and matched
  query terms.
- Ask keeps restricted evidence hidden while still reporting hidden match
  counts and returning only visible citations to the current user.
- Updated the Company Memory search UI to show citation scores and matched
  terms beside answer/search evidence.

Portfolio angle:

- Makes the RAG layer explainable: users can see why evidence appeared, not
  only that an answer was generated.
- Demonstrates permission-aware retrieval quality, including visible citation
  filtering and hidden-match disclosure.
- Strengthens the final product story because Slack/Gmail/Drive/approved
  knowledge can now flow into answerable, cited company memory.

Cost/security note:

- Ranking and citation generation are deterministic local scoring operations.
  They do not call embedding providers or paid LLMs.
- Permission checks still run before citations are returned, so restricted
  source URLs/snippets are not exposed to employee viewers.

Verification evidence:

- Added RED tests for ranked search citations and Ask citations with hidden
  restricted matches.
- `uv run pytest backend/tests/test_rag_quality.py backend/tests/test_search_permissions.py backend/tests/test_ask_api.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_vector_retriever.py -v`
  passed with 16 tests.
- `uv run ruff check backend/app/api/v1/search.py backend/app/api/v1/ask.py backend/app/agents/rag_orchestrator_agent/agent.py backend/app/agents/rag_orchestrator_agent/service.py backend/tests/test_rag_quality.py`
  passed.
- `npx eslint src/app/search/page.tsx src/lib/api/types.ts` passed.
- `npm run build` passed and included `/search`.

## Google Live Delta And Retry Guardrails

What changed:

- Added Gmail and Google Drive incremental cursor support to the live Google
  connector.
- Gmail collection now sends an `after:<unix_seconds>` query from the last
  stored message `internalDate`.
- Drive collection now sends a `modifiedTime > '<timestamp>'` query from the
  last stored file modification time.
- Google source events now persist common `sync_partition` and `sync_cursor`
  metadata so ingestion can resume without connector-specific database logic.
- Added bounded retry handling for Google API 429 and 5xx responses, including
  `Retry-After` support.

Portfolio angle:

- Shows the project is moving from demo integration buttons toward production
  ingestion behavior: delta fetch, retry, and observable failure boundaries.
- Makes the Google track easier for another developer to own because the
  ingestion cursor contract is shared with Slack rather than hidden in one
  connector.

Cost/security note:

- The connector fetches only source deltas before review, agent execution, or
  embedding work. This prevents every sync from reprocessing unchanged Gmail
  and Drive content.
- Retry is bounded, so rate limit or server-side failures do not create runaway
  API usage.

## Whole-App Playwright Regression Matrix

What changed:

- Added a route inventory guard that fails when a new `app/**/page.tsx` route is
  not represented in Playwright coverage.
- Added a whole-page regression matrix for desktop and mobile, dark and light
  modes, including `/`, static pages, OAuth callback pages, and the dynamic
  agent run detail page.
- The matrix checks that each page mounts the workspace shell, avoids Next error
  screens, has visible glass surfaces, stays nonblank, and does not introduce
  horizontal viewport overflow.
- Added an AppShell hydration marker so interaction tests wait for real React
  handlers before clicking theme toggles or submitting global search.
- Hardened existing smoke tests around hydration, route interception, and admin
  audit-log text duplication.
- Fixed Search page compatibility with older Ask responses that do not include
  `citations`, and removed duplicate React keys in search evidence rendering.

Portfolio angle:

- Turns browser QA from manual spot checks into repeatable desktop/mobile
  coverage for every current Next.js page.
- Demonstrates integration discipline: new routes must be added to the
  regression inventory instead of silently escaping visual smoke coverage.

Cost/security note:

- The new page regression matrix validates UI and local route health only; it
  does not trigger paid LLM or embedding calls.
- Existing cost-related smoke tests still mock dry-run and approval responses
  so CI-style browser checks remain deterministic.

## Google Source Quality And Calendar Delta Sync

What changed:

- Gmail live collection now hydrates messages with `format=full` and extracts
  bounded `text/plain` payload content for review/RAG instead of relying only on
  snippets.
- Gmail source metadata now records thread id, labels, date header, body source,
  and whether the extracted body was truncated for ingestion safety.
- Google Drive source events now include description, owner, last modifier,
  created time, modified time, and richer searchable body text.
- Google Calendar live collection now paginates events and supports `updatedMin`
  incremental sync through the shared `sync_partition` / `sync_cursor` contract.
- Calendar source events now include description, location, start/end time,
  attendee count, and reusable sync cursor metadata.

Portfolio angle:

- Moves Google integrations beyond "connected" status into useful business
  memory ingestion: mail context, document metadata, and calendar timelines now
  carry enough structure for review and RAG.
- Shows cross-connector consistency because Gmail, Drive, and Calendar all share
  the same incremental cursor pattern.

Cost/security note:

- Gmail body extraction is bounded before review and embedding stages, reducing
  the risk of large messages driving unnecessary downstream token cost.
- Calendar sync uses `updatedMin` so repeated syncs avoid full event history
  collection.

## Source Evidence Review Drawer

What changed:

- Review Queue responses now include structured `source_evidence` rows with
  source URL, snippet, permission level, confidence score, rank, importance
  score, source id, author/timestamp when available, and originating AgentRun.
- The Review UI drawer now presents source evidence as reviewer-ready cards
  instead of only raw links and snippets.
- The "request more evidence" workflow now captures a reviewer note and stores
  it in the ReviewItem payload before moving the item to
  `needs_more_evidence`.

Portfolio angle:

- Makes the human-in-the-loop trust boundary more concrete: reviewers can see
  exactly what evidence supports an AI-generated timeline, history, decision,
  or todo candidate.
- Shows practical product ownership for Track C because orchestration output is
  now reviewable by Korean business users, not only visible in backend logs.

Cost/security note:

- Structured evidence is assembled from already persisted ReviewItem and
  AgentRun metadata. It does not call Slack, Google, embeddings, or paid LLMs.
- The Drawer preserves permission labels and source snippets so reviewers can
  reject or request more evidence before any candidate becomes trusted
  knowledge.

## LangGraph HITL Checkpoint Strategy

What changed:

- Company Memory orchestration now emits a structured `hitl_checkpoint` output
  from the `draft_review_candidates` node.
- The checkpoint records the Review Queue as the current HITL store, the target
  ReviewItem ids, required statuses, resume node, resume policy, and whether
  trusted knowledge requires human approval.
- The orchestration status cost policy now explicitly reports
  `hitl_checkpointing`, `checkpoint_store=review_queue`, and
  `trusted_knowledge_requires_approval`.

Portfolio angle:

- Shows that ParaWorks' LangGraph flow is not a black-box automation pipeline:
  it has an explicit human review stop before generated memory becomes trusted
  organizational knowledge.
- Gives the three-developer team a stable integration contract for future
  long-running checkpoint/resume work without changing each agent's local
  implementation.

Cost/security note:

- HITL checkpoint metadata is generated from local ReviewItem ids and
  orchestration state. It does not call paid LLMs, embeddings, Slack, or Google.
- The checkpoint keeps the trust boundary visible: generated outputs can be
  reviewed, rejected, or marked as needing more evidence before promotion.

## Quality And Permission Regression Suite

What changed:

- Added a focused backend regression suite for ParaWorks' core trust promises.
- The suite verifies that source-less Review Queue items cannot be approved.
- It verifies that employee/viewer RAG responses report hidden restricted
  matches without leaking restricted snippets or citations.
- It verifies that Company Memory orchestration emits a Review Queue HITL
  checkpoint without triggering paid calls.
- It verifies that cache-hit orchestration runs do not duplicate AgentRun or
  ReviewItem records.

Portfolio angle:

- Converts product principles into executable tests: evidence-first, permission
  safe, cost-aware, and human-reviewed.
- Gives the three-developer team a shared safety net before connector quality
  and structured LangChain outputs become more complex.

Cost/security note:

- The regression suite uses deterministic local fixtures and fake harness
  models. It does not call live Slack, Google, OpenAI, Gemini, embeddings, or
  external APIs.
- The tests make cost control observable by asserting cache reuse and no
  duplicate review/agent records on unchanged evidence.

## Cross-Agent Evidence Summary Metadata

What changed:

- Added a shared `build_evidence_summary` helper for turning `EvidencePacket`
  messages into AgentRun evidence summary rows.
- Mail/Document Agent runs now persist source id, URL, source type, timestamp,
  author, permission, rank, importance score, and snippet metadata.
- Track C Timeline/History/Decision/Todo extraction runs now persist the same
  evidence summary metadata, so Review Drawer rows can become richer beyond
  Slack-only candidates.

Portfolio angle:

- Strengthens the three-track architecture because Drawer evidence is no
  longer a Slack-specific affordance; Mail/Docs and orchestration-owned memory
  agents now expose the same review/debug metadata.
- Makes future LangChain structured-output replacement safer because the
  Review UI depends on shared EvidencePacket-derived metadata rather than each
  agent inventing local evidence shapes.

Cost/security note:

- Evidence summaries are derived from already-selected evidence packets and
  stored with AgentRun metadata. They do not trigger extra provider calls.
- Permission labels remain attached to each evidence row for reviewer and RAG
  safety checks.

## Search Retrieval Backend Alignment

What changed:

- Verified that `/search` page calls both `/api/v1/ask` and `/api/v1/search`.
- Before this update, `/api/v1/ask` could use pgvector behind the feature flag,
  while `/api/v1/search` always used deterministic lexical ranking.
- Added a shared pgvector search adapter builder and wired `/api/v1/search` to
  use the same pgvector feature-flag path when PostgreSQL, pgvector search flag,
  and an OpenAI embedding key are available.
- Search responses now disclose `retrieval_backend` and cost policy metadata,
  and the `/search` UI shows whether the current result used pgvector or the
  zero-cost deterministic search path.

Portfolio angle:

- Makes RAG behavior explainable to users and interviewers: answer generation
  and evidence search now report which retrieval path they used.
- Shows cost-aware product design because query-time embedding calls are
  explicit instead of hidden behind a generic search button.

Cost/security note:

- Default SQLite/demo mode remains `deterministic_lexical` with no embedding or
  paid LLM call.
- pgvector search performs a query embedding only when
  `RAG_USE_PGVECTOR_SEARCH=true`, PostgreSQL is active, and `OPENAI_API_KEY` is
  configured.
- Permission filtering and hidden-match accounting remain enforced in both
  retrieval paths.

## Slack Thread Context-Aware Chunking

What changed:

- Slack thread replies now preserve parent-message context in the SourceEvent
  body before ingestion creates the document chunk.
- Reply metadata now records `thread_parent_text`, `thread_reply_index`, and
  `thread_context_window=parent_plus_reply`.
- The connector still fetches thread replies incrementally from the channel
  cursor, so this quality improvement does not require re-fetching entire
  channel history by default.

Portfolio angle:

- Improves evidence quality for real collaboration data: short replies such as
  "동의합니다" or "좋아요" become useful to agents/RAG because the parent
  decision context travels with the reply chunk.
- Strengthens Track A ownership by making Slack ingestion more agent-ready,
  not just API-connected.

Cost/security note:

- This is deterministic preprocessing over already fetched Slack events. It
  does not call Slack more than the existing reply fetch, and it does not call
  LLMs or embeddings.
- Parent context is bounded to one parent message plus one reply, avoiding
  whole-thread prompt inflation.

## Gmail Thread And Domain Metadata Quality

What changed:

- Gmail SourceEvents now parse participants from From, To, and Cc headers.
- Gmail metadata now records `thread_context_key`, `from_domain`,
  `participant_domains`, `external_domains`, and
  `has_external_participants`.
- Existing body extraction, truncation, label ids, thread id, and delta cursor
  behavior remain intact.

Portfolio angle:

- Makes Gmail evidence more useful for business review: agents can distinguish
  internal-only messages from customer/vendor-involved threads.
- Supports future permission and routing policies without hard-coding Gmail
  parsing logic inside agent implementations.

Cost/security note:

- This is local header parsing over already fetched Gmail message payloads.
  It does not add Google API calls, LLM calls, or embedding calls.
- Domain metadata enables safer future filtering while keeping raw content
  behind the existing Review/RAG permission checks.

## Drive Parser Status And Version Metadata

What changed:

- Google Drive SourceEvents now preserve `parser_name`, `parser_status`,
  `parser_status_reason`, `document_version`, `revision_id`, and
  `content_signature`.
- Drive API collection now requests `version` and `headRevisionId` so future
  parser/indexing work can decide whether content actually changed.
- The current parser status is explicit as `metadata_only`, matching the
  harness stage before full file export/parsing is enabled.

Portfolio angle:

- Shows product-quality ingestion design: document evidence carries parser and
  version provenance instead of appearing as anonymous text.
- Supports later incremental parsing, embedding skip decisions, and reviewer
  trust signals without changing agent contracts.

Cost/security note:

- This adds metadata fields to the existing Drive files list request; it does
  not export document bodies, call LLMs, or call embedding APIs.
- `content_signature` gives the future indexer a cheap guardrail for skipping
  unchanged Drive files before paid embedding work.

## Calendar Event Quality Metadata

What changed:

- Calendar SourceEvents now preserve `event_context_key`, `event_status`,
  organizer/creator emails, `recurring_event_id`, attendee response counts,
  attendee domains, external domains, and event duration.
- Participants still come from attendee emails, but metadata now explains who
  accepted, declined, or has not responded.
- The connector keeps the same delta sync boundary through the event `updated`
  cursor.

Portfolio angle:

- Makes calendar evidence more useful for Korean business review flows:
  meetings can be understood as internal/external, confirmed/cancelled, and
  time-bounded evidence.
- Gives future Timeline/History agents better deterministic signals before
  spending LLM tokens.

Cost/security note:

- This is local metadata derivation from already fetched Calendar event
  payloads. It adds no Google calls, LLM calls, or embedding calls.
- External-domain flags support safer future permission and disclosure
  policies without leaking hidden event content.

## Connector Golden Dataset Fixtures

What changed:

- Added `backend/tests/fixtures/connector_golden_payloads.json` covering
  Slack, Gmail, Drive, and Calendar payloads.
- Added a regression test that asserts each connector preserves agent-ready
  metadata: Slack thread context, Gmail external domains, Drive parser/version
  metadata, and Calendar RSVP/duration/external-domain metadata.
- The fixture is intentionally deterministic and local, so it can run in every
  developer and coding-assistant workflow.

Portfolio angle:

- Demonstrates team-scale AI-assisted development discipline: connector quality
  is measured by stable examples, not only by manual UI inspection.
- Gives three developer tracks a shared contract for evidence metadata before
  they build more source-specific agents and RAG evaluation.

Cost/security note:

- Golden tests use static local payloads and make no Slack, Google, LLM, or
  embedding calls.
- The fixture protects future cost optimizations such as hash/signature skips
  by keeping metadata expectations explicit.

## RAG Precision Recall Smoke Metrics

What changed:

- Added `backend/app/rag/evaluation.py` with deterministic retrieval metrics:
  precision@k, recall@k, hit rate, expected/retrieved counts, and matched
  expected source ids.
- Added `backend/tests/fixtures/rag_smoke_eval_cases.json` and a smoke test
  that seeds known chunks, runs deterministic retrieval, and verifies the
  expected sources are recovered.
- The test complements `/search` backend disclosure by measuring whether the
  zero-cost retrieval path still finds the right evidence.

Portfolio angle:

- Shows evaluation-minded RAG engineering: retrieval quality is tracked with a
  repeatable smoke metric before adding more expensive model-based evaluation.
- Gives interview/demo material for explaining why ParaWorks avoids blind LLM
  calls and validates evidence selection first.

Cost/security note:

- The smoke metric uses local fixtures and deterministic retrieval only.
  It makes no paid LLM, embedding, Slack, or Google calls.
- This is the correct first guardrail before enabling broader pgvector or
  model-judge evaluation.

## Structured LangChain Memory Extraction Adapter

What changed:

- Added a Track C `LangChainMemoryExtractionModel` adapter that implements the
  existing `MemoryExtractionModel` contract.
- The adapter uses `chat_model.with_structured_output` with a Pydantic
  `StructuredMemoryExtractionOutput` schema, keeping Timeline/History/Decision
  Record/Todo extraction behind the same deterministic agent interface.
- Added prompt rendering with bounded evidence windows and source metadata.

Portfolio angle:

- Shows the correct migration path from deterministic harness logic to real
  LangChain structured-output agents without breaking shared contracts.
- Demonstrates that agent implementation can evolve independently while Review
  Queue, cost accounting, and evidence-first contracts remain stable.

Cost/security note:

- No live model provider is invoked by default. The adapter accepts an injected
  chat model and is covered by fake-model tests.
- Evidence rendering is bounded by `max_input_chars`, preserving the project
  rule that full source sync does not mean full LLM input.

## Product Memory Pages

What changed:

- Expanded `/api/v1/knowledge` to include approved Timeline records alongside
  Decisions, History, and Todos.
- Rebuilt `/knowledge` as an approved company-memory overview with collection
  cards and latest approved records.
- Added `/decisions`, `/timeline`, and `/history` pages backed by the same
  Knowledge API and shared glass-card memory component.
- Extended Playwright route inventory and clean-render checks to include the
  new pages.

Portfolio angle:

- Makes the multi-agent result visible as a product, not only as backend
  Review Queue rows: approved decisions, timelines, and history now have
  browsable surfaces.
- Shows the Review Queue trust boundary end-to-end: candidate -> approval ->
  trusted knowledge -> product memory page -> RAG/search evidence.

Cost/security note:

- These pages are read-only API views and do not trigger LLMs, embeddings, or
  sync jobs.
- They reuse reviewed records and preserve permission/confidence/source
  metadata for every card.

## Production Auth Plan

What changed:

- Added `docs/superpowers/runbooks/production-auth.md`.
- Documented the migration from demo `X-Demo-User` auth to httpOnly cookie
  sessions with rotating refresh tokens.
- Defined backend tables, auth endpoints, frontend API-client changes,
  CSRF/rate-limit/audit guardrails, and permission-model alignment.

Portfolio angle:

- Shows that ParaWorks is being built toward a real company product, not a
  demo-only harness.
- Connects auth design to the core product promise: permission-safe company
  memory and source evidence.

Cost/security note:

- Auth checks must remain cheap session/database reads and must never trigger
  LLM calls, embeddings, connector sync, or RAG reindexing.

## Deployment Runbook

What changed:

- Added `docs/superpowers/runbooks/deployment.md`.
- Documented production components: Next.js, FastAPI, PostgreSQL + pgvector,
  Redis, Celery worker, Slack/Google OAuth, and optional parser/object storage.
- Added deployment order, verification commands, cost gates, rollback plan,
  monitoring checklist, and production readiness checklist.

Portfolio angle:

- Shows that ParaWorks has a credible path from local harness to deployable
  company-memory product.
- Makes infrastructure choices explainable: Docker/Postgres/Redis are for
  production parity, pgvector search, background indexing, and reliable sync.

Cost/security note:

- Deployment checklist keeps budget gates active and separates status/sync
  endpoints from paid LLM or embedding work.
- Secrets are explicitly kept out of git and moved to the deployment secret
  manager.

## Review And AgentRun Notifications

What changed:

- Added `/api/v1/notifications` as a derived alert API.
- Notifications summarize pending Review Queue items, items needing more
  evidence, and recent non-complete AgentRuns.
- Added `/notifications` frontend page and sidebar navigation entry.
- Added Playwright route inventory coverage for the new page.

Portfolio angle:

- Improves the operational product loop: users can see what needs attention
  without opening every review or agent-run page manually.
- Keeps notifications tied to the trust workflow rather than generic activity
  noise.

Cost/security note:

- Notifications are read-only database summaries and do not call LLMs,
  embeddings, Slack, Google, sync jobs, or reindex jobs.

## Knowledge Map

What changed:

- Added `/api/v1/knowledge/map` as a read-only graph endpoint over approved
  Decisions, Timeline, History, and Todo records.
- The map creates memory nodes, source-evidence nodes, and `supported_by`
  edges from stored source links.
- Added `/knowledge-map` frontend page, sidebar navigation, and route
  regression coverage.
- The Knowledge Library now links to the map as a product-facing memory view.

Portfolio angle:

- Shows the core ParaWorks story visually: AI-generated memory is only useful
  when users can inspect which evidence supports each decision, timeline, or
  history record.
- Reinforces the 3-track architecture because Track C can render trusted
  company memory without importing source-specific agent internals.

Cost/security note:

- Knowledge Map only reads approved database records and source-link metadata.
  It does not call LLMs, embeddings, Slack, Google, sync jobs, or reindex jobs.
  Restricted memory nodes keep their restricted permission label, and shared
  evidence nodes use the strictest connected permission level.

## Production Auth Cookie Slice

What changed:

- Added persistent `auth_users` and `refresh_tokens` models.
- `POST /api/v1/auth/login` now issues httpOnly `paraworks_session` and
  `paraworks_refresh` cookies while preserving demo account selection.
- `GET /api/v1/auth/me` now prefers the signed session cookie over
  `X-Demo-User`; demo headers remain available only as a fallback in demo mode.
- Added `POST /api/v1/auth/refresh` with refresh-token rotation and
  `POST /api/v1/auth/logout` with refresh family revocation and cookie clearing.
- Frontend API calls now include credentials so cookie-authenticated requests
  work through the Next.js API rewrite.

Portfolio angle:

- Shows the migration path from a demo harness to production-style auth without
  breaking the MVP flow.
- Demonstrates security-conscious incremental delivery: httpOnly cookies,
  hashed refresh tokens, rotation, revocation, and fail-closed production mode.

Cost/security note:

- Auth is a cheap database/session lookup path. It does not call LLMs,
  embeddings, connector sync, RAG retrieval, or reindex jobs.

## Portfolio Demo Script

What changed:

- Added `docs/superpowers/runbooks/portfolio-demo-script.md`.
- The script walks through login, integrations, AgentRun observability, Review
  Queue evidence inspection, approved knowledge pages, Knowledge Map, and
  permission-aware RAG.
- Added explicit cost and security talking points for portfolio recording.

Portfolio angle:

- Turns the implementation into a coherent story: evidence-first AI, human
  review, company memory, permission safety, and cost-aware orchestration.

Cost/security note:

- The script instructs future demos to keep live provider calls behind
  preflight and explicit confirmation, and to avoid exposing secrets.

## Azure OpenAI-Compatible Provider Alias

What changed:

- Added an Azure integration design spec for Azure Container Apps,
  PostgreSQL pgvector, Redis, Key Vault, Managed Identity, and provider rollout.
- Added `azure_openai` as a valid Slack LLM provider-order alias.
- The first `azure_openai` slice intentionally reuses the existing
  `OPENAI_API_KEY` and OpenAI-compatible chat path so the user can swap keys
  without code changes.
- Added an OpenAI-compatible embedding config helper that also accepts the
  `azure_openai` alias.
- Updated the deployment runbook with the Azure target mapping and current
  alias boundary.

Portfolio angle:

- Shows cloud-readiness without prematurely spending Azure budget or committing
  secrets.
- Keeps the model-provider boundary testable and replaceable before real Azure
  endpoint/deployment variables are introduced.

Cost/security note:

- The Azure alias does not call providers during status/preflight checks.
  Actual paid LLM runs still require preflight and explicit confirmation.
  No Azure, OpenAI, Slack, Google, or database secrets are committed.

## Commit Timeline

- `091c21f feat: add Korean UX and messenger MVP`
- `82e76d1 chore: add SQLite smoke mode`
- `b68caaa feat: persist messenger data`
- `53be213 feat: send messenger items to review`
- `e90d4f9 feat: prepare Slack connector boundary`
- `ce5c23e docs: define agentic Slack timeline slice`
- `1667aba feat: add agent runtime contracts`
- `8fe0190 feat: add agent registry contract`
- `e15ad16 feat: refresh workspace UI`
- `65b36ac feat: add slack agent skeleton`
- `39f96c9 feat: connect slack agent to review queue`
- `7b0a6f5 feat: expose slack agent review action`
- `924f9d8 feat: improve agent-aware review UI`
- `e7c6927 feat: persist agent run metadata`
- `e53bec0 feat: add mail document agent slice`
- `79e7bc7 feat: expose mail docs agent in integrations`
- `af3c1f0 feat: add rag orchestrator agent`
- `2b377fb feat: add company memory ask ui`
- `15e1864 feat: add agent run observability`
- `b90a709 feat: persist rag agent runs`
- `870813c feat: promote approved review items`
- `84707e2 feat: add knowledge library`
- `6f6deab feat: use approved knowledge in rag`
- `3161dff feat: add agent run detail view`
- `9381bb1 fix: isolate smoke frontend cache`
- `aee1e04 feat: add agent run operations summary`
- `9f3a7b8 feat: add review vector orchestration foundations`
- `9e397f4 feat: add pgvector rag adapter`
- `feat: add rag vector indexing pipeline`
- `feat: add incremental vector indexing`
- `feat: add embedding provider and vector retrieval path`
- `feat: show rag indexing observability`
- `chore: document pgvector dev path`
- `feat: queue rag indexing jobs with celery`
- `feat: add slack live connector boundary`
- `feat: add slack oauth installation boundary`
- `feat: show slack oauth connection status`
- `feat: wire installed slack connection sync`
- `fix: harden frontend route smoke`
- `feat: add google oauth boundary`
- `feat: add google installed sync boundary`
- `feat: add langgraph orchestration foundation`
- `feat: expose langgraph orchestration api`
- `feat: show langgraph orchestration status`
- `feat: add langgraph dry-run operations ux`
- `feat: run agents through langgraph`
- `feat: bulk approve agent candidates`
- `feat: show slack runtime status`
- `feat: show google runtime status`
- `feat: add execution cost plan`
- `fix: redact runtime status secrets`
- `docs: add portfolio case study`
- `test: harden integration sync smoke selector`
- `feat: add liquid glass frontend theme`
- `feat: intensify liquid glass theme`
- `feat: add dark liquid glass mode`
- `style: tune dark glass gray purple palette`
- `style: refine dark glass consistency`
- `style: unify integration runtime glass`
- `style: tokenize shell theme chrome`
- `style: tune light gray purple palette`
- `style: soften light purple palette`
- `feat: add agent cost budget guardrails`
- `feat: expose agent budget observability`
- `feat: activate global search bars`
- `feat: activate google oauth callback`
- `style: tune light deep purple palette`
- `feat: add demo login and admin console`
- `style: unify workspace glass cards`
- `chore: harden pgvector dev path`
- `feat: gate paid embedding reindex cost`
- `feat: add rag reindex approval ux`
- `feat: harden google live collection`
- `feat: add slack incremental sync cursor`
- `feat: add slack live sync retry guardrails`
- `feat: reuse cached langgraph evidence`
- `feat: add admin audit logs`
- `feat: improve rag citation ranking`
- `feat: harden google live sync deltas`
- `test: expand whole-app playwright regression`
- `feat: enrich google live source quality`
- `feat: strengthen slack live agent handoff`
  - Slack sync can now receive selected channel IDs from the integrations UI while keeping `.env` channel IDs as the default safe fallback.
  - Slack live collection now follows thread replies incrementally from the same channel cursor, avoiding full-thread re-vectorization/reprocessing on every sync.
  - Slack runtime status exposes channel options, latest sync counts, actionable Slack error hints, and whether synced Slack sources are ready for agent testing.
  - Verification: backend suite `185 passed, 1 skipped`; frontend lint/build passed; Playwright integrations desktop dark/light regression passed.
- `test: validate slack live sync path`
  - Switched local smoke mode to live, restarted backend/frontend, and verified backend health reported `demo_mode=False`.
  - Executed Slack live sync for the configured selected channel; Slack API path completed successfully with no new delta events.
  - Ran Slack Agent review on existing synced Slack sources; one review candidate was created and runtime status reported agent testing readiness.
  - Verification: Playwright integrations desktop dark/light regression passed in live mode.
- `feat: add slack real llm adapter guardrails`
  - Added a LangChain-based Slack Agent adapter with OpenAI as the primary provider and Gemini as a fallback provider chain.
  - Added paid-run preflight that reports provider availability, estimated tokens, estimated cost, budget status, and requires explicit confirmation before live LLM calls.
  - Kept the deterministic Slack Agent as the default safe harness while exposing a separate real LLM test action in the integrations UI.
  - Verification: backend suite `191 passed, 1 skipped` with demo-mode override; frontend lint/build passed; Playwright integrations desktop dark/light regression passed.
- `fix: make slack llm preflight conservative`
  - Ran one confirmed real Slack LLM test with OpenAI primary and Gemini fallback configured; it created one review candidate and persisted an AgentRun.
  - Found the first preflight underestimated Korean/Slack JSON token usage, then tightened prompt caps and changed input-token estimation to a conservative character-count floor.
  - After the fix, the same live Slack evidence window is blocked as `over_budget` instead of allowing another paid run under an optimistic estimate.
  - Verification: backend suite `193 passed, 1 skipped` with demo-mode override; frontend lint passed; Playwright integrations desktop dark/light regression passed.
- `feat: bound slack llm evidence window`
  - Limited paid Slack LLM runs to a recent evidence window instead of sending every synced Slack message to the model.
  - Added shared windowing for preflight and paid execution so the estimated input and actual prompt use the same bounded packet.
  - Added UI visibility for evidence message count and kept the conservative budget cap, enabling a live run over 12 recent Slack messages within the configured budget.
  - Verification: backend suite `195 passed, 1 skipped` with demo-mode override; frontend lint/build passed; Playwright integrations desktop dark/light regression passed.
- `feat: rank slack llm evidence`
  - Replaced the temporary recent-only paid Slack LLM window with deduped, importance-ranked evidence selection while keeping full Slack sync unchanged.
  - Ranking now prioritizes decision, action, cost, technical, thread, and recency signals; duplicate message bodies collapse before top-k selection.
  - Preflight and paid execution use the same ranked source window, and prompt rendering dynamically shrinks evidence text to stay inside the configured per-run cost budget.
  - Verification: backend suite `197 passed, 1 skipped`; frontend build passed; Playwright integrations desktop dark/light regression passed; live preflight returned `slack:live:ranked:12` at `$0.000966 / $0.001`.
- `feat: expose ranked evidence in orchestration`
  - Ran a confirmed live ranked Slack LLM test; the persisted AgentRun used `slack:live:ranked:12` and actual usage was 2,525 tokens at about `$0.000435`.
  - AgentRun records now store a compact ranked evidence summary, and the detail API/UI promote rank, score, source, permission, and snippet for review/debugging.
  - LangGraph company-memory orchestration now uses the same ranked Slack evidence window and exposes source window, selection strategy, evidence count, and cost plan metadata.
  - Verification: backend suite `199 passed, 1 skipped`; frontend build passed; AgentRun desktop/mobile Playwright regression passed; local orchestration API returned `orchestrated-slack:ranked:12` at `$0.000104 / $0.001`.
- `feat: add track c extraction boundaries`
  - Added deterministic Track C agents for Timeline, History, Decision Record, and Todo extraction plus a Validation gate before Review Queue persistence.
  - Extended `ReviewCandidate` with structured payload fields so each candidate can preserve type-specific fields such as `decision_summary`, `result_summary`, `reason`, `priority`, and `priority_reason`.
  - LangGraph company-memory orchestration now runs Track C extraction after source-specific agents create fresh review candidates, while cache-hit runs avoid duplicate candidate generation.
  - Verification: backend suite `203 passed, 1 skipped`; Ruff passed; local orchestration API returned cache-safe `memory_review_items_created=0` when source agents reused cached evidence.
- `feat: add review source evidence drawer`
  - Review Queue API exposes structured source evidence and originating AgentRun metadata for Drawer rendering.
  - Reviewers can request more evidence with a note, preserving why the candidate was not ready for approval.
- `feat: add orchestration hitl checkpoint policy`
  - Company Memory orchestration now emits Review Queue checkpoint metadata with resume policy and required review statuses.
  - Orchestration status APIs expose HITL checkpointing as part of the cost/trust policy.
- `test: add quality permission regression suite`
  - Added focused guardrails for evidence-first approval, restricted RAG hiding, HITL checkpoint metadata, and cache dedupe.
- `feat: add cross-agent evidence summaries`
  - Mail/Docs and Track C memory extraction AgentRuns now persist source evidence summary metadata for richer Review Drawer inspection.
- `feat: align search retrieval backend`
  - `/api/v1/search` now reports its retrieval backend and can use the same pgvector feature-flag path as `/api/v1/ask`.
- `feat: add slack thread context chunks`
  - Slack reply chunks now include parent message context and thread metadata for better Review/RAG evidence quality.
- `feat: enrich gmail thread domain metadata`
  - Gmail events now preserve thread context keys, participants, participant domains, and external-domain flags.
- `feat: add drive parser version metadata`
  - Drive events now preserve parser status, document version, revision id, and content signatures for safer parsing/indexing.
- `feat: add calendar event quality metadata`
  - Calendar events now preserve event context, status, organizer, RSVP counts, duration, and external attendee domains.
- `test: add connector golden dataset`
  - Added static Slack/Gmail/Drive/Calendar golden payloads and metadata regression assertions.
- `test: add rag retrieval smoke metrics`
  - Added local precision/recall/hit-rate evaluation for deterministic RAG retrieval fixtures.
- `feat: add structured memory extraction adapter`
  - Added a LangChain structured-output adapter behind the existing Track C memory extraction contract.
- `feat: add product memory pages`
  - Added Decisions, Timeline, and History pages backed by approved Knowledge API records.
- `docs: add production auth plan`
  - Documented httpOnly cookie sessions, refresh token rotation, RBAC, audit, and demo-mode migration.
- `docs: add deployment runbook`
  - Documented production runtime components, verification, cost gates, monitoring, and rollback.
- `feat: add review agent notifications`
  - Added derived Review Queue and AgentRun notifications with a product page and route regression coverage.
- `feat: add knowledge map`
  - Added a zero-paid-call approved-memory graph endpoint and `/knowledge-map` product page.
- `feat: add cookie auth session slice`
  - Added httpOnly session/refresh cookies, refresh rotation, logout revoke, and frontend credentialed fetches.
- `docs: add portfolio demo script`
  - Added a recording-ready product walkthrough with cost and security talking points.
- `feat: add azure openai compatible alias`
  - Added `azure_openai` provider-order support backed by the existing OpenAI API key path.
- `feat: add google identity rbac`
  - Separated ParaWorks Google identity login from Gmail/Drive/Calendar data integration OAuth so sign-in uses only identity scopes while data connectors keep explicit read-only consent.
  - Added seeded portfolio accounts: `hanvv3@gmail.com` as admin, `hanvv3@koreacu.ac.kr` as employee, plus reviewer/employee demo users for role comparison.
  - Added RBAC helpers, admin user management APIs/UI, role-aware navigation filtering, Google account picker login URL, and Review Queue approval checks.
  - Cost/security angle: login and role checks are zero-paid-call paths, unknown Google accounts are rejected, refresh tokens remain httpOnly/hashed, and role changes create audit logs.
- `feat: expose google identity readiness`
  - The login URL API now reports `redirect_uri` and exact missing Google identity configuration keys instead of treating a partial client id as ready.
  - The login page surfaces the required redirect URI and missing settings so Google Cloud setup can be completed without guessing.
  - Verification: backend suite `245 passed, 1 skipped`; frontend lint/build passed; targeted Playwright login/admin regression `14 passed`.
- `feat: restrict cost observability to admins`
  - Added backend admin guards to AgentRun cost/token APIs and RAG reindex/indexing observability APIs.
  - Direct `/agent-runs` and AgentRun detail URLs now render an admin-required state instead of crashing when a non-admin context reaches the page.
  - Verification: targeted admin-only API tests passed, backend suite `252 passed, 1 skipped`, and targeted Playwright AgentRun regression `10 passed`.
- `feat: redesign review-centered frontend shell`
  - Reworked the global frontend shell around a Korean-first company-memory console with a persistent Ask/search bar, review count, role-aware navigation, and operational status signals.
  - Redesigned `/dashboard` as a Review Queue workbench instead of a generic widget dashboard, foregrounding evidence-backed candidates, trusted knowledge, AgentRun cost, permissions, and sync health.
  - Visual direction shifted from purple glass styling to a quieter business-console palette with compact panels, status badges, and evidence-first calls to action.
  - Verification: `npm.cmd exec tsc -- --noEmit` passed; `npm.cmd run build` passed after rerunning outside the sandbox because the sandboxed build hit a Windows `.next` rename `EPERM`.
- `feat: redesign ui as macos sequoia productivity app`
  - Rebuilt the global shell around translucent macOS-style sidebars, soft glass toolbars, native icon controls, dark/light material tokens, and Spotlight-style command search.
  - Reframed `/dashboard` as a desktop productivity workspace for AI-detected decisions: center timeline, approval workflow, project history, and right-side source evidence inspector.
  - Added reusable glass, native button, badge, inspector, sheet, sidebar, and motion utility classes in the global UI system for consistent follow-on pages.
  - Verification: `npm.cmd run build` passed; local Next dev server returned HTTP 200 for `http://127.0.0.1:3000/dashboard`.
- `feat: align dashboard with prody-style project workspace`
  - Revised the previous glass-heavy direction into a cleaner desktop app frame inspired by the supplied reference: light gray sidebar, rounded white workspace, breadcrumb top bar, compact controls, project header, chart card, and spreadsheet-like decision table.
  - Dashboard data now reads as an enterprise archive project with confidence trend, source score indicators, review actions, and rows for decision/evidence/source records.
  - Verification: `npm.cmd run build` passed; `http://127.0.0.1:3000/dashboard` returned HTTP 200.
- `feat: implement asset-referenced enterprise console frontend`
  - Used the `data/assets/pages (*.png)` references to rebuild the global shell and dashboard around a white Korean business console: fixed sidebar, top search/actions, compact source cards, live activity stream, review-priority table, and right-side operations summary.
  - Replaced mojibake Korean copy in the touched shell/dashboard surfaces and added compatibility CSS tokens for existing Liquid Glass pages so the broader frontend keeps rendering while the new console direction lands.
  - Verification: `npm.cmd exec tsc -- --noEmit` passed; `npm.cmd run build` passed; local Next dev server returned HTTP 200 for `http://127.0.0.1:3000/dashboard`. Playwright screenshot capture could not run because the local Playwright browser binary is not installed.
- `style: tighten frontend to supplied reference ratios`
  - Rechecked the asset references and moved the shell/dashboard closer to the supplied console proportions: 216px left rail, narrower top search, 320px right rail, compact 8px panels, smaller 13px operational text, softer blue active states, and brighter off-white canvas.
  - Rebuilt `/dashboard` around the reference dashboard composition: today's workflow cards, critical-action cards, review table rows, right-side live activity, agent summary, quick Ask, and bottom AI insight cards.
  - Verification: `npm.cmd exec tsc -- --noEmit` passed; `npm.cmd run build` passed; local `/dashboard` returned HTTP 200.
- `style: align dashboard with 0508 page references`
  - Used `data/assets/0508-pages/0508 (2).png`, `0508 (3).png`, and `0508 (4).png` to tighten the ParaWorks shell and dashboard into a Korean enterprise operations console.
  - Restored readable Korean labels in the touched shell/dashboard surfaces, matched the fixed 216px left rail, compact top search, source metric cards, real-time activity stream, review table, and 320px right operations panels.
  - Verification: `npm.cmd exec tsc -- --noEmit` passed; `npm.cmd run build` passed; local `/dashboard` returned HTTP 200 on `http://127.0.0.1:3000/dashboard`.
- `style: connect developed product pages to current console`
  - Reworked Messages, Ask/Search, Knowledge Library, Decisions, Timeline, History, Knowledge Map, and Notifications around the existing backend API contracts instead of placeholder content.
  - Added clear API-unavailable states for server-backed knowledge pages so users can distinguish developed features from a stopped backend.
  - Verification: `npm.cmd exec tsc -- --noEmit` passed; `npm.cmd run build` passed; local HTTP smoke returned 200 for dashboard, messages, search, knowledge, decisions, timeline, history, knowledge map, notifications, review, agent-runs, integrations, admin, and login.
- `feat: polish account-aware frontend shell`
  - Connected the sidebar account card to the current `/api/v1/auth/me` user instead of a hard-coded placeholder, including role labels and profile image fallback from `frontend/public/profile`.
  - Added a dedicated `/account` page so the account dropdown's "내 계정" action shows the current user's email, role, title, department, status, account id, permission levels, and profile image without logging out or returning to the login screen.
  - Hid admin-only navigation entries from non-admin users, including Admin Console and AgentRun execution records, while keeping the existing backend/API authorization checks for direct URL access.
  - Fixed escaped Korean text rendering in the account header and global search placeholder so users see readable Korean copy instead of raw `\u...` sequences.
  - Verification: `npm.cmd exec tsc -- --noEmit` passed; targeted auth/admin/AgentRun backend tests passed; `npm.cmd run build` passed with `/account` included in the route manifest.
- `feat: add mail document calendar project grouping`
  - Added `docs/mail-doc-calendar-agent-status.md` to summarize the current Developer B agent state, gaps, and next work for Google Drive, Gmail, and Calendar evidence.
  - Extended the Mail/Document Agent evidence packet to include Calendar chunks and preserve event context/status/organizer/duration metadata.
  - Added `GET /api/v1/projects`, which groups Gmail, Gmail attachment, Drive, and Calendar evidence by `project_key`/`scenario` with permission-aware hidden project accounting.
  - Updated backend test fixtures so CSRF cookies/headers and auth rate-limit state match the current production-like security middleware during tests.
  - Verification: `uv run pytest backend/tests -v` passed with 287 tests and 1 skipped pgvector integration test.
- `feat: add assistant optimistic turns and email approval drafts`
  - AI 비서 now shows the user's message immediately while the assistant turn is still running, then reveals the assistant answer with a smooth typing-style stream effect.
  - Added a modular assistant email action path that detects direct email-send requests without forcing RAG evidence, drafts a business-tone subject/body, and stores the draft as pending approval in assistant message metadata.
  - Added a Gmail approval endpoint that sends only after explicit user approval and only when an installed Gmail connection already has send-capable OAuth scope.
  - Verification: `uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py -q`, `npm.cmd run lint`, `npm.cmd run test:visual -- assistant-memory.spec.ts --project=chromium-desktop`, and `npm.cmd run build` passed.
- `feat: connect work data and harden assignment extraction`
  - Fixed dashboard timeline data to use real `TimelineEvent` fields and connected `/projects` to the existing permission-aware project memory API instead of empty frontend state.
  - Repaired future todo-to-timeline promotion copy so generated Korean timeline entries no longer contain mojibake.
  - Preserved source snippets in Mail/Docs and Memory Extraction evidence packets, added richer Review evidence metadata, and expanded deterministic extraction for Korean/English work assignments from Gmail, Drive, and Calendar evidence.
  - Added Mail/Docs and Memory Extraction LLM preflight responses that expose evidence counts and estimated cost while keeping live LLM execution closed for this slice.
  - Verification: `uv run pytest backend/tests/test_dashboard_api.py backend/tests/test_knowledge_api.py backend/tests/test_review.py backend/tests/test_mail_document_agent.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_memory_extraction_agent.py backend/tests/test_memory_extraction_review_bridge.py backend/tests/test_agent_preflight.py -q` passed with 29 tests; `uv run ruff check ...` fixed and cleared touched backend files; `npm run lint` and `npm run build` passed.
- `fix: scope sync-driven agent reviews`
  - Collapsed connector sync and deterministic Agent review generation into one user action while keeping cost control at the ingestion boundary: unchanged source content now returns no changed source ids, so duplicate syncs do not rerun review extraction.
  - Scoped Slack and Mail/Document evidence packets by changed source ids so Gmail, Drive, and Slack review candidates are generated only from the connector that just changed.
  - Added a confidence-gated low-cost AI 비서 routing layer for email drafts and lightweight replies, preserving RAG for ambiguous/company-memory questions.
  - Verification: targeted backend sync/assistant/review tests passed (`46`, `29`, and `17` tests); ruff passed; `npm.cmd run lint` and `npm.cmd run build` passed.
- `fix: improve developer b google review ingestion`
  - Updated the Mail/Document Agent sync path so changed Google Drive files create separate Review Queue candidates instead of one over-aggregated item, while Gmail keeps parent email and attachment evidence grouped together.
  - Tightened live Gmail collection with a business-focused query window and spam/trash/social/promotions/forums exclusions, plus explicit Gmail message `content_signature` metadata for safer dedupe.
  - Verification: 63 targeted Google/Mail-Document/connector runtime tests passed; ruff passed on touched backend files.

- `feat: harden mail document operating mvp`
  - Added permission-filtered Mail/Document evidence packets, grouped manual/orchestrated ReviewItem generation, and source-id preservation through review rejection so rejecting AI candidates no longer deletes connector evidence.
  - Added Slack-style Mail/Docs live LLM preflight and explicit paid-run endpoint using existing `AGENT_LLM_*` settings; sync remains deterministic/cost-safe and live LLM is only user-triggered.
  - Stored operational details in `AgentRun.metadata_`, `AuditLog.metadata_`, and API responses without adding log-path environment variables; legacy Slack sync now uses a module logger instead of `print()`.
  - Hardened RAG indexing against malformed approved `payload.source_ids` and included approved `TimelineEvent` rows as knowledge documents.
  - Verification: `63` targeted backend tests passed; ruff passed on touched backend paths; frontend TypeScript check and Next production build passed.

- `fix: connect slack sync to agent_slack llm pipeline`
  - Slack sync 후 `Redis 큐 관련 결정사항 추출됨` 1건만 생성되던 원인이 sync 경로의 결정론/fake Slack Agent 호출임을 확인했다.
  - 운영형 local/prod 모드와 provider key가 있는 경우 `/api/v1/integrations/slack/sync`가 `agent_slack.process_daily_slack_sync()` 기반 분석 경로를 타도록 연결했다.
  - `trigger_slack_agent_analysis()`가 변경된 Slack `Source.source_id`만 받아 분석하도록 좁혀, 최근 7일 전체 재분석으로 인한 중복 비용과 중복 ReviewItem 생성을 피했다.
  - demo/test 모드와 provider key가 없는 환경은 기존 결정론 스모크 경로를 유지해 테스트가 live LLM을 호출하지 않도록 했다.
  - Verification: targeted Slack sync/Agent API tests passed (`1`, `8`, and `23` tests); ruff passed.

- `fix: upgrade assistant model and log tool calls`
  - AI Assistant RAG answering now has a stronger primary OpenAI model setting:
    `AGENT_LLM_OPENAI_PRIMARY_MODEL=gpt-5.4`, with
    `AGENT_LLM_OPENAI_MODEL=gpt-5.4-mini` kept as the fallback.
  - Added assistant tool-call trace logging through the Python
    `AssistantTool` logger, using English lines such as
    `[Tool: rag_retrieval] result backend=keyword source_count=...`; the
    docker scripts surface these lines through the backend stderr log file.
  - The trace covers email action routing, RAG retrieval backend selection, and
    RAG answer model start/result/error events, making it easier to see whether
    the assistant used a tool or company-memory retrieval.
  - Verification: targeted model/logging tests passed, wider assistant/RAG
    backend tests passed with 40 tests, and ruff passed on touched files.
- `fix: split assistant email routing agents`
  - Split the AI Assistant email path into an `email_intent_gate` that only
    detects email intent and an `email_draft_composer` that only writes
    approval-only drafts or clarification questions.
  - Removed the active combined prompt that tried to classify email actions,
    general replies, and company-memory RAG in one low-cost call; non-email
    messages now naturally continue to the RAG answer path.
  - Added orchestration for "RAG result to email" requests: when the intent gate
    marks `requires_rag_result`, the assistant retrieves the company-memory
    answer first and passes that answer/source context to the draft composer.
  - Verification: targeted email-agent/API tests passed, wider assistant/RAG
    backend tests passed with 43 tests, and ruff passed on touched files.
- `fix: preserve assistant email continuation context`
  - Fixed recipient-only email follow-ups by preserving complete recent
    conversation JSON rows and passing recent assistant answers as explicit
    draft source context to the email draft composer.
  - This supports flows like "recent decisions only" -> "send this to
    kjw4work@gmail.com" without asking for the same email content again.
  - Adjusted `scripts/paraworks-docker.ps1` to wait for backend health before
    starting the frontend, reducing startup `ECONNREFUSED` proxy noise.
  - Verification: assistant/RAG/script tests passed with 47 tests; ruff and
    PowerShell parser checks passed.
- `fix: harden docker startup migrations`
  - Added checked native-command execution and a Postgres readiness wait to
    `scripts/paraworks-docker.ps1`, so Docker, Alembic, and schema failures no
    longer scroll by before a misleading final ready message.
  - Made the `project_key` Alembic migration idempotent for fresh databases
    where the current-schema baseline already created those columns and
    indexes.
  - Kept pgvector `vector(1536)` dimension validation while suppressing the
    expected SQLAlchemy reflection warning from the CLI schema check.
  - Verification: Docker script tests, DB schema operation tests, pgvector
    runbook tests, ruff, PowerShell parser check, real docker startup, backend
    `/health`, and frontend `/login` smoke all passed.
- `feat: resolve assistant email recipients`
  - Added a deterministic AI Assistant recipient resolver between
    `email_intent_gate` and `email_draft_composer` so natural-language
    recipients can be mapped to known email addresses before draft generation.
  - The resolver collects candidates from recent conversation contact pairs,
    active `AuthUser` rows, `demo_auth.USERS`, and Google source metadata from
    Gmail, Drive, and Calendar.
  - The email draft prompt now receives `resolved_recipients`, while all actual
    sends still require the existing pending approval and Gmail send endpoint.
  - Verification: targeted recipient resolver/API tests passed, wider assistant
    backend tests passed with 37 tests, and ruff passed on touched files.
- `fix: upgrade email draft composer model`
  - Split the email sub-agent model setting so the cheap intent gate remains on
    `gpt-4.1-nano`, while the email draft composer defaults to
    `gpt-5.4-mini`.
  - Added `ASSISTANT_EMAIL_DRAFT_AGENT_MODEL` to `.env.example` and wired
    `build_email_draft_composer()` to use the dedicated stronger model.
  - Verification: targeted model-routing tests passed, wider assistant backend
    tests passed with 39 tests, and ruff passed on touched files.
- `fix: route assistant contact lookups`
  - Added a dedicated contact lookup route before `email_intent_gate`, so
    address lookup requests such as `김종우님 이메일 알려줘.` do not become email
    draft clarification loops.
  - Added Korean aliases for demo contacts and kept active `AuthUser` records
    higher priority than demo fallback contacts.
  - Verification: targeted contact lookup tests passed, wider assistant backend
    tests passed with 47 tests, and ruff passed on touched files.
- `fix: preserve referenced assistant content in email drafts`
  - Added an explicit referenced-content email draft path so `이 내용으로
보내줘` and draft correction complaints use the latest sendable assistant
    answer or pending draft state before falling through to normal RAG chat.
  - Added a guardrail that appends the selected source content when the draft
    composer produces a generic body that omits the actual referenced answer.
  - Verification: targeted referenced-email tests passed, wider assistant
    backend tests passed with 49 tests, and ruff passed on touched files.
- `fix: generate assistant content before email drafting`
  - Added a generate-then-email path for requests like `ParaWorks 회사 소개서
작성해서 용희님한테 메일 보내줘`, extracting the requested artifact question,
    running RAG first, then using that generated answer as the draft body source.
  - This keeps combined artifact creation plus email requests out of the
    generic clarification loop that asks the user to provide the content.
  - Verification: targeted generate-then-email test passed, wider assistant
    backend tests passed with 50 tests, and ruff passed on touched files.
- `fix: harden assistant recipient correction`
  - Tightened recipient resolution so contacts only surface when the latest
    message actually matches a name, alias, email, or title, preventing unknown
    names from reusing previous draft recipients.
  - Added pending-draft recipient correction handling for wrong-address feedback
    and recipient-only follow-ups such as `SeungHun Han님한테 보내줘`.
  - Verification: targeted recipient-correction tests passed, wider assistant
    backend tests passed with 53 tests, and ruff passed on touched files.
- `docs: define gmail drive project routing collaboration`
  - Added a Korean collaboration guide for moving Gmail and Google Drive review
    candidates to the same LangChain tool-based project routing contract as the
    Slack Agent.
  - The guide separates ownership: shared router contracts in
    `backend/app/agent_runtime/`, Mail/Document work in
    `backend/app/agents/mail_document_agent/`, Slack work in `agent_slack/`,
    and Review/Timeline/Projects UI under an integration branch.
  - It documents the shared ReviewItem payload fields, Gmail body plus
    attachment grouping, Drive file-level grouping, AgentRun metadata, backend
    tests, and Playwright responsibilities.
- `fix: 프로젝트/타임라인 원본 링크 노출`
  - 타임라인 상세 패널의 `Open source` 링크가 새 탭으로 열리도록
    `target="_blank"`와 `rel="noopener noreferrer"`를 추가했다.
  - 프로젝트 탭의 `연결된 원본 근거`와 `승인된 프로젝트 활동` 카드에서 원본
    근거 링크를 바로 열 수 있게 했다.
  - 검증: frontend lint/build 통과, Playwright 원본 링크 검증 2개 통과.
- `fix: 타임라인 source time과 프로젝트 근거 UX 개선`
  - 타임라인을 승인 시각이 아닌 실제 Slack 대화 시각 기준 `occurred_at`으로
    정렬하고, 날짜 단위 compact/detail 토글을 추가했다.
  - 타임라인 리스트는 기본적으로 title만 보이게 하여 스캔 속도를 높였다.
  - 프로젝트 탭이 승인된 활동의 source evidence를 `연결된 원본 근거`로
    표시하도록 바꿨다.
  - 검증: backend 프로젝트/승인 테스트 44개 통과, ruff 통과, frontend
    lint/build 통과, Playwright 타임라인/프로젝트/Slack 흐름 3개 통과.
- `fix: 타임라인 날짜 accordion UX 조정`
  - 타임라인 날짜 헤더를 직접 클릭하는 accordion으로 바꿔 모든 날짜를 항상
    보이게 하고, 선택한 날짜의 타임라인만 펼쳐지게 했다.
  - 펼쳐진 타임라인 카드에는 title과 source 시간, source type, 승인 상태,
    summary를 함께 표시한다.
  - 검증: frontend lint/build 통과, Playwright 타임라인/Slack 흐름 2개 통과.
- `fix: 타임라인 목록 summary 노출 제거`
  - 타임라인 목록 카드에서 `result_summary` 노출을 제거하고, 상세 내용은
    Slack history 버튼을 눌렀을 때 오른쪽 상세 패널에서만 보이게 했다.
  - 검증: frontend lint/build 통과, Playwright 타임라인/Slack 흐름 2개 통과.
- `fix: 대시보드 todo 완료 상태 영구 저장`
  - 대시보드의 완료 버튼이 로컬 숨김에 그치지 않도록 `Todo.completed_at`,
    `completed_by`를 추가하고 `POST /api/v1/todos/{todo_id}/complete`로
    완료 상태를 DB에 저장하게 했다.
  - 대시보드 `today_todos`는 approved `ReviewItem` 대신 승인된 미완료
    `Todo`를 기준으로 표시하며, Review 승인 시 담당자와 마감일도 `Todo`에
    저장한다.
  - 프로젝트/타임라인 응답에 완료 정보를 포함해 완료된 할 일이 프로젝트 활동과
    타임라인에서 `완료`로 보이게 했다.
  - 완료 API는 사용자가 접근할 수 없는 permission level의 todo를 403으로
    거부한다.
  - 검증: backend 관련 테스트 36개 통과, ruff 통과, frontend lint/build
    통과, Playwright 대시보드/타임라인/프로젝트 5개 통과, Docker Postgres
    migration 적용 확인.
- `fix: 타임라인 날짜 탐색과 검토 카드 동기화 개선`
  - 타임라인 상태 표시의 `approved`를 `승인됨`으로 한글화하고, 최근 7일은
    기본 펼침, 이전 날짜는 접힘 상태로 시작하게 했다.
  - 날짜가 많아질 때 스캔할 수 있도록 월별 sticky header, 좌측 날짜 인덱스,
    `활동 있는 날짜만 보기 / 전체 날짜 보기` 토글을 추가했다.
  - 검토사항 우클릭 메뉴를 항목 제목이 보이는 빠른 승인/반려 드롭다운으로
    다듬고 화면 가장자리에서 잘리지 않게 위치를 보정했다.
  - 대시보드 검토사항 카드는 Review Queue 정렬과 같은 pending item 3개만
    표시하되 배지 숫자는 실제 pending review 총수를 사용하도록 맞췄다.
  - 검증: `backend/tests/test_dashboard_api.py` 6개 통과, Python ruff 통과,
    frontend lint/build 통과, Playwright 타임라인/대시보드/검토 bulk 테스트
    8개 통과.
- `fix: 대시보드 검토사항 카드 deep link와 표시 제목 정합성 개선`
  - 대시보드 검토사항 카드의 각 항목 링크가 `/review?itemId=...`로 이동해
    검토사항 페이지에서 해당 항목이 포함된 그룹을 자동으로 펼치고 스크롤한다.
  - `ParaWorks source 연결`처럼 낮은 정보량의 payload title은 summary, reason,
    task/source title 같은 실제 검토 큐 표시 텍스트로 대체하는 공용 display title
    규칙을 추가했다.
  - 대시보드 API와 Review API가 같은 display title 규칙을 사용해 목록 불일치를
    줄였다.
  - 검증: dashboard/review API 테스트 8개 통과, Python ruff 통과, frontend
    lint/build 통과, Playwright 대시보드/검토 bulk 테스트 4개 통과.
- `fix: 대시보드 검토사항 카드 중복 그룹 접기`
  - 대시보드 검토사항 카드의 목록을 Review Queue와 같은 display title + item type
    그룹 기준으로 dedupe해 같은 후보가 여러 개 있어도 카드에는 하나만 보이게 했다.
  - `pending_review_count` 배지는 실제 검토 대기 총수를 그대로 유지한다.
  - 검증: `backend/tests/test_dashboard_api.py` 8개 통과, Python ruff 통과,
    Playwright 대시보드 테스트 2개 통과.
- `style: 프로젝트 워크스페이스 UI 리디자인`
  - 프로젝트 페이지를 대시보드와 같은 SaaS workspace 톤으로 재구성했다.
  - 선택 프로젝트 overview hero, metric mini cards, 강조된 프로젝트 목록,
    source filter tab이 있는 원본 근거 패널, timeline형 승인 활동 패널을 추가했다.
  - 2XL에서는 3영역, 1440/1366급에서는 2영역+활동 하단, 태블릿 이하에서는
    세로 stack으로 전환되게 했다.
  - 기존 프로젝트 검색, 생성, 새로고침, 원본 근거 링크, source/type badge,
    empty state 흐름은 유지했다.
  - 검증: frontend lint/build 통과, Playwright 프로젝트 페이지 테스트 4개 통과.
- `style: 프로젝트 워크스페이스 board UI 정교화`
  - ParaWorks 사이드바와 전역 셸 동작은 그대로 두고 프로젝트 페이지 본문만 target
    이미지의 calm kanban/workspace 톤으로 다듬었다.
  - 페이지 헤더, 선택 프로젝트 summary, 프로젝트 목록 lane, 원본 근거 lane,
    승인 활동 lane을 white/off-white glass surface, rounded board card, soft shadow,
    pill chip 체계로 정리했다.
  - Drive/Gmail/Slack/Calendar 원본 근거와 활동 유형별 카드에 아주 옅은 pastel
    tint를 적용해 정보 구조는 유지하면서 workspace board 느낌을 강화했다.
  - 기존 프로젝트 선택, 검색, 생성, 새로고침, source filter, 원본 근거 링크,
    승인 활동 렌더링, responsive 3/2/1 column 흐름은 유지했다.
  - 검증: frontend lint/build 통과, Playwright 프로젝트 페이지 테스트 4개 통과.
- `style: 프로젝트 목록 sticky follow 적용`
  - 프로젝트 페이지의 프로젝트 목록 lane을 `fixed`가 아닌 normal flow 기반
    `sticky` 패널로 바꿔 스크롤 시 부드럽게 따라오도록 했다.
  - 1280px 이상에서만 sticky를 적용하고, 목록 내부는 viewport 높이에 맞춰
    스크롤되게 해 노트북 화면에서 패널이 잘리지 않도록 했다.
  - 검증: sticky 회귀 Playwright 테스트 추가, frontend lint/build 통과.
- `fix: 대시보드 업무/프로젝트 카드 링크 교체`
  - 대시보드 `오늘 해야 할 업무` 카드의 우측 링크를 `타임라인 보기`로 바꾸고
    `/timeline`으로 이동하게 했다.
  - `담당 프로젝트` 카드의 우측 링크는 `프로젝트 보기`로 바꾸고 `/projects`로
    이동하게 했다.
  - 검증: 대시보드 Playwright 회귀 테스트에 두 링크의 라벨과 href를 고정하고
    frontend lint/build 통과.
- `fix: 검토사항 Agent 배지 source별 분리와 sticky action bar`
  - Review item의 Agent 배지를 `agent_name`만 보지 않고 `payload.source_type`과
    `source_evidence.source_type`을 함께 사용해 Slack, Mail, Google Drive,
    Calendar Agent로 구분했다.
  - 배지 색상은 프로젝트 페이지 source badge와 맞춰 Slack violet, Mail rose,
    Drive blue, Calendar emerald 계열로 통일했다.
  - 검토사항 상단 bulk action bar를 `fixed`가 아닌 sticky로 바꿔 스크롤 시
    문서 흐름 안에서 따라오게 했다.
  - 검증: Review Playwright 회귀 테스트에 source별 Agent label/color와 sticky
    action bar를 고정하고 frontend lint/build 통과.
- `style: 타임라인 Explorer UI 압축 리디자인`
  - 타임라인 페이지 본문을 대시보드/프로젝트 페이지와 같은 soft SaaS workspace
    톤으로 정리했다.
  - summary strip, pill형 프로젝트 탭, compact filter toolbar, 월별 sticky header,
    좌측 compact month navigator를 추가/정리했다.
  - 기본 진입 시 최근 월과 최근 날짜만 펼치고 오래된 월/날짜는 접힘 상태로 두어
    긴 로그 리스트 스크롤 압박을 줄였다.
  - 날짜 그룹은 기본 3개 항목만 보여주고 `N건 더 보기`로 점진 확장하며, history
    row는 source/status badge와 1줄 title/preview 중심 compact card로 정리했다.
  - 기존 프로젝트 탭, 기간/소스/상태 필터, 전체 날짜 보기, 필터 초기화, 날짜
    jump, source link detail panel, 더 보기 동작은 유지했다.
  - 검증: 타임라인 Playwright 테스트 4개 통과, frontend lint/build 통과.
- `fix: 타임라인 날짜 인덱스 sticky 동작 보정`
  - 타임라인 날짜 인덱스가 `position: sticky`여도 상위 `overflow-hidden` 때문에
    페이지 스크롤을 따라오지 못하던 문제를 보정했다.
  - timeline list panel을 `overflow-visible`로 바꿔 인덱스가 fixed overlay 없이
    normal flow 안에서 전역 top bar 아래로 자연스럽게 붙게 했다.
  - 검증: 스크롤 후 날짜 인덱스가 sticky top 근처에 유지되는 Playwright 회귀
    테스트 추가, frontend lint/build 통과.
- `style: 유틸리티 워크스페이스 페이지 SaaS 톤 정리`
  - AI 비서, 에이전트 실행 기록, 연동 관리, 알림, 관리자 콘솔 페이지를
    `utility-workspace` 스코프로 묶고 대시보드/프로젝트/타임라인과 같은 soft SaaS
    workspace surface 체계로 정리했다.
  - 사이드바와 각 페이지의 API/data/동작은 유지하고, page header, summary badge,
    panel/card, integration card, admin table, AI chat shell, action button 스타일만
    white/off-white glass surface, rounded card, subtle shadow, pill control 톤으로
    업그레이드했다.
  - `frontend/e2e/utility-workspace-style.spec.ts`를 추가해 다섯 페이지가 공통
    스타일 스코프를 유지하는지 회귀 검증한다.
  - 검증: frontend lint/build 통과, Playwright utility workspace + integrations
    sync modal 테스트 12개 통과.
- `fix: AI 비서 채팅 히스토리 접기 컨트롤 복구`
  - AI 비서 페이지에서 채팅 히스토리를 펼친 뒤 다시 접는 컨트롤을 명확히
    복구했다.
  - 기존 히스토리 패널 안의 작은 닫기 버튼은 명시적으로 `setSidebarCollapsed(true)`를
    호출하도록 고정하고, 채팅 본문 좌상단에도 `히스토리 접기` 버튼을 추가했다.
  - AI 비서 루트에 hydration 신호를 추가해 Playwright 상호작용 테스트가 실제
    클라이언트 핸들러 연결 이후 실행되도록 했다.
  - 검증: assistant memory + utility workspace Playwright 테스트 desktop/mobile
    통과, frontend lint/build 통과.
- `fix: 대시보드 캘린더 refresh 시 오늘 날짜 유지`
  - 대시보드 캘린더가 선택 날짜에 일정이 없을 때 연동 일정 중 가장 빠른 날짜를
    자동 선택하던 로직을 제거했다.
  - 2026-05-17 새로고침 시 오늘 날짜를 유지하고, 2026-04-17 같은 과거 연동 일정은
    해당 월로 이동했을 때 dot/목록으로만 확인되도록 했다.
  - 회귀 테스트로 이전 월에만 연동 일정이 있어도 2026년 5월과 5월 17일 선택 상태를
    유지하는 케이스를 추가했다.
- `fix: 검토사항 프로젝트 연결 캘린더 raw metadata 표시 정리`
  - Calendar source로 생성된 프로젝트 연결 후보가 `Description`, HTML 태그,
    `Location`, `Start`, `End` 메타데이터를 제목처럼 노출하던 문제를 수정했다.
  - 프로젝트 분류기 생성 단계에서 캘린더 후보 summary를 이벤트 제목 중심으로 정제하고,
    Review API와 프론트 표시 단계도 기존 raw payload를 방어적으로 정리한다.
  - 검증: 관련 backend Review/Project 테스트 45개 통과, Review Playwright 테스트 4개 통과,
    frontend lint/build 통과, 수정 파일 ruff 통과.
- `fix: 프로젝트 목록 설명에서 자동 연결 통계 문구 제거`
  - 프로젝트 페이지 좌측 프로젝트 목록 카드의 설명란에서
    `승인된 원본 근거 00건과 승인된 프로젝트 활동 00건이 연결되어 있습니다.` 자동 문구를
    제거하고, 사용자가 입력한 프로젝트 설명만 보이도록 했다.
  - 프로젝트 상세/metric 정보는 유지하고 목록 하단의 `근거 · 활동 · 검토 대기` 수치도 그대로 둔다.
  - 검증: Projects Playwright 테스트 4개 통과, frontend lint/build 통과.
- `fix: 검토사항 프로젝트 연결 메일/Drive metadata 표시 정리`
  - Gmail 규칙 기반 프로젝트 연결 후보에서 `From`, `Date` 헤더와 깨진 발신자명이
    제목/연결 내용에 붙어 보이던 문제를 캘린더 raw metadata 정리와 같은 경로로 해결했다.
  - Drive와 Gmail 첨부의 `Mime type`, `Owner`, `Parent subject`, `Attachment size` 같은
    metadata label도 프로젝트 연결 후보 표시와 생성 summary에서 제거한다.
  - Calendar/메일/Drive/첨부 회귀 테스트를 추가했고, Slack은 metadata header를 붙이지 않는
    connector 구조라 기존 메시지 본문/스레드 정리 흐름을 유지했다.
  - 검증: 관련 backend Review/Project 테스트 47개 통과, Review Playwright 테스트 4개 통과,
    frontend lint/build 통과, 수정 파일 ruff 통과.
- `fix: 타임라인 완료 상태 병합 및 상태 한글화`
  - 타임라인 상태 필터와 row chip을 `승인됨`, `완료`로 한글화했다.
  - 완료된 todo를 타임라인에 새 항목으로 추가하지 않고, 같은 프로젝트/source link의
    기존 `[할 일] ...` 타임라인 이벤트에 `completed_at`, `completed_by`를
    병합해 화면에서 `완료`로 보이게 했다.
  - 검증: backend 프로젝트/대시보드/todo 테스트 32개 통과, ruff 통과,
    frontend lint/build 통과, Playwright 대시보드/타임라인 3개 통과.

## 2026-08-28 Auto-review trust promotion persistence hardening

- Added the additive C.5 persistence boundary at Alembic revision
  `7c5a2e9f4b10`, preserving nullable V2.0 rows while storing exact V2.1
  extraction/provider/model/reasoning/route snapshots.
- PostgreSQL enforces same-owner identities, deferred validation completeness,
  exact current Assistant evidence lineage, append-only provider/rollout/audit
  ledgers, immutable trusted provenance, and source/parser/chunk authority.
  Provider counters/cost/time/reason/gate/timestamps require the exact event
  backpointer, and event kinds are semantic: an overrun cannot replace the
  authorized estimator/framing/price authority, while only an operator clear
  may replace it after an open breaker. Operator and call attribution are
  explicitly non-null and mutually exclusive by event kind. Rollout
  authorization, breaker open,
  breaker close, and generation invalidation each have disjoint prior/new
  control transitions, so a correctly backpointed event cannot use the wrong
  kind. Metric-only updates still increment `state_version` while preserving
  `control_epoch` and event sequence; cumulative correction counts cannot
  decrease.
- Bootstrap, connector reset, and empty-only downgrade share exhaustive,
  fail-closed C.5 retained-state detection. Database bootstrap errors abort
  startup before C.5 service construction, and populated schemas refuse
  downgrade. Legacy parser/chunk rows remain eligible only as retained audit
  data: null authority/lineage cannot be attached later by UPDATE.
- Slack remains outside C.5 automatic eligibility. Verification uses local
  SQLite and isolated PostgreSQL/pgvector only, with no live provider calls.
- Round-4 verification used a pinned manual `2f6a8b9c0d1e` schema fixture,
  exact Task 2 index assertions, and finally-dropped PostgreSQL schemas:
  `176 passed` in the focused suite, including `87` PostgreSQL tests with zero
  skips; the Task 1 compatibility suite remains `118 passed`.

## 2026-08-28 C.5 Task 3 immutable V2.1 extraction hardening

- Production preflight now dispatches V2.0 and V2.1 explicitly. V2.1 stores
  its own graph/checkpoint identity and the complete immutable extraction
  registry, provider/model/reasoning/route, estimator/price, cap, timing,
  rollout, safety, and aggregate plan snapshots.
- Paid extraction uses a PostgreSQL-authoritative E1/E2/E3 ledger. Owner,
  permission, runtime key, source/version, cancellation, lease, signed budget,
  and provider-safety state are rechecked under the prescribed locks. Known
  usage is charged exactly; unknown post-marker usage charges the reserve;
  overruns open the extraction breaker atomically.
- The provider grant is a store-owned post-commit capability with no module
  issuer/factory or caller-visible transport. The real store authenticates the
  exact live grant and rechecks the locked attempt in a short transaction,
  commits and releases all database/grant locks, then consumes the permit
  before its one body-blind Task 1 HTTP-hook dispatch. Grants and permits are
  non-copyable, non-pickleable, redacted and process-local; retry, fallback,
  cache, callbacks and tracing stay disabled. Terminal, attempt-zero
  cancellation, drift, corruption, recovery, and lease-expiry paths revoke
  retained authority. Post-E2 cancellation is instead an output-discard latch:
  its still-live permit may dispatch exactly once before expiry, then E3
  discards and charges.
- Candidate completion is callback-independent: E3 re-queries exactly one
  same-workflow ReviewItem and its contiguous immutable evidence children,
  recomputes the exact selected message-set HMAC from prepared slot identities
  and current canonical refs, requires the ReviewItem permission to equal the
  recomputed strictest selected-evidence permission, then re-derives
  candidate/terminal HMACs. Completion or replay mismatch persists only
  bounded `evidence_binding_mismatch` failure state with no candidate.
- Alembic head `9d7f3a1c6e20` conditionally replaces the extraction lifecycle
  constraint when the Task 2 table exists, so supported pinned legacy schemas
  remain upgradeable. Real `7c -> 9d`, empty downgrade/cycle, retained-row
  upgrade, and retained-state downgrade refusal are covered.
- The product service keeps disabled mode on exact V2.0. Shadow/enforce
  requires a supplied V2.1 launch authority, persists the exact V2.1 request,
  and remains in `created` until Task 12 registers the V2.1 graph. Missing or
  invalid authority fails bounded and performs no drafting/provider call.
- Round-three verification used fake providers only: `147` core Task 3
  unit/adapter tests, `47` real PostgreSQL lifecycle/authority/concurrency
  cases, `220` proportional Task 1/2 regressions, `118`
  service/API/integration tests, and `5` standalone migration tests, zero
  skips. Barrier tests prove cancellation commits while success, failure, or
  timeout provider I/O remains blocked, after which E3/failure accounting is
  terminal and candidate-free.
