# Current session handoff

2026-09-21: **E/D.1/S-1~S-3 implemented; task and whole-Slack reviews CLEAN.
Formal release remains NOT CLEAN and capability P1 is unresolved.**

## Resume here

- Local service script maintenance: see [scripts guide](../../../scripts/README.md)
  for start/stop/restart, service checks, provider configuration vs metadata
  connectivity, and missing `.env` entries. Reuse the existing key; actual `.env`
  remains unchanged. Normal launch is non-demo and does not auto-seed or auto-start
  Docker. Old unmanaged listeners must be stopped by their original owner, not
  by port-based killing. No paid model call, rollout or release authority granted.
  `.env.example` now documents E/D1 settings and explicit optional consumers;
  never copy it wholesale over a populated environment. Formal R1 remains planning.

- Local UI startup fix: demo seeding must bootstrap the runtime key identity
  before ingesting keyed sample sources. `init_db` now uses the existing bootstrap
  first; orphaned keyed DBs still fail closed. Original `.tmp/paraworks-smoke.db`
  was preserved; the recovered local demo uses `.tmp/paraworks-smoke-keyed-20260921.db`.
  Start via `scripts/start-smoke.ps1 -DatabasePath .tmp/paraworks-smoke-keyed-20260921.db`
  after stopping existing servers, not by launching duplicate listeners. Keep
  `UV_PROJECT_ENVIRONMENT=.venv-task4-r3-review` for this workstation.
  This is SQLite UI smoke, not enabled production GraphRAG or release approval.

- Worktree: `.worktrees/review-hitl-v2-design`, branch `codex/rag-orchestrator-agent`.
  Verified code: `9066d8a30f02e97058c37bdf6d77276ea2bd329f`. Later docs-only commits
  do not rerun or replace this evidence. Inspect fresh HEAD/status.
- Read [roadmap](../../../plan.md), [Slack spec](../specs/2026-09-20-slack-recovery-design.md),
  [S3 runbook](s-3-slack-integrated-demo.md), and latest [portfolio](../../portfolio-log.md).
- Next is **planning**: decide formal release scope and R1 threat model before
  implementing release changes. Do not
  restart completed Tasks1–22, E/C1–C3, or S1–S3. Actual Slack source selection is a
  separate user choice, not a prerequisite for the completed synthetic scenario.
- Keep six pre-existing stat-only D/E/common documents untouched. Root `main` is a
  separate checkout; no push, merge, paid run, `.env` or rollout was authorized.

## Synthetic integration boundary

Shared sync with exact local adapter → current signed source → registered Slack
draft → pending Review → real authorized human approval → actual PG/pgvector and
Neo4j → D1 answer cache. No valid source signature or approved target is seeded.
Public Review catalog/DTO, approval/permission/cost policies and default flags stay.

Raw Slack discovery/index eligibility remains closed. An approved graph child can
be reconstructed only through its current scoped approved envelope and exact
source/snippet/version. Structural format recognition is not authority. Pre-send
C5 accepts a non-indexed child only through the bound, freshly revalidated graph
path; finalization/cache recheck the same dependencies. Pending, unsigned, forged,
revoked or stale child evidence cannot use this path.

`SLACK_TEN` remains the exact historical selector set. Fresh ten passed; the active
deferred Slack set is empty. The default offline runner still runs those ten and
refuses an empty historical set. This says nothing about a full release run.

Use `.venv-task4-r3-review/Scripts/python.exe` with explicit selectors and fresh
`.tmp` directories. Offline runners disable dotenv/file secrets and inherited
provider settings. Actual DB runner requires process-only disposable PG/Neo4j
locators; permits only its explicit loopback graph port and drops only leased
schemas/unique graph scopes. Metadata/scorer bootstrap is not a full migration gate.

Final whole-Slack P1 (distinct from formal capability P1) is fixed and re-reviewed:
same-body legacy deliveries narrow source/chunk permissions without signing;
Slack packets filter both levels before model input and retain the stricter label.
Final affected 155 and historical ten passed, transport 0. Actual S3 DB demo seven
passed before this separate legacy fix; no claim of rerunning it afterward.

The task-created `paraworks-e-postgres` and `paraworks-e-neo4j` containers are now
stopped, not removed; their data remains. Restart those exact containers for a
new explicitly configured local DB run. Existing `paraworks-postgres` remained
stopped and untouched. No credentials are recorded here.

## Prior evidence and limits

- [S1](s-1-slack-synthetic-ingestion.md): bounded late replies/cursors, strictest
  permission and explicit discovery limits. [S2](s-2-slack-synthetic-review.md):
  actual Review, current parent evidence, PG cursor parity and vector lifecycle.
- [E1](e-1-graph-projection.md), [E2](e-2-relationship-retrieval.md),
  [E3](e-3-graphrag-comparison.md): canonical ordered relationships, actual drivers,
  restart/revoke checks and fixed synthetic comparison. Neo4j is never authority.
- [C1](c-1-answer-cache-storage.md), [C2](c-2-answer-cache-runtime.md),
  [C3](c-3-answer-cache-comparison.md): signed TTL cache, fresh retrieval/final
  revalidation and run/cost/audit on hits. Cache and graph stay default-off.
- Fake costs are simulated accounting; no live quality, latency SLO, source
  recovery or actual API savings claim. Unknown/out-of-window threads and
  undelivered source changes cannot be claimed discovered.

## Deferred release work

F1/F2 functional evidence exists. Formal reader/30-case runner, reviewer/OAuth,
quality publication and run CLI remain deferred. Existing memory capability P1
and [R1 threat-model proposal](../specs/2026-09-20-remaining-deliverables-design.md#r1-실행-권한과-위협-모델변경안)
remain unresolved; S3 does not mint or reuse paid authority. Live Slack requires
explicit choice/access/consent for an owned workspace, authorized export or other
source. Exports do not prove live connector behavior. See [archived handoff](../archive/2026-09-20-session-handoff-history.md)
for older details, not current next-step instructions.
