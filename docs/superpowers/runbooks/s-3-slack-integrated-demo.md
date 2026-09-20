# S-3 — synthetic Slack → Review → GraphRAG → answer cache

2026-09-21: implementation complete; independent S3 review pending. Base
`d91ed9df63e4bfdcdcf130db93fb04a12f378aab`. E/D1/S1/S2 are CLEAN. This is
synthetic functional evidence, **not formal release approval**. Formal release
remains NOT CLEAN; capability P1 and the deferred R1 threat-model decision remain.

## What the scenario proves

`test_slack_synthetic_demo.py` uses S1's invented connector payload and S2's
`sync_connector_events`, server signing, registered draft, real pending Review,
and authorized human approval/retry. No valid target signature or approved row
is seeded. Actual PostgreSQL/pgvector, actual Neo4j projection/traversal, and the
existing D1 production service composition run with fake extraction/generation.

- Initial/late sync fetch 3/1; ingestion creates zero Review items. Replay skips
  one source event. Draft creates one pending item; replay saves one extraction
  call. Pending evidence yields no supported answer or generation call.
- Human approval creates history/timeline. Exactly two approved documents are
  embedded locally; repeat indexing saves two embedding calls. Raw Slack is not
  directly discoverable/indexable, even after approval.
- The approved graph has four nodes/four relationships: two approved outputs
  linked to parent/reply evidence. Cold/warm requests each carry four influencing
  evidence slots and use actual Neo4j. The warm request has a distinct persisted
  run/cost/audit and zero generation cost; only the cold request calls the fake
  provider. This is simulated accounting, not paid API savings or a latency SLO.
- Shared-sync source edit/delete/permission reduction prevents the old graph and
  cache answer from being exposed before graph reconciliation; reconciliation
  removes the relationships. Reduced current actor access also refuses reuse.
  Separate approval-link corruption tests cover inactive approval. There is no
  claim of a new human approval-revocation UI/workflow.
- Missing/mismatched bound graph paths refuse provider send. A shared-sync
  deletion after cache lookup is caught by final validation and redacts output.
  Graph-off and cache-off preserve the existing rollback behavior.

## Narrow consumer contract

The raw discovery/index whitelist in `source_observations.py` remains unchanged.
The new exact approved-child resolver validates current scoped approved envelope,
canonical source/link/snippet, signature and source version. Projection, graph
retrieval and exact candidate/final reconstruction share this boundary.

Slack is recognized in a separately named **structural format** set; this grants
no discovery or authority. Under the existing C5 transaction, a deliberately
non-indexed Slack child may substitute for an absent lexical row only when its
full identity, permission and canonical version envelope match a node in the
freshly revalidated bound graph path. Mere graph metadata is insufficient.
Public DTOs/catalogs, raw discovery whitelist, approval/permission/cost policy,
runtime hash algorithms and production default-off flags are unchanged.

The controller approved this narrow integration decision. Risk if wrong:
unintended direct materialization or extra evidence exposure. Direct-raw denial
before/after approval, unsigned/forged/unapproved child denial, stale source/
approval/path and final cache refusal tests plus independent review mitigate it.

## Reproduction and verification

Run from the review worktree with `.venv-task4-r3-review/Scripts/python.exe`
(shown as `PY` below). Never use bare root pytest. The runners create fresh
`.tmp/slack-s1-*` temp/cache directories, disable dotenv/file-secret sources and
inherited provider settings, and reject outbound provider HTTP/socket calls.

```text
PY -m backend.tests.slack_offline_runner
PY -m backend.tests.slack_offline_runner backend/tests/test_slack_graph_children.py backend/tests/test_release_contracts.py backend/tests/test_backend_release_matrix.py -q --tb=short --show-capture=no
PY -m backend.tests.slack_synthetic_demo_runner backend/tests/test_slack_synthetic_demo.py -q -rP --tb=short --show-capture=stdout
```

For the last command, explicitly supply process-only
`PARAWORKS_TEST_POSTGRES_URL` and `PARAWORKS_TEST_NEO4J_URI/USER/PASSWORD` for the
approved disposable services. The PG validator requires loopback/test database/
test role; the Neo4j exception permits only `bolt://127.0.0.1:17687`, only with
explicit PG. Do not persist credentials or loosen the ordinary offline guard.
Each case leases its own schema and graph security scope; teardown deletes only
that scope and drops only its lease. ORM metadata plus the existing scorer
installation is **not a full migration/release gate**.

Fresh bounded gates in this implementation session:

| Gate | Result |
| --- | --- |
| Exact historical ten, no selectors | 10 passed; external attempts 0 |
| E/source/history/release affected selection | 85 passed, 3 PG-gated skips |
| API/Review/release/secret selection | 104 passed |
| Final cost/projection/authority selection | 214 passed, 3 PG-gated skips |
| Explicit actual-PG E/C2/S2 selection | 8 passed, no skips |
| Final seven-case actual PG/pgvector/Neo4j demo | 7 passed in 228.41s; external attempts 0 |
| Frontend existing public interface | `tsc --noEmit --incremental false`, exit 0 |

The later explicit PG selection includes all three previously gated E tests;
the skips are not presented as actual-PG coverage. C2 uses fake traversal while
S3 and E's parameterized driver test use actual Neo4j. Final seven-case demo raw
output is retained locally at `.tmp/s3-final-evidence/demo.log`; the task report
records its exact result and all expanded gate commands. No broad formal release
suite was rerun or declared passing.

The edit-case artifact reports three total fake generation calls because it
also executes cache-off and graph-off rollback. The initial cold/warm pair still
uses exactly one generation call; the denied stale request adds none.

## Historical-ten repair, not test suppression

The six extraction fixtures lacked the ranked classifier's accepted action
wording. Their new work-action text preserves extraction, permission, routing,
cost and cache assertions; signing them would not repair that intent. One
quality count now uses the post-source-seeding baseline because canonical Gmail
fixture seeding itself creates two pending items. Three PKCE fixtures now assert
the existing hard-disabled Slack policy, including explicit `use_pkce=True`;
Google behavior is unchanged. One OAuth sync fake accepts the current `job_id`
argument and asserts synchronous `None`. No production OAuth policy changed.

`SLACK_TEN` remains immutable/reproducible; the separate active
`DEFERRED_SLACK_FAILURES` is empty only after the exact ten passed. Contract tests
require no historical failure allowance/deselection and reject an empty default
historical runner selector. No tests were deleted, skipped or xfailed to hide
the old baseline. A stale prepared-evidence v2 digest fixture was also corrected
to the existing v3 canonical contract (including graph paths); baseline HEAD
produced the same digest, independently reconstructed in the assertion. Runtime
hashing was not changed.

## Next decision and limits

Complete independent S3/cross-slice review, then plan formal release readiness.
Live Slack is a separate user-owned decision: owned workspace with consent,
authorized export, or another approved source. Exports do not prove live API
behavior. No live Slack/OAuth/provider call, production rollout, paid authority,
`.env` use, secret file, push or bulk trust migration occurred. Unknown/out-of-
window threads or undelivered source changes remain outside discovery claims.

## Final whole-Slack review fix — unsigned permission-only delivery

The final review found a P1 in ordinary unsigned Slack ingestion: same-body
deduplication ignored delivered restrictions, leaving stored chunks visible at
the old level. The final fix wave narrows source and existing chunks even when
the body hash is unchanged, including a previously restricted source with stale
internal chunks. It neither signs nor reparses legacy data; subsequent unchanged
replay remains skipped and a lower-permission replay cannot broaden visibility.

The regression also exposed missing permission filtering in the shared Slack
packet builder. It now checks both source and chunk against the current allowed
levels before ranking/windowing and retains their strictest label. This enforces
the existing policy for `/slack/agent-review` and company-memory consumers; role
names alone confer no visibility. Tests cover low-privilege exclusion, authorized
restricted inclusion, both mismatch directions and no-input cost semantics.
Final-fix affected tests: **155 passed in 20.09s**, historical ten **10 passed in
1.31s**, external attempts **0**; changed-file Ruff/diff clean. The existing bridge
fixture now explicitly allows restricted evidence, instead of relying on an admin
role label with public/internal default permissions. Verification commands and raw
logs are appended to the S3 task report. Independent
final review remains pending; formal release/P1 status above is unchanged (this
Slack finding is distinct from the deferred formal-release capability P1).
