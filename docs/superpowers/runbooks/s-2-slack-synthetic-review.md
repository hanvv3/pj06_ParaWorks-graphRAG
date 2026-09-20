# S-2 local synthetic Slack: source authority, Review and retrieval

2026-09-21. Starts at CLEAN S1 `70d93b6899964fae52e9ab048cb460c228c14d3a`.
S2 implementation verified; independent review pending. Invented local observations,
not recovered company history. Formal release/capability P1 remain NOT CLEAN.

## Entry and authority boundary

`LocalSyntheticSlackConnector` takes copied in-memory channel/message/reply/user
data and explicit `deleted_messages` pairs. It owns its local client, accepts no
token/network-client parameter, uses `https://synthetic.invalid`, and reuses the
Slack adapter, registry manifest and `sync_connector_events`. There is no route,
OAuth/factory selection, rollout flag or provider configuration for this adapter.

Only the exact adapter passed out-of-band by shared sync enables canonical Slack
ingestion. A `synthetic` flag, mock manifest or fabricated metadata cannot do so.
New sources use the existing server parser, canonical content signature, current
document pointer, key-generation/corpus locks and post-commit reconciliation.
Unsigned ID collisions fail; a normal connector cannot update a signed synthetic
row. Existing unsigned rows remain observations with no current authority.

Canonical Slack timestamp is exact decimal seconds with six fractional digits.
Semantic workspace/channel/thread identifiers and active/deleted state join the
canonical hash. Replay skips unchanged sources. Explicit deletion is a signed
tombstone observation; Python and SQL current-authority checks reject it. Live
Slack history does not discover deletion merely because this local event exists.

## Review and parent evidence

`agent_runtime.slack_synthetic_review.create_local_slack_review_items` is the
internal local entry. Call it after shared sync with an explicitly supplied local
model and current user. It bounds sources, filters permissions before the model,
resolves current authority, and uses an exact internal `slack_agent` catalog.
The registered real SlackAgent runs through existing ReviewDraftService and a real
LangGraph StateGraph. Tests also inject a real LangChain Runnable with fake output.
No hardcoded candidate/approval rows or raw signature seeds are used.

The default fixed Review catalog and public Review source/agent DTOs still exclude
Slack. The internal catalog uses the same exact-manifest validation; stored-agent
validation consults that injected catalog. Existing source revalidation, leases,
prompt/evidence cache, C5 candidate binding, costs, and persistence are reused.
The internal creation identity exposes only the already-used client request ID.
Internal `CanonicalSourceType`/SourceVersionRef additionally accept Slack's existing
`CHANNEL:timestamp` IDs. No DB migration, public output schema, approval policy,
permission policy or token budget policy changes.

Signed replies contain only their own text. Connector-cached parent text and user
are removed before signing. Current parent chunks are separately resolved within
the same bounded source window and carry their own source/version/permission
bindings. Missing, unsigned or inaccessible parents contribute no text. Parent
edit/delete/restriction therefore affects approved knowledge through its own link;
a new signed reply cannot give stale parent text new authority.

Candidates are `pending_review`, C5 evidence-bound and not trusted knowledge.
Current human ReviewTransitionService approval creates normal trusted provenance;
unauthorized/stale approval is rejected, and retry is idempotent. V1 indexing/search
and D V2 **trusted knowledge** citation resolution accept current signed Slack
evidence. D V2's public raw-source observation whitelist is unchanged. Raw ingestion
and pending candidates do not become trusted facts.

## Fresh evidence

Interpreter: `.venv-task4-r3-review/Scripts/python.exe`. Commands run at the worktree
root. Harness disables dotenv/file secrets, inherited provider settings and real
HTTP/socket transports; creates fresh `.tmp/slack-s1-*` basetemp/cache directories.
The S1 runner name remains compatible; it is also used for S2 selectors.

```powershell
.venv-task4-r3-review/Scripts/python.exe -m backend.tests.slack_offline_runner backend/tests/test_slack_synthetic_authority.py -q --tb=short
.venv-task4-r3-review/Scripts/python.exe -m backend.tests.slack_offline_runner backend/tests/test_slack_synthetic_authority.py backend/tests/test_review_v2_drafting.py backend/tests/test_review_v2_preflight.py -q --tb=short
.venv-task4-r3-review/Scripts/python.exe -m backend.tests.slack_synthetic_pg_runner backend/tests/test_slack_synthetic_postgres.py -q --tb=short
```

The PG command requires a process-only `PARAWORKS_TEST_POSTGRES_URL` for a disposable
127.0.0.1 database/role ending `_test`. No DSN is saved. Each test leases a new schema,
creates ORM tables, initializes the real key service with an ephemeral fixture key,
creates pgvector storage, and drops the leased schema afterward. This is real SQL,
locking, Review and vector I/O evidence; it is not a full migration/release gate.

- RED source entry: 6 failed (missing explicit adapter); GREEN 6 passed.
- RED local Review entry: 3 failed/6 passed (missing internal bridge); GREEN with
  existing drafting tests 47 passed.
- RED explicit delete delivery: 3 failed/11 passed; GREEN 14 passed.
- RED D V2 trusted citation: 1 failed because Slack was absent from canonical
  evidence whitelist; GREEN 15 passed. Only trusted-child support was extended.
- Added existing-behavior verification: real Runnable, stale/unauthorized approval,
  missing evidence, registered prompt change/cache, and signed/unsigned signal
  classifier parity. No production change was needed for those checks.
- Affected SQLite selection: **215 passed in 11.64s**, external attempts **0**.
- Actual PostgreSQL/pgvector final: **3 passed in 29.59s**, external attempts **0**.
  Numeric channel cursor and bounded known-thread query parity are now verified.
  Covers late reply/replay, pending→human approval/retry, index→skip accounting,
  current vector search, and parent edit/delete/restriction filtering old vectors.
- Final formatted Review/preflight selection: **89 passed in 7.64s**. Final
  synthetic-only selection: **24 passed in 3.67s** (includes a later negative
  invalid-signature DB CHECK test); both external attempts **0**.
  Commit is recorded in the S2 task report.
  Ruff and `git diff --check` are required before commit.

## Remaining work

Historical SLACK_TEN rerun: **10 failed in 2.07s**, external attempts **0**; no
skip/xfail/deletion or manifest adjustment. S1's six-authority-fixture diagnosis was
incorrect: ranked extraction removes wording without recognized work-action
signals (e.g. “Redis should support…”). A four-case signed/unsigned comparison
proves the independent signal cause. Three PKCE opt-in expectations and one fake
`job_id` signature remain unchanged. See corrected exact IDs in the S1 runbook.

S3 owns meaningful historical fixture repair, release manifest accounting, E graph
and D.1 cache lifecycle demo and frontend/integration handoff. This S2 result does
not claim those gates. Bounds/discovery limits from S1 remain: undelivered edits,
deletes, revokes and unknown/out-of-window old roots cannot be claimed current.
No live Slack/OAuth/LLM/embedding calls, `.env` reads, push or rollout occurred.
