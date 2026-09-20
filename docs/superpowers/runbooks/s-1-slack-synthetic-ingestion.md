# S-1 synthetic Slack ingestion

2026-09-21. Base `8ad1ed9909406f8305bf579fe790d7b2103e2090`, code
`1ddfc21f5bfcf1ccf4537af4cc1d58308fc847ec`. S1 implemented, independent review pending.
Invented fixture ingestion only, not recovered company history or live Slack.

## R1 corrections

- Missing/failed/incomplete channel metadata now yields `restricted`. Only explicit
  `is_channel=True` plus `is_private=False` permits internal, and private/IM/MPIM
  metadata overrides it. Fetch-budget exceptions propagate through optional metadata
  and fallback paths, so a conversations.list page overflow cannot persist a partial sync.
- History-only thread broadcasts are replies from their first observation. Missing
  parent text/user remains missing (`thread_parent_missing`, `reply_without_parent`);
  a reply's own text/user is never substituted for a root. When a real root observation
  exists, it supplies context without rewinding the newer per-thread cursor.
- New failures reproduced before fixes: **6 failed, 8 deselected in 1.15s**. Focused
  GREEN **48 passed in 1.98s**; final affected selection below **101 passed in 3.94s**,
  zero external attempts, no skips/warnings. Ruff and diff checks pass.
- SQLite/fake evidence only. The production PostgreSQL JSON/numeric cursor aggregate
  remains unverified on a real PostgreSQL server; S2/integration must establish parity.
  The earlier historical-ten classification and source authority/S2 boundary remain.

## Offline reproduction

```powershell
.venv-task4-r3-review/Scripts/python.exe -m backend.tests.slack_offline_runner
.venv-task4-r3-review/Scripts/python.exe -m backend.tests.slack_offline_runner backend/tests/test_slack_synthetic_ingestion.py backend/tests/test_slack_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_google_connector.py backend/tests/test_mock_sync.py -q --tb=short
```

The harness disables dotenv/file-secret sources, removes inherited settings/provider
variables before application startup, forces in-memory SQLite, disables plugin autoload,
and creates fresh `.tmp/slack-s1-*` temp/cache directories. It blocks real HTTP transports
and socket connects, counts attempts, and permits only the Windows stdlib socketpair's
own connection for TestClient. MockTransport remains usable. No local `.env` is read.
Initial affected selection: **93 passed in 3.82s**; R1 final: **101 passed in 3.94s**.
Both had attempts **0**, no skips/warnings.
Changed-file Ruff and diff checks pass. No PostgreSQL/Neo4j/live API evidence is claimed.

## Historical ten: fresh classification, still unresolved

S2 correction (2026-09-21): the six rows formerly described as unsigned-authority
failures actually lose their fixture text in the ranked work-signal filter. The
same text is excluded with or without a server signature; recognized work-action
wording is included with either. Current source authority is a separate S2 gap.

The first command runs the exact `SLACK_TEN` tuple; no skip/deselection/xfail was added.
Before product edits: **10 failed in 2.13s**. After S1: **10 failed in 2.17s**.
Both runs had **0 external transport attempts**. Full node IDs follow.

| Node | Observed failure / classification |
|---|---|
| `backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_runs_real_agent_services` | Slack pending count 0 vs 1; fixture lacks recognized work-action wording |
| `backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_skips_agents_that_exceed_cost_budget` | `no_slack_evidence` vs `budget_exceeded`; same ranked signal filter |
| `backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_uses_cache_when_evidence_is_unchanged` | `skip` vs `run`; same ranked signal filter |
| `backend/tests/test_oauth_pkce.py::test_slack_oauth_pkce_generation` | Missing `code_challenge`; stale default-PKCE expectation, current default is `use_pkce=False` |
| `backend/tests/test_oauth_pkce.py::test_slack_callback_with_custom_redirect_uri_and_pkce` | Missing `code_verifier`; same default-contract mismatch |
| `backend/tests/test_oauth_pkce.py::test_api_endpoints_support_redirect_uri` | Missing Slack `code_challenge`; same default-contract mismatch |
| `backend/tests/test_orchestration_api.py::test_company_memory_orchestration_api_runs_agent_services` | Slack pending count 0 vs 1; ranked signal fixture wording |
| `backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_company_memory_emits_review_checkpoint_without_paid_calls` | Review IDs 5 vs 6; ranked signal fixture wording |
| `backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_cache_hit_does_not_duplicate_agent_runs_or_review_items` | `skip` vs `use_cache`; ranked signal fixture wording |
| `backend/tests/test_slack_oauth.py::test_slack_sync_endpoint_uses_installed_connection_token_without_exposing_it` | Fake sync rejects `job_id`; stale fake signature |

These ten show no S1 connector defect or remaining environment failure. Six need
meaningful work-signal fixtures; OAuth/fake tests later need current-contract
expectations without changing OAuth policy. Release manifest is unchanged. An initial
harness attempt had 7 failures/3 setup errors because blanket socket blocking broke
Windows socketpair; it was corrected before classification. Those setup errors are
harness/environment diagnostics, not product failures or baseline evidence.

## Bounds and S2 consumer contract

- SourceEvent/ConnectorManifest/registry/shared sync remain entry points. Optional
  `SlackConnector.fetch_events_since(..., known_messages_by_channel=...)` preserves
  original one-argument use. No shared DTO/storage migration. Registry reports the
  adapter's existing nine scopes; requested OAuth policy is unchanged.
- Workspace URL/team (when supplied), channel/message/thread IDs, raw snippet/user IDs,
  parent/reply participants, source links and timestamp text are preserved. Public
  channels stay internal; private/IM/MPIM and restricted parent context stay restricted.
  Cached parent context is an observation, never current source authority.
- Shared sync aggregates scoped channel cursors and selects at most 50 newest message
  observations per channel. Each known thread uses its own latest cursor; another
  thread's newer message cannot hide a late reply. Zero-reply roots within this window
  are eligible. Channel context is separate; each event contains parent plus one reply.
- Bounds: 50 selected channels, 50 thread fetches/channel, 20 HTTP pages/call.
  Fetch overflow raises before source persistence, never a successful partial batch.
  Operators must narrow scope after overflow. The DB aggregate can scan scoped rows,
  but application memory receives bounded channels/observations. No whole-workspace
  prompt, CDC, background scheduler or external service is introduced.
- Unknown old roots absent from upstream history and roots outside the observation
  window are not discovered. The fixture explicitly bootstraps an old snapshot; this
  does not prove live 7-day history can discover a 20-day root. Initial history retains
  the 7-day lookback. Undelivered edits/deletes/access revokes cannot be claimed detected.
  Legacy `channel:timestamp` source IDs remain; cross-workspace identity migration is
  outside S1. URL scoping prevents cursor/context mixing, not a new global identity scheme.
- The injected synthetic fixture labels invented records with `synthetic` and
  `fixture_origin` and uses `.invalid` links; labels grant no trust. Counts are
  fetched/created-review/skipped **3/0/0** initially (3 changed sources), **1/0/0** for
  a late reply (1 changed source), **1/0/1** for replay. No additional source/Review on replay.
- S2 must validate NEW synthetic sources server-side and create pending Review through
  existing services. Old unsigned rows, fixture flags and raw connector content hashes
  are not server signatures. S1 does not seed approval, migrate legacy trust or create
  trusted knowledge. Source mutation/revocation and graph/cache propagation remain
  S2/S3. Formal release and capability P1 remain NOT CLEAN; no push/rollout/paid calls.
