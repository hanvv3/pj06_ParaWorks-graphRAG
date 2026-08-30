# Whole-Suite PostgreSQL Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the complete ParaWorks backend release gate order-independent and privacy-safe by giving every PostgreSQL-dependent test an owned schema lease, keeping ordinary tests on an isolated SQLite database, and making one fail-closed controller prove exact coverage and cleanup.

**Architecture:** A test-only release controller validates but never creates or replaces the shared `paraworks-postgres` container, creates one owned `_test` role/database and one sentinel schema, launches hermetic serial child processes, and validates authoritative pytest sidecars. PostgreSQL consumers receive only an immutable base locator and open unique schema leases whose URLs set `search_path=<owned_schema>,public`. The controller owns database/role/process/artifact cleanup; each lease owns only its exact schema. Production application modules, migrations, trust policy, and the approved Slack baseline remain unchanged.

**Tech Stack:** Python 3.12, pytest 9, SQLAlchemy 2, psycopg 3, Alembic, PostgreSQL 17, pgvector, SQLite, Ruff, Docker CLI, PowerShell-compatible subprocess execution

**Spec:** `docs/superpowers/specs/2026-08-29-whole-suite-postgresql-isolation-design.md`

**Status:** Written spec user-approved; this implementation plan awaits user approval; test/helper/controller implementation is not authorized.

## Global Constraints

- Execute infrastructure Tasks 1–11 only at Deliverable C.5 Task 16 entry, after product Tasks 6–15 are implemented and reviewed. Task 12 is the official backend proof mapped to C.5 Task 16 Steps 4–7. Writing or approving this plan does not reorder those product tasks.
- Do not change `backend/app/**`, `backend/migrations/**`, Alembic revisions, public API schemas, permission rules, token/cost policy, Review Queue trust boundaries, promotion/revoke/duplicate rules, LangChain/LangGraph behavior, or frontend behavior under this plan.
- Keep the exact shared-service contract: container `paraworks-postgres`, image `pgvector/pgvector:pg17`, Compose service `postgres`, `running:healthy`, host binding `127.0.0.1:55432->5432`, and server identity `paraworks:postgres`. Refuse any mismatch without mutation.
- Never invoke `docker compose up`, `docker compose down`, container remove, volume remove, or `scripts/start-pgvector-dev.ps1`. The controller may create and drop only its generated role, database, sentinel, and lease schemas.
- Build every child environment from a literal allowlist. Do not copy the parent environment, load the repository `.env`, or expose environment values in output.
- Keep release execution serial. Refuse `xdist`, `-n`, worker environment variables, plugin autoload, or an unlisted pytest plugin before test collection.
- Use fake/deterministic providers only. Strip live LLM, embedding, Slack, Google, OAuth, connector, tracing, and paid-evaluation credentials/flags from every child. Do not call any provider API.
- The exact ten approved Slack node ids are the only deferred failures/deselections. A baseline change, new skip, new xfail, or new exclusion requires a separate human decision.
- The Review V2 provenance diagnosis remains a hypothesis until a clean leased `-vv --tb=long` RED reproduces the exact six failing nodes at the evidence boundary. Do not change its fake draft before that observation.
- Raw child stdout/stderr, traceback, JUnit, collection, sidecar partials, DSNs, passwords, database/role/schema names, prompts, evidence, and model output never enter the aggregate report or Git.
- Every newly added parametrized test uses explicit stable, non-sensitive `ids=` from its first RED. Task 7's collection RED is therefore expected to find only the eight pre-existing families frozen there.
- Every behavior slice follows `failing test -> smallest implementation -> GREEN -> Ruff/compile/diff -> reviewed commit`. Do not defer all code into one final commit.
- A `--child-id` run is a focused development aid only. Its report must set `release_proof=false` and can never satisfy a profile or Task 16 release gate. Official proof commands omit `--child-id`.
- Cleanup failure overrides every earlier success, verification failure, or expected Slack-baseline match. Cleanup attempts continue after an earlier cleanup error.
- If implementation requires a production-code change or a change to a human-gated contract, stop and write a separate spec/plan amendment before editing that boundary.

## Frozen Test-Only Contracts

### Files and ownership

Create:

- `backend/tests/release_contracts.py`
- `backend/tests/test_release_contracts.py`
- `backend/tests/release_bootstrap.py`
- `backend/tests/test_release_bootstrap.py`
- `backend/tests/release_evidence_plugin.py`
- `backend/tests/test_release_evidence_plugin.py`
- `backend/tests/postgres_isolation.py`
- `backend/tests/test_postgres_isolation.py`
- `scripts/backend_release_matrix.py`
- `backend/tests/test_backend_release_matrix.py`
- `backend/tests/fixtures/deferred_backend_baseline_v1.json`
- after observed release proof, `docs/superpowers/runbooks/backend-release-matrix.md`

Modify only the safe-id and PostgreSQL test consumers named in Tasks 7–10, plus the status/runbook documents named in Task 12. No production file is in scope.

### Shared release types

`backend/tests/release_contracts.py` owns the types shared by the controller and plugin:

```python
ReleaseProfileName = Literal[
    'settings-diagnostic',
    'postgres',
    'compatibility',
    'non-slack',
    'full',
]
ReleaseOutcomeCode = Literal[
    'preflight_refused',
    'environment_refused',
    'collection_mismatch',
    'evidence_refused',
    'verification_failed',
    'cleanup_failed',
    'baseline_mismatch',
    'passed',
]
FailureReasonCode = Literal['review_v2_post_c5_provenance_guard']
ChildKind = Literal['canonical_collection', 'verification']

@dataclass(frozen=True)
class ReleaseChild:
    child_id: str
    kind: ChildKind
    pytest_argv: tuple[str, ...]
    timeout_seconds: int
    allowed_pytest_exit_codes: tuple[int, ...] = (0,)

@dataclass(frozen=True)
class ReleaseProfile:
    name: ReleaseProfileName
    children: tuple[ReleaseChild, ...]
    timeout_seconds: int

@dataclass(frozen=True)
class ReleaseInvocationIdentity:
    run_id: str
    profile: ReleaseProfileName
    child_id: str
    invocation_hash: str

@dataclass(frozen=True)
class ReleaseEvent:
    nodeid: str
    phase: Literal['setup', 'call', 'teardown']
    outcome: Literal['passed', 'failed', 'skipped']
    wasxfail: bool
    collection_index: int
    failure_reason_code: FailureReasonCode | None

@dataclass(frozen=True)
class LeaseLifecycleEvent:
    lease_id_sha256: str
    state: Literal['created', 'dropped']

@dataclass(frozen=True)
class ReleaseEvidenceSidecar:
    schema_version: int
    identity: ReleaseInvocationIdentity
    collection_nodeids: tuple[str, ...]
    collection_sha256: str
    events: tuple[ReleaseEvent, ...]
    lease_events: tuple[LeaseLifecycleEvent, ...]
    pytest_exitstatus: int
    events_sha256: str
    complete: Literal[True]

@dataclass(frozen=True)
class ReleaseManifest:
    controller_schema_version: int
    sidecar_schema_version: int
    explicit_pytest_plugins: tuple[str, ...]
    deferred_slack_failure_nodeids: tuple[str, ...]
    postgres_release_critical_modules: tuple[str, ...]
    safe_parameter_case_counts: Mapping[str, int]
    profiles: Mapping[ReleaseProfileName, ReleaseProfile]
```

The JSON manifest uses schema version `1`, sidecar version `1`, and exactly one plugin: `backend.tests.release_evidence_plugin`.

### Exact profile-to-child manifest

`settings-diagnostic` has canonical child `settings-collection` and verification child `settings-contracts`. The verification child selects exactly these six nodes:

```text
backend/tests/test_agent_runtime_checkpointing.py::test_enabled_checkpoint_mode_rejects_unsupported_backend_without_secret
backend/tests/test_agent_runtime_checkpointing.py::test_production_rejects_the_local_default_fingerprint_secret
backend/tests/test_auto_review_contracts.py::test_non_disabled_mode_rejects_local_default_fingerprint_secret
backend/tests/test_auto_review_contracts.py::test_disabled_sqlite_smoke_may_use_process_local_placeholder_without_durable_ready_state
backend/tests/test_auto_review_contracts.py::test_process_local_sqlite_smoke_is_limited_to_disabled_in_memory_url[sqlite-memory-disabled]
backend/tests/test_auto_review_contracts.py::test_file_backed_sqlite_placeholder_refuses_every_c5_bound_durable_write
```

`settings-contracts.pytest_argv` is the six selectors in that order followed by `-q`. `settings-collection.pytest_argv` is the same six selectors followed by `--collect-only` and `-q`.

`postgres` has canonical child `postgres-collection` and these nine disjoint verification children: one helper-contract child plus the eight consumer modules.

| Child id | Exact module |
|---|---|
| `postgres-isolation-contract` | `backend/tests/test_postgres_isolation.py` |
| `postgres-agent-runtime-checkpoint` | `backend/tests/test_agent_runtime_postgres_checkpoint.py` |
| `postgres-auto-review-migration` | `backend/tests/test_auto_review_migration.py` |
| `postgres-auto-review-provenance` | `backend/tests/test_auto_review_provenance.py` |
| `postgres-review-transition` | `backend/tests/test_review_transition_postgres.py` |
| `postgres-review-v21-extraction` | `backend/tests/test_review_v21_extraction_postgres.py` |
| `postgres-review-v2` | `backend/tests/test_review_v2_postgres.py -vv --tb=long` |
| `postgres-pgvector` | `backend/tests/test_pgvector_integration.py` |
| `postgres-auto-review-c5` | `backend/tests/test_auto_review_postgres.py` |

The last module must exist when Task 16 begins. Its absence is `collection_mismatch`, never a skip.

Each PostgreSQL verification child's `pytest_argv` is its exact module followed by `-q`, except `postgres-review-v2`, whose suffix is `-vv --tb=long`. `postgres-collection.pytest_argv` is the nine module paths in table order followed by `--collect-only` and `-q`.

`compatibility` has canonical child `compatibility-collection` and six disjoint verification children. Their selector arrays are frozen as follows:

1. `compatibility-contracts-provenance`:

```text
backend/tests/test_auto_review_contracts.py
backend/tests/test_auto_review_cost_policy.py
backend/tests/test_auto_review_migration.py
backend/tests/test_assistant_models.py
backend/tests/test_keyed_mutation_guard.py
backend/tests/test_auto_review_key_bootstrap.py
backend/tests/test_provider_send_fence.py
backend/tests/test_review_v21_preflight.py
backend/tests/test_review_v21_drafting.py
backend/tests/test_review_v21_extraction.py
backend/tests/test_review_resolution_actors.py
backend/tests/test_auto_review_provenance.py
```

2. `compatibility-serving-revoke`:

```text
backend/tests/test_auto_review_revocation.py
backend/tests/test_auto_review_quality_revoke.py
backend/tests/test_auto_review_source_reconciliation.py
backend/tests/test_auto_review_source_reconciliation_admin.py
backend/tests/test_source_content_signature.py
backend/tests/test_review_evidence_visibility.py
backend/tests/test_google_connector.py
backend/tests/test_connector_ingestion_contract.py
backend/tests/test_document_ingestion_service.py
backend/tests/test_rag_indexing.py
backend/tests/test_pgvector_store.py
backend/tests/test_pgvector_integration.py
backend/tests/test_knowledge_api.py
backend/tests/test_dashboard_api.py
backend/tests/test_review.py
backend/tests/test_todos_api.py
backend/tests/test_notifications_api.py
backend/tests/test_mock_sync.py
backend/tests/test_integration_runtime_status.py
backend/tests/test_search_permissions.py
backend/tests/test_search_retrieval_backend.py
backend/tests/test_ask_api.py
backend/tests/test_assistant_api.py
backend/tests/test_assistant_service.py
backend/tests/test_assistant_email_agent.py
backend/tests/test_company_memory_orchestration_service.py
backend/tests/test_orchestration_api.py
backend/tests/test_project_memory_api.py
backend/tests/test_rag_orchestrator_agent.py
backend/tests/test_rag_orchestrator_service.py
```

3. `compatibility-policy-validation`:

```text
backend/tests/test_auto_review_eligibility.py
backend/tests/test_auto_review_policy.py
backend/tests/test_auto_review_validator.py
backend/tests/test_auto_review_model_router.py
backend/tests/test_auto_review_validation_store.py
backend/tests/test_auto_review_orchestrator.py
backend/tests/test_auto_review_rollout.py
backend/tests/test_auto_review_audit.py
backend/tests/test_auto_review_rollout_admin.py
backend/tests/test_auto_review_launch_confirmation.py
backend/tests/test_review_workflow_facade.py
```

4. `compatibility-v21-release`:

```text
backend/tests/test_review_v21_state.py
backend/tests/test_review_v21_graph.py
backend/tests/test_review_v21_service.py
backend/tests/test_review_v21_api.py
backend/tests/test_auto_review_call_recovery.py
backend/tests/test_auto_review_api.py
backend/tests/test_auto_review_golden.py
backend/tests/test_auto_review_evaluation_cli.py
backend/tests/test_auto_review_extraction_compatibility_cli.py
backend/tests/test_auto_review_postgres.py
backend/tests/test_auto_review_smoke.py
```

5. `compatibility-review-v2`:

```text
backend/tests/test_review_v2_schemas.py
backend/tests/test_review_v2_preflight.py
backend/tests/test_review_v2_drafting.py
backend/tests/test_review_transitions.py
backend/tests/test_review_v2_graph.py
backend/tests/test_review_v2_service.py
backend/tests/test_review_v2_api.py
backend/tests/test_review_v2_postgres.py
backend/tests/test_review_transition_postgres.py
backend/tests/test_review_knowledge_promotion.py
backend/tests/test_review_rbac.py
```

6. `compatibility-agent-runtime`:

```text
backend/tests/test_agent_preflight.py
backend/tests/test_agent_runtime_state.py
backend/tests/test_agent_runtime_fingerprints.py
backend/tests/test_agent_workflow_models.py
backend/tests/test_agent_runtime_migration.py
backend/tests/test_agent_runtime_checkpointing.py
backend/tests/test_agent_runtime_lifespan.py
backend/tests/test_agent_runtime_bootstrap.py
backend/tests/test_agent_runtime_retention.py
backend/tests/test_agent_runtime_graph_versions.py
backend/tests/test_agent_runtime_checkpoint_execution.py
backend/tests/test_agent_runtime_postgres_checkpoint.py
backend/tests/test_db_init.py
backend/tests/test_db_schema_operations.py
backend/tests/test_data_reset.py
backend/tests/test_langchain_langgraph_dependency_compat.py
```

Each compatibility verification child's `pytest_argv` is its numbered selector array in the listed order followed by `-q`. `compatibility-collection.pytest_argv` is the ordered, de-duplicated concatenation of arrays 1 through 6 followed by `--collect-only` and `-q`. No other selector or flag is present.

`non-slack` has canonical child `non-slack-collection` and verification child `non-slack-backend`. Both select `backend/tests` while deselecting exactly these ten nodes:

```text
backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_runs_real_agent_services
backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_skips_agents_that_exceed_cost_budget
backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_uses_cache_when_evidence_is_unchanged
backend/tests/test_oauth_pkce.py::test_slack_oauth_pkce_generation
backend/tests/test_oauth_pkce.py::test_slack_callback_with_custom_redirect_uri_and_pkce
backend/tests/test_oauth_pkce.py::test_api_endpoints_support_redirect_uri
backend/tests/test_orchestration_api.py::test_company_memory_orchestration_api_runs_agent_services
backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_company_memory_emits_review_checkpoint_without_paid_calls
backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_cache_hit_does_not_duplicate_agent_runs_or_review_items
backend/tests/test_slack_oauth.py::test_slack_sync_endpoint_uses_installed_connection_token_without_exposing_it
```

`non-slack-backend.pytest_argv` is `backend/tests` followed by the ten `--deselect=<nodeid>` arguments in the listed order and then `-q`. `non-slack-collection.pytest_argv` is the same vector with `--collect-only` inserted immediately before `-q`.

`full` has canonical child `full-collection` and verification child `full-backend`, both selecting unfiltered `backend/tests`. Native pytest exit `1` is accepted only for `full-backend` and only when the verified sidecar proves the failure set equals the exact Slack ten, errors/skips/xfails are empty, and collection parity is exact. The controller still returns aggregate outcome `passed` with `deferred_baseline_match=true`; any difference returns `baseline_mismatch`.

`full-backend.pytest_argv` is exactly `('backend/tests', '-q')`. `full-collection.pytest_argv` is exactly `('backend/tests', '--collect-only', '-q')`.

### Hermetic child environment

The literal environment builder may pass only:

- Windows process keys: `PATH`, `SystemRoot`, `WINDIR`, `TEMP`, `TMP`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`
- POSIX process keys: `PATH`, `HOME`, `TMPDIR`, `LANG`, `LC_ALL`
- controller values: `DATABASE_URL` pointing to the child SQLite file, `PARAWORKS_TEST_POSTGRES_URL` and `PARAWORKS_PGVECTOR_TEST_DATABASE_URL` pointing to the owned base PostgreSQL database, `PARAWORKS_DEMO_MODE=true`, `PARAWORKS_ENV=local`, `AUTO_REVIEW_MODE=disabled`, `AUTO_REVIEW_ENFORCE_PERCENTAGE=0`, `AGENT_LLM_ENABLED=false`
- pytest controls: `PYTEST_ADDOPTS=`, `PYTEST_PLUGINS=`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- release identity: `PARAWORKS_RELEASE_RUN_ID`, `PARAWORKS_RELEASE_PROFILE`, `PARAWORKS_RELEASE_CHILD_ID`, `PARAWORKS_RELEASE_INVOCATION_HASH`, `PARAWORKS_RELEASE_SIDECAR_PATH`, `PARAWORKS_RELEASE_EXPECTED_DATABASE`, and `PARAWORKS_RELEASE_EXPECTED_ROLE`

It must omit `PARAWORKS_DATABASE_URL`, `PARAWORKS_DEMO_DATABASE_URL`, fingerprint secret/key overrides, `PYTEST_XDIST_WORKER`, tracing/debug controls, live provider/connector/OAuth credentials, paid-evaluation flags, and every unrecognized parent key.

### Exact owned-resource identities

The controller generates one 12-character lowercase-hex `run_id` and uses only:

```text
run prefix: paraworks_c5t16_<run_id>
database:   paraworks_c5t16_<run_id>_database_test
role:       paraworks_c5t16_<run_id>_role_test
sentinel:   paraworks_c5t16_<run_id>_sentinel
lease:      paraworks_c5t16_<run_id>_schema_<sanitized_scope>_<lower_hex>
```

Every identifier must match its anchored literal pattern and PostgreSQL's 63-byte bound before use. Database and role must end in `_test`. The controller latches the exact database/role into each child; suffix or prefix membership alone never confers ownership.

## Task 1: Freeze Release Contracts and the Exact Manifest

**Files:**

- Create: `backend/tests/release_contracts.py`
- Create: `backend/tests/test_release_contracts.py`
- Create: `backend/tests/fixtures/deferred_backend_baseline_v1.json`

- [ ] **Step 1: Write manifest/parser tests first**

Cover:

- exact schema versions and plugin tuple;
- exact five profile names and child ids above;
- exact canonical `pytest_argv` equal to the ordered de-duplicated union of each profile's verification selectors plus only the frozen collection flags;
- exact Slack ten with no duplicate;
- exact helper-contract module plus eight PostgreSQL-critical consumer modules;
- selector paths confined to `backend/tests`;
- unique verification-child partitions;
- `settings-diagnostic=120` seconds and every other profile `1200` seconds;
- rejection of unknown fields, unknown profiles/plugins, shell metacharacter strings, duplicate child ids, duplicate selectors, or a changed baseline;
- frozen safe-parameter case counts `2, 6, 5, 4, 2, 2, 3, 2` for the eight functions in Task 7.

Run RED:

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_contracts.py -q
```

Expected RED: import/fixture failures because the contract module and manifest do not exist.

- [ ] **Step 2: Implement strict immutable parsing**

Use dataclasses from the frozen contract above. Parse JSON into typed immutable tuples, reject extra/missing keys, validate every identifier with anchored ASCII patterns, and compute invocation hashes from canonical UTF-8 JSON containing only profile, child id, kind, argv, timeout, manifest hash, and commit SHA.

Do not include environment values, paths outside the repository-relative selectors, or resource identities in the hash payload.

- [ ] **Step 3: Run GREEN and static checks**

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_contracts.py -q
uv run --no-cache --locked ruff check backend/tests/release_contracts.py backend/tests/test_release_contracts.py
uv run --no-cache --locked python -m compileall -q backend/tests/release_contracts.py
git diff --check
```

Expected: all selected tests pass; Ruff, compile, and diff checks are silent.

- [ ] **Step 4: Commit the contract slice**

```powershell
git add backend/tests/release_contracts.py backend/tests/test_release_contracts.py backend/tests/fixtures/deferred_backend_baseline_v1.json
git commit -m "test: freeze backend release contracts"
```

## Task 2: Add the Guarded Test Bootstrap

**Files:**

- Create: `backend/tests/release_bootstrap.py`
- Create: `backend/tests/test_release_bootstrap.py`

- [ ] **Step 1: Write failing bootstrap tests**

Add:

- `test_apply_test_environment_guard_refuses_prepopulated_settings_cache`
- `test_guarded_probe_ignores_adverse_dotenv_and_parent_credentials`
- `test_guarded_sqlite_init_uses_only_controller_file_url`
- `test_guarded_bootstrap_reports_only_allowlisted_effective_state`
- `test_guarded_bootstrap_rejects_xdist_before_conftest_import`
- `test_guard_module_has_no_top_level_application_import`

Subprocess fixtures create an adverse temporary `.env` and inherited fake provider/rollout values. Assertions inspect only the bounded state DTO, never the values.

Run RED:

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_bootstrap.py -q
```

Expected RED: the module/entry points are missing.

- [ ] **Step 2: Implement the late-import guard**

`backend/tests/release_bootstrap.py` exposes:

```python
class ReleaseBootstrapRefused(RuntimeError):
    code = 'environment_refused'

@dataclass(frozen=True)
class EffectiveBootstrapState:
    resolved_database_backend: Literal['sqlite']
    dotenv_source_enabled: Literal[False]
    demo_mode: Literal[True]
    auto_review_mode: Literal['disabled']
    agent_llm_enabled: Literal[False]
    live_provider_credentials_present: Literal[False]
    paid_provider_flags_present: Literal[False]
    pytest_worker_count: Literal[1]

def apply_test_environment_guard() -> None: ...
def probe_effective_state() -> EffectiveBootstrapState: ...
def initialize_guarded_sqlite() -> None: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

The module has no top-level `backend.app` import. `apply_test_environment_guard()` rejects non-empty `get_settings` cache, changes `Settings.model_config['env_file']` to `None` process-locally, verifies it, and clears the cache before any `db.session` or `init_db` import. `probe` and `init-sqlite` call the guard first. Output is one bounded JSON object; stderr is empty.

- [ ] **Step 3: Run GREEN and static checks**

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_bootstrap.py -q
uv run --no-cache --locked ruff check backend/tests/release_bootstrap.py backend/tests/test_release_bootstrap.py
uv run --no-cache --locked python -m compileall -q backend/tests/release_bootstrap.py
git diff --check
```

- [ ] **Step 4: Commit the bootstrap slice**

```powershell
git add backend/tests/release_bootstrap.py backend/tests/test_release_bootstrap.py
git commit -m "test: add guarded release bootstrap"
```

## Task 3: Add Privacy-Safe Pytest Evidence

**Files:**

- Create: `backend/tests/release_evidence_plugin.py`
- Create: `backend/tests/test_release_evidence_plugin.py`

- [ ] **Step 1: Write failing plugin tests**

Add:

- `test_plugin_applies_dotenv_guard_before_conftest_import`
- `test_plugin_writes_complete_sidecar_only_at_sessionfinish`
- `test_plugin_refuses_unsafe_collected_or_reported_nodeid_before_persistence`
- `test_plugin_uses_unique_partial_and_refuses_preexisting_final_target`
- `test_plugin_hashes_lease_identity_and_rejects_invalid_lifecycle`
- `test_plugin_excludes_traceback_output_paths_urls_and_resource_names`
- `test_plugin_refuses_xdist_and_unlisted_plugins`
- `test_completed_sidecar_reader_rejects_identity_hash_exit_or_completion_mismatch`
- `test_plugin_maps_only_the_exact_review_v2_guard_to_allowlisted_reason_without_persisting_longrepr`

Run RED:

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_evidence_plugin.py -q
```

Expected RED: plugin hooks and sidecar reader are absent.

- [ ] **Step 2: Implement the sidecar state machine**

Import `ReleaseEvent`, `LeaseLifecycleEvent`, `ReleaseEvidenceSidecar`, and `ReleaseInvocationIdentity` from `backend.tests.release_contracts`. Expose:

```python
SIDECAR_SCHEMA_VERSION = 1

def validate_safe_nodeid(nodeid: str) -> None: ...
def record_lease_event(
    *, lease_identity: str, state: Literal['created', 'dropped']
) -> None: ...
def read_completed_sidecar(path: Path) -> ReleaseEvidenceSidecar: ...
def validate_completed_sidecar(
    sidecar: ReleaseEvidenceSidecar,
    *,
    expected_identity: ReleaseInvocationIdentity,
    native_exit_code: int,
) -> None: ...
```

Load `apply_test_environment_guard()` from `pytest_load_initial_conftests`. Validate every collection/report node id before keeping it. Keep events only in process memory and a private unique partial path; at `pytest_sessionfinish` add pytest's `exitstatus`, collection/event hashes, and `complete=true`, fsync, then atomically rename to a final path that did not exist. Only the controller knows the subprocess native return code; `validate_completed_sidecar` compares that return code with `pytest_exitstatus` and refuses disagreement. Never serialize captured output, traceback, exception text, DSN, filesystem path, resource name, or environment value.

For the Task 10 diagnostic only, inspect `report.longrepr` in memory after an exact node-id/call-phase match. Map the exact PostgreSQL text `post-C.5 workflow ReviewItem requires exact provenance` to `review_v2_post_c5_provenance_guard`; every other failure stores `None`. Discard `longrepr` immediately and prove its text and synthetic sensitive fixture markers are absent from the sidecar and aggregate report.

`record_lease_event` is active only between `pytest_configure` and `pytest_unconfigure`, hashes the lease identity immediately, and enforces one `created -> dropped` sequence. Outside a release child it is a no-op so ordinary focused tests remain usable.

- [ ] **Step 3: Run GREEN and static checks**

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_evidence_plugin.py -q
uv run --no-cache --locked ruff check backend/tests/release_evidence_plugin.py backend/tests/test_release_evidence_plugin.py
uv run --no-cache --locked python -m compileall -q backend/tests/release_evidence_plugin.py
git diff --check
```

- [ ] **Step 4: Commit the evidence slice**

```powershell
git add backend/tests/release_evidence_plugin.py backend/tests/test_release_evidence_plugin.py
git commit -m "test: add privacy-safe release evidence"
```

## Task 4: Add the PostgreSQL Schema-Lease Boundary

**Files:**

- Create: `backend/tests/postgres_isolation.py`
- Create: `backend/tests/test_postgres_isolation.py`

- [ ] **Step 1: Write failing lease tests**

Cover:

- malformed/non-PostgreSQL URL rejection before connecting;
- parsed and live database/user equality, `_test` suffix, controller-latched expected identity, and known non-test denylist;
- generated lowercase name, sanitized scope, literal run prefix, uniqueness, and 63-byte bound;
- refusal to adopt or drop a pre-existing exact schema;
- preservation of unrelated URL query keys while replacing only `options` with encoded `-csearch_path=<schema>,public`;
- ownership latching only after successful `CREATE SCHEMA`;
- exact create/drop event order with hashed identity;
- one cleanup attempt, bounded `lock_timeout=5s` and `statement_timeout=30s`, and cleanup-failure precedence;
- public-table preflight once per pytest release session;
- environment snapshot/restore and `get_settings.cache_clear()` before/after Alembic;
- two leases cannot see each other's marker, Alembic version, checkpoint, or row.

Use fake connection/admin adapters for mutation-order and failure tests. Mark the two-live-lease test as release-critical and run it through the controller later; it may skip only in the local unit command when no locator exists.

Run RED:

```powershell
uv run --no-cache --locked pytest backend/tests/test_postgres_isolation.py -q
```

Expected RED: lease helpers do not exist.

- [ ] **Step 2: Implement the exact helper contract**

```python
@dataclass(frozen=True)
class PostgresSchemaLease:
    base_url: str
    database_url: str
    schema_name: str

@contextmanager
def lease_postgres_schema(
    base_url: str,
    *,
    run_id: str,
    scope_name: str,
) -> Iterator[PostgresSchemaLease]: ...

@contextmanager
def leased_database_environment(database_url: str) -> Iterator[None]: ...
```

Read `PARAWORKS_RELEASE_EXPECTED_DATABASE`, `PARAWORKS_RELEASE_EXPECTED_ROLE`, and `PARAWORKS_RELEASE_RUN_ID` as controller latches; do not infer ownership from suffix alone. Use generated identifiers only, quote them through the SQLAlchemy dialect, and never include the URL or driver exception in the public error.

`leased_database_environment` snapshots exactly `PARAWORKS_DEMO_MODE`, `PARAWORKS_DATABASE_URL`, and `DATABASE_URL`; sets false/leased/leased; clears Settings before and after; and restores absent versus present values exactly.

Immediately after `CREATE SCHEMA` succeeds and ownership is latched, call `record_lease_event(lease_identity=lease_identity, state='created')`. On exit, close the consumer first, drop only the quoted exact generated schema with `CASCADE`, verify exact `pg_namespace` absence on the same bounded admin connection, and only then call `record_lease_event(lease_identity=lease_identity, state='dropped')`. A failed create emits no event; a failed or unverified drop emits no `dropped` event and raises the cleanup failure. Outside a release child the recorder remains the tested no-op.

- [ ] **Step 3: Run local GREEN and static checks**

```powershell
uv run --no-cache --locked pytest backend/tests/test_postgres_isolation.py -q
uv run --no-cache --locked ruff check backend/tests/postgres_isolation.py backend/tests/test_postgres_isolation.py
uv run --no-cache --locked python -m compileall -q backend/tests/postgres_isolation.py
git diff --check
```

Expected: fake/unit cases pass. The real two-lease case is not accepted as release evidence until Task 11 runs `--profile postgres` with zero skips.

- [ ] **Step 4: Commit the lease slice**

```powershell
git add backend/tests/postgres_isolation.py backend/tests/test_postgres_isolation.py
git commit -m "test: add postgres schema leases"
```

## Task 5: Add Controller Preflight and Owned Resource Lifecycle

**Files:**

- Create: `scripts/backend_release_matrix.py`
- Create: `backend/tests/test_backend_release_matrix.py`

- [ ] **Step 1: Write failing preflight/resource tests**

Add:

- `test_controller_has_no_application_settings_session_or_init_import`
- `test_preflight_refuses_missing_unhealthy_or_mismatched_container_without_mutation`
- `test_preflight_refuses_wrong_image_service_port_or_server_identity`
- `test_preflight_refuses_preexisting_exact_or_run_prefix_without_adoption`
- `test_generated_database_and_role_are_lower_hex_bounded_and_end_test`
- `test_ownership_latches_only_after_each_successful_create`
- `test_controller_creates_vector_extension_public_preflight_and_sentinel_in_order`
- `test_controller_uses_argument_vectors_and_never_shell_interpolation`
- `test_secure_artifact_targets_are_unique_private_and_nonpreexisting`

Fake command/admin logs must prove that refusal paths issue no create, terminate, drop, container, or volume mutation.

Run RED:

```powershell
uv run --no-cache --locked pytest backend/tests/test_backend_release_matrix.py -q -k "controller or preflight or resource or ownership or artifact"
```

- [ ] **Step 2: Implement bounded resource ownership**

Use these controller-owned types:

```python
@dataclass
class OwnedPostgresResources:
    run_id: str
    role_name: str | None = None
    database_name: str | None = None
    sentinel_schema_name: str | None = None
    role_owned: bool = False
    database_owned: bool = False
    sentinel_owned: bool = False

@dataclass(frozen=True)
class ReleaseOutcome:
    code: ReleaseOutcomeCode
    release_proof: bool
    counts: Mapping[str, int]
    unexpected_nodeids: tuple[str, ...]
```

Generate one 12-character lowercase-hex run id. Validate generated identifiers before every SQL statement. Use the literal local admin service contract internally, generate a random role password, and keep both admin/test DSNs out of exceptions and output. Validate the Docker inspect/health/port/Compose labels and `SELECT current_database(), current_user` before any mutation.

Create role, database, `vector` extension, public preflight, and sentinel in that order. Latch each ownership bit only after its statement succeeds.

- [ ] **Step 3: Implement minimal always-run cleanup**

In an outer `try/finally`:

1. terminate only the controller-created child process group and confirm exit;
2. open one bounded inspection connection and latch its pid;
3. disable new exact-database connections and terminate all other exact-database sessions;
4. verify lease-prefix count zero and sentinel present;
5. close inspection;
6. drop exact owned database;
7. drop exact owned role;
8. verify exact and run-prefix database/role counts `0:0:0:0`;
9. delete only exact controller-created SQLite, JUnit, collection, sidecar, and partial paths.

Continue cleanup after any individual failure. Preserve the first cleanup failure as the final `cleanup_failed` outcome.

- [ ] **Step 4: Run GREEN and static checks**

```powershell
uv run --no-cache --locked pytest backend/tests/test_backend_release_matrix.py -q -k "controller or preflight or resource or ownership or artifact"
uv run --no-cache --locked ruff check scripts/backend_release_matrix.py backend/tests/test_backend_release_matrix.py
uv run --no-cache --locked python -m compileall -q scripts/backend_release_matrix.py
git diff --check
```

- [ ] **Step 5: Commit the resource slice**

```powershell
git add scripts/backend_release_matrix.py backend/tests/test_backend_release_matrix.py
git commit -m "test: add release resource ownership"
```

## Task 6: Add Hermetic Children, Profiles, and Coverage Authority

**Files:**

- Modify: `scripts/backend_release_matrix.py`
- Modify: `backend/tests/test_backend_release_matrix.py`
- Modify: `backend/tests/fixtures/deferred_backend_baseline_v1.json`

- [ ] **Step 1: Write failing environment/profile tests**

Add:

- `test_child_environment_is_a_literal_allowlist`
- `test_child_environment_removes_settings_provider_paid_trace_and_pytest_controls`
- `test_settings_profile_runs_guarded_probe_then_sqlite_init_then_pytest`
- `test_controller_refuses_missing_duplicate_unexpected_or_partial_sidecars`
- `test_controller_refuses_identity_hash_completion_or_native_exit_mismatch`
- `test_multichild_profile_requires_exact_child_set_disjoint_nodes_and_union`
- `test_non_slack_selection_is_canonical_minus_exact_slack_ten`
- `test_full_accepts_exit_one_only_for_exact_slack_failure_set`
- `test_child_id_run_sets_release_proof_false_and_accepts_manifest_ids_only`
- `test_controller_deletes_raw_child_output_and_retains_only_aggregate_fields`

Run RED:

```powershell
uv run --no-cache --locked pytest backend/tests/test_backend_release_matrix.py -q -k "environment or profile or sidecar or multichild or baseline or child_id"
```

- [ ] **Step 2: Implement child execution**

Every pytest child command is an argument vector:

```python
(
    sys.executable,
    '-m',
    'pytest',
    *child.pytest_argv,
    '-p',
    'backend.tests.release_evidence_plugin',
)
```

Before pytest, run:

```text
python -m backend.tests.release_bootstrap probe
python -m backend.tests.release_bootstrap init-sqlite
```

Construct a new allowlisted environment for every subprocess. Use unique sidecar/JUnit/SQLite paths, one process group, bounded child/profile timeout, and `shell=False`. A canonical collection child runs the same selectors with `--collect-only` and the same plugin. Verification children must be pairwise disjoint and their union must equal the canonical collection exactly.

`--child-id` runs only one manifest child through the same full preflight/resource/environment/evidence/cleanup lifecycle. Reject a missing/unknown child, set `release_proof=false`, and never let its result update release documentation.

- [ ] **Step 3: Implement aggregate decision rules**

The controller prints one JSON line containing only:

```text
schema_version, commit_sha, profile, outcome, release_proof,
collected_count, selected_count, passed_count, failed_count,
error_count, skipped_count, xfailed_count, deselected_count,
unexpected_nodeids, lease_created_count, lease_dropped_count,
failure_reason_counts,
expected_sidecar_count, observed_sidecar_count,
collection_hash, event_hash, pre_drop_lease_schema_count,
sentinel_present, database_owned, role_owned,
exact_database_count, exact_role_count,
prefix_database_count, prefix_role_count,
no_live_provider
```

Never print raw subprocess output. Validate the exact sidecar identity set, completion/native-exit agreement, hashed lease state machines, collection partition, skip/xfail policy, and baseline before choosing an outcome.

- [ ] **Step 4: Run GREEN and full local unit verification**

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_contracts.py backend/tests/test_release_bootstrap.py backend/tests/test_release_evidence_plugin.py backend/tests/test_postgres_isolation.py backend/tests/test_backend_release_matrix.py -q
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-isolation-contract
uv run --no-cache --locked ruff check backend/tests/release_contracts.py backend/tests/release_bootstrap.py backend/tests/release_evidence_plugin.py backend/tests/postgres_isolation.py backend/tests/test_release_contracts.py backend/tests/test_release_bootstrap.py backend/tests/test_release_evidence_plugin.py backend/tests/test_postgres_isolation.py backend/tests/test_backend_release_matrix.py scripts/backend_release_matrix.py
uv run --no-cache --locked python -m compileall -q backend/tests/release_contracts.py backend/tests/release_bootstrap.py backend/tests/release_evidence_plugin.py backend/tests/postgres_isolation.py scripts/backend_release_matrix.py
git diff --check
```

Expected: all local unit cases pass, and the focused real helper child passes with zero skips, balanced lease events, `release_proof=false`, sentinel survival, and cleanup `0:0:0:0`.

- [ ] **Step 5: Commit the profile slice**

```powershell
git add scripts/backend_release_matrix.py backend/tests/test_backend_release_matrix.py backend/tests/fixtures/deferred_backend_baseline_v1.json
git commit -m "test: add backend release profiles"
```

## Task 7: Observe Collection Privacy RED, Then Add Stable Safe IDs

**Files:**

- Modify: `backend/tests/test_agent_runtime_checkpointing.py`
- Modify: `backend/tests/test_agent_runtime_postgres_checkpoint.py`
- Modify: `backend/tests/test_agent_runtime_state.py`
- Modify: `backend/tests/test_auto_review_contracts.py`
- Modify: `backend/tests/test_auto_review_provenance.py`
- Modify: `backend/tests/test_database_initialization.py`
- Modify: `backend/tests/test_review_transitions.py`
- Modify: `backend/tests/test_release_contracts.py`

- [ ] **Step 1: Run the authoritative collection RED before changing ids**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile full --child-id full-collection
```

Expected RED: `evidence_refused`, `release_proof=false`, no final sidecar containing an unsafe parameter value, full outer cleanup `0:0:0:0`, and exactly the eight safe test-function names listed below. If the observed family set differs in either direction, stop and amend this plan rather than silently broadening or pre-emptively changing ids.

- [ ] **Step 2: Add only these explicit id arrays**

| Function | Exact ids in case order |
|---|---|
| `test_memory_runtime_owns_one_strict_saver_per_runtime` | `demo-postgres`, `local-sqlite` |
| `test_unsafe_database_identity_is_rejected_before_any_mutation` | `non-test-postgres-database`, `non-test-paraworks-database`, `non-test-development-database`, `non-test-production-database`, `non-test-postgres-role`, `non-test-paraworks-role` |
| `test_checkpoint_state_rejects_sensitive_or_non_json_values` | `question`, `source-url`, `model-output`, `review-item-ids`, `session-object` |
| `test_process_local_sqlite_smoke_is_limited_to_disabled_in_memory_url` | `postgres-disabled`, `sqlite-file-disabled`, `sqlite-memory-disabled`, `sqlite-memory-shadow` |
| `test_key_admin_module_cli_bounds_storage_initialization_failure` | `malformed-url`, `unknown-dialect` |
| `test_configuration_failures_are_typed_and_sanitized` | `malformed-url`, `missing-dialect` |
| `test_session_adapter_forwards_the_resolved_database_url_in_a_fresh_process` | `demo-precedence`, `paraworks-precedence`, `fallback-precedence` |
| `test_transition_rechecks_evidence_before_approval` | `missing-link`, `blank-snippet` |

Do not change parameters, assertions, function names, case order, or Slack ids. Extend `test_release_contracts.py` so collected `(function, case-count)` remains `2,6,5,4,2,2,3,2` and every emitted node id passes the plugin validator.

- [ ] **Step 3: Run collection GREEN and Settings diagnostic**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile full --child-id full-collection
uv run --locked python scripts/backend_release_matrix.py --profile settings-diagnostic
```

Expected: the collection-only run passes with `release_proof=false` and exact case parity; `settings-diagnostic` reports `6 passed`, zero skip/xfail/error, `release_proof=true`, and cleanup `0:0:0:0`.

- [ ] **Step 4: Run touched tests and lint**

```powershell
uv run --no-cache --locked pytest backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_postgres_checkpoint.py::test_unsafe_database_identity_is_rejected_before_any_mutation backend/tests/test_agent_runtime_state.py backend/tests/test_auto_review_contracts.py backend/tests/test_auto_review_provenance.py::test_key_admin_module_cli_bounds_storage_initialization_failure backend/tests/test_database_initialization.py backend/tests/test_review_transitions.py backend/tests/test_release_contracts.py -q
uv run --no-cache --locked ruff check backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_postgres_checkpoint.py backend/tests/test_agent_runtime_state.py backend/tests/test_auto_review_contracts.py backend/tests/test_auto_review_provenance.py backend/tests/test_database_initialization.py backend/tests/test_review_transitions.py backend/tests/test_release_contracts.py
git diff --check
```

- [ ] **Step 5: Commit the safe-id slice**

```powershell
git add backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_postgres_checkpoint.py backend/tests/test_agent_runtime_state.py backend/tests/test_auto_review_contracts.py backend/tests/test_auto_review_provenance.py backend/tests/test_database_initialization.py backend/tests/test_review_transitions.py backend/tests/test_release_contracts.py
git commit -m "test: sanitize release collection ids"
```

## Task 8: Convert Module-Scoped PostgreSQL Consumers

**Files:**

- Modify: `backend/tests/test_agent_runtime_postgres_checkpoint.py`
- Modify: `backend/tests/test_auto_review_migration.py`
- Modify: `backend/tests/test_auto_review_provenance.py`

- [ ] **Step 1: Add failing fixture-contract tests before each conversion**

For each module, prove through a focused test that:

- its engine/Alembic/checkpointer URL contains the exact leased schema;
- `current_schema()` and `alembic_version` resolve inside that schema;
- the base `public` schema is not used for application tables;
- every engine, pool, saver, and session is closed before lease exit;
- module restart/reuse behavior remains inside one lease;
- the lease drop event occurs after consumer disposal.

Run each RED through controller-owned resources:

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-agent-runtime-checkpoint
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-auto-review-migration
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-auto-review-provenance
```

Expected RED: each child reaches the new assertion but the existing module still uses the base/public path; every run remains `release_proof=false` and outer cleanup still reaches `0:0:0:0`.

- [ ] **Step 2: Rewire the three module fixtures**

- `test_agent_runtime_postgres_checkpoint.py`: acquire one module lease, keep restart assertions within it, and close every `CheckpointRuntime` pool/saver and engine before context exit.
- `test_auto_review_migration.py`: replace the public-table freshness guard, manual schema, and manual `alembic_version` creation with one module lease. Wrap every individual Alembic upgrade/downgrade invocation in `leased_database_environment(lease.database_url)` so `migrations/env.py` resolves the leased URL, clears Settings before/after each invocation, and restores process environment immediately afterward. `_postgres_engine` must return an engine bound to `lease.database_url`, assert its exact `current_schema()`, and dispose it before module-lease exit; do not leave `PARAWORKS_DATABASE_URL` set for the module lifetime.
- `test_auto_review_provenance.py`: acquire one module lease, keep the strong test-only fingerprint identity scoped inside it, use `leased_database_environment` for Alembic, and dispose its engine before lease exit.

Do not change production assertions or skip behavior.

- [ ] **Step 3: Run each module through controller-owned resources**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-agent-runtime-checkpoint
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-auto-review-migration
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-auto-review-provenance
```

Expected: each focused run passes with zero skips, `release_proof=false`, balanced lease create/drop counts, pre-drop lease-prefix zero, sentinel present, and final `0:0:0:0`.

- [ ] **Step 4: Run lint/diff and commit**

```powershell
uv run --no-cache --locked ruff check backend/tests/test_agent_runtime_postgres_checkpoint.py backend/tests/test_auto_review_migration.py backend/tests/test_auto_review_provenance.py
git diff --check
git add backend/tests/test_agent_runtime_postgres_checkpoint.py backend/tests/test_auto_review_migration.py backend/tests/test_auto_review_provenance.py
git commit -m "test: isolate module postgres fixtures"
```

## Task 9: Convert Function-Scoped PostgreSQL Consumers

**Files:**

- Modify: `backend/tests/test_review_transition_postgres.py`
- Modify: `backend/tests/test_review_v21_extraction_postgres.py`
- Modify: `backend/tests/test_pgvector_integration.py`
- Modify: `backend/tests/test_auto_review_postgres.py`

- [ ] **Step 1: Add failing lease assertions**

For every fixture, assert exact `current_schema()`, lease-owned tables, engine/session closure before drop, and one create/drop pair per test. For `pgvector`, remove the collection-time `skipif` entirely. Its runtime fixture calls `pytest.fail` when `PARAWORKS_RELEASE_CHILD_ID` is present but `PARAWORKS_PGVECTOR_TEST_DATABASE_URL` is absent; it may call `pytest.skip` only when both release identity and the optional local locator are absent.

Run each RED through controller-owned resources:

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-review-transition
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-review-v21-extraction
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-pgvector
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-auto-review-c5
```

Expected RED: manual/base schema paths do not satisfy the new assertions; every run remains `release_proof=false` and exact outer cleanup still completes.

- [ ] **Step 2: Rewire each fixture**

- `test_review_transition_postgres.py`: one function lease per race test; all concurrency participants share that lease; remove row-by-row database cleanup that exists only to compensate for sharing, while preserving domain assertions.
- `test_review_v21_extraction_postgres.py`: replace manual schema and manual `alembic_version` with a function lease; run migrations through `leased_database_environment`.
- `test_pgvector_integration.py`: implement the exact release/local locator branch above, use a function lease, create/search/drop only inside it, close the session and engine before lease exit, and require zero skip in release mode.
- `test_auto_review_postgres.py`: use a function lease unless one test's restart/concurrency actors require sharing; those actors share one test lease, never a module/global schema.

- [ ] **Step 3: Run focused controller children**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-review-transition
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-review-v21-extraction
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-pgvector
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-auto-review-c5
```

Expected: each child passes with zero skip and clean lease/resource evidence. If `test_auto_review_postgres.py` is absent after Tasks 6–15, stop because Task 16 entry prerequisites are incomplete.

- [ ] **Step 4: Run lint/diff and commit**

```powershell
uv run --no-cache --locked ruff check backend/tests/test_review_transition_postgres.py backend/tests/test_review_v21_extraction_postgres.py backend/tests/test_pgvector_integration.py backend/tests/test_auto_review_postgres.py
git diff --check
git add backend/tests/test_review_transition_postgres.py backend/tests/test_review_v21_extraction_postgres.py backend/tests/test_pgvector_integration.py backend/tests/test_auto_review_postgres.py
git commit -m "test: isolate function postgres fixtures"
```

## Task 10: Observe Review V2 RED and Conditionally Correct Only Its Fake Provenance

**Files:**

- Modify: `backend/tests/test_review_v2_postgres.py`

- [ ] **Step 1: Convert only the database fixture to a function lease**

Use `lease_postgres_schema` and `leased_database_environment`. Rewrite `_PostgresHarness.cleanup()` so it closes every app's `CheckpointRuntime`, saver/pool, application engine, open session, and harness engine only. Remove its ReviewItem/effect/source/workflow/checkpoint row-deletion SQL entirely; the function lease owns those rows and the helper's quoted exact-schema `DROP SCHEMA <owned-schema> CASCADE` is the sole data cleanup. This avoids deleting an immutable `ReviewItemEvidenceRef` or its parent in an invalid order.

Replace the existing single-case `test_cleanup_runs_checkpoint_and_dispose_after_application_cleanup_failure` with `test_cleanup_disposes_applications_and_engine_without_row_deletes`. Its fake `session_factory` must raise if called; expected events are each app `close` in reverse order followed by one harness-engine `dispose`, with no row/checkpoint delete connection. The controller sidecar for each function fixture must record exactly one `created -> dropped` lifecycle after catalog-confirmed schema absence. Do not add a tenth module test, and do not yet change `_DeterministicDraftService.draft()` or any business/restart/privacy assertion.

- [ ] **Step 2: Run and inspect the clean long-trace RED**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-review-v2
```

Expected pre-fix evidence: exactly `6 failed, 3 passed, 0 skipped`. The failed sidecar events must be the call phase for exactly:

```text
backend/tests/test_review_v2_postgres.py::test_postgres_interrupt_survives_pool_and_app_restart_then_resumes_same_thread
backend/tests/test_review_v2_postgres.py::test_postgres_saved_tuple_contains_no_source_or_model_content
backend/tests/test_review_v2_postgres.py::test_checkpoint_commit_then_status_failure_reconciles_without_duplicate_effects
backend/tests/test_review_v2_postgres.py::test_draft_commit_then_checkpoint_failure_repairs_without_duplicate_effects
backend/tests/test_review_v2_postgres.py::test_concurrent_exact_batch_launches_create_one_thread
backend/tests/test_review_v2_postgres.py::test_concurrent_resume_has_one_state_version_winner
```

The controller must report `failure_reason_counts={'review_v2_post_c5_provenance_guard': 6}` from the allowlisted in-memory classifier, then delete the private long traceback. The sidecar/report must contain neither that traceback nor any synthetic sensitive fixture value.

If the failing node set, phase, or cause differs, do not edit the fake draft. Restore only the uncommitted fixture attempt if necessary, document the observed safe failure classification, and return to the owning contract for a separately reviewed plan.

- [ ] **Step 3: Add the minimum real C.5 fake shape only after confirmation**

Import `ReviewItemEvidenceRef` and update the fake transaction to:

```python
canonical_workflow_evidence_ref = db.scalars(
    select(AgentWorkflowEvidenceRef)
    .where(
        AgentWorkflowEvidenceRef.workflow_thread_id
        == workflow_thread_id,
        AgentWorkflowEvidenceRef.ordinal == 1,
    )
    .order_by(AgentWorkflowEvidenceRef.id)
).one()
run = AgentRun(
    agent_name='history_agent',
    prompt_version='postgres-deterministic-v1',
    status='complete',
    source_window='postgres-release-window',
    cache_key=f'postgres-cache:{workflow_thread_id}',
    model_name='deterministic-test-adapter',
    workflow_thread_id=workflow_thread_id,
    generation_provider='deterministic-test',
    generation_reasoning_effort='none',
    generation_route_version='postgres-review-v2-fixture:v1',
    generation_output_contract_version='history-candidate:c5-v1',
    input_tokens=64,
    output_tokens=32,
    total_tokens=96,
    estimated_cost_usd=0.0,
    permission_level='internal',
    metadata_={'adapter': 'deterministic'},
    effect_key=f'postgres-effect:{workflow_thread_id}',
    completed_at=datetime.now(UTC),
)
db.add(run)
db.flush()
item = ReviewItem(
    id=int(uuid4().hex[:7], 16) + 100_000_000,
    item_type='history_event',
    payload={
        'title': 'PostgreSQL recovery candidate',
        'reason': 'Durable review recovery must be proven.',
        'agent_name': 'history_agent',
        'agent_run_id': run.id,
        'source_ids': ['gmail:postgres-sensitive-source'],
        'prompt': 'postgres-sensitive-provider-prompt',
        'model_output': 'postgres-sensitive-model-output',
        'provider_error': 'postgres-sensitive-provider-error',
        'api_key': 'postgres-sensitive-api-key',
    },
    source_links=['https://postgres-sensitive.invalid/source'],
    source_snippets=['postgres-sensitive-source-snippet'],
    confidence_score=0.93,
    permission_level='internal',
    status='pending_review',
    workflow_thread_id=workflow_thread_id,
    candidate_key=f'postgres-candidate:{workflow_thread_id}',
    agent_run_id=run.id,
    candidate_contract_version='c5-v1',
)
db.add(item)
db.flush()
db.add(
    ReviewItemEvidenceRef(
        review_item_id=item.id,
        workflow_thread_id=workflow_thread_id,
        workflow_evidence_ref_id=canonical_workflow_evidence_ref.id,
        candidate_slot_ordinal=1,
        message_content_fingerprint='e' * 64,
        fingerprint_key_version='postgres-test-v1',
        fingerprint_key_material_verifier='f' * 64,
    )
)
db.commit()
db.refresh(item)
```

Perform that exact query/add/flush/add/flush/add/commit order in one transaction so the deferred provenance trigger observes the parent, run, and child together. Do not create a second source identity or rely on the payload's legacy `agent_run_id`. Keep sensitive fake payload fields so checkpoint-privacy assertions remain meaningful.

- [ ] **Step 4: Run GREEN**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile postgres --child-id postgres-review-v2
```

Expected: `9 passed, 0 skipped`. The focused report has `release_proof=false`, one verified schema drop per fixture instance, balanced lease lifecycle, sentinel present, and final cleanup `0:0:0:0`. Existing restart, privacy, repair, exact-batch, concurrent-resume, and race assertions are unchanged.

- [ ] **Step 5: Lint/diff and commit**

```powershell
uv run --no-cache --locked ruff check backend/tests/test_review_v2_postgres.py
git diff --check
git add backend/tests/test_review_v2_postgres.py
git commit -m "test: align review v2 postgres evidence"
```

## Task 11: Harden Failure, Timeout, and Cleanup Precedence

**Files:**

- Modify: `scripts/backend_release_matrix.py`
- Modify: `backend/tests/test_backend_release_matrix.py`
- Modify: `backend/tests/test_postgres_isolation.py`
- Modify: `backend/tests/test_release_evidence_plugin.py`

- [ ] **Step 1: Write adverse-path RED tests**

Add:

- `test_timeout_terminates_only_controller_process_group`
- `test_cleanup_quiesces_exact_database_before_catalog_inspection`
- `test_cleanup_requires_zero_lease_prefix_and_live_sentinel_before_drop`
- `test_cleanup_continues_database_role_and_artifacts_after_prior_failure`
- `test_cleanup_failure_overrides_verification_and_baseline_outcomes`
- `test_postflight_requires_exact_and_prefix_counts_zero`
- `test_cleanup_never_adopts_preexisting_database_role_schema_or_container`
- `test_lease_cleanup_failure_overrides_setup_or_test_failure`
- `test_duplicate_create_drop_or_missing_drop_event_is_evidence_refused`
- `test_adverse_module_order_has_no_setup_error`

Run RED:

```powershell
uv run --no-cache --locked pytest backend/tests/test_backend_release_matrix.py backend/tests/test_postgres_isolation.py backend/tests/test_release_evidence_plugin.py -q -k "timeout or cleanup or postflight or adverse or duplicate"
```

- [ ] **Step 2: Implement bounded precedence and order permutations**

Apply connect, lock, statement, process, and profile deadlines at every blocking boundary. On timeout, terminate only the latched process group. Continue exact cleanup even if termination cannot be confirmed, but refuse to call catalog state clean. Never retry/adopt a residual identity automatically.

Add a manifest-driven adverse PostgreSQL ordering test that runs:

1. checkpoint -> migration -> V2.1 extraction -> Review V2;
2. Review V2 -> provenance -> migration -> pgvector;
3. reverse of all PostgreSQL verification children.

Each permutation must produce the same selected node set, zero setup errors/skips, balanced leases, live sentinel before outer drop, and `0:0:0:0`.

- [ ] **Step 3: Run GREEN and static gates**

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_contracts.py backend/tests/test_release_bootstrap.py backend/tests/test_release_evidence_plugin.py backend/tests/test_postgres_isolation.py backend/tests/test_backend_release_matrix.py -q
uv run --no-cache --locked ruff check backend/tests/release_contracts.py backend/tests/release_bootstrap.py backend/tests/release_evidence_plugin.py backend/tests/postgres_isolation.py backend/tests/test_release_contracts.py backend/tests/test_release_bootstrap.py backend/tests/test_release_evidence_plugin.py backend/tests/test_postgres_isolation.py backend/tests/test_backend_release_matrix.py scripts/backend_release_matrix.py
uv run --no-cache --locked python -m compileall -q backend/tests/release_contracts.py backend/tests/release_bootstrap.py backend/tests/release_evidence_plugin.py backend/tests/postgres_isolation.py scripts/backend_release_matrix.py
uv lock --check
git diff --check
```

- [ ] **Step 4: Commit the hardening slice**

```powershell
git add scripts/backend_release_matrix.py backend/tests/test_backend_release_matrix.py backend/tests/test_postgres_isolation.py backend/tests/test_release_evidence_plugin.py
git commit -m "test: harden backend release cleanup"
```

## Task 12: Run C.5 Task 16 Steps 4–7 Official Profiles and Record Only Observed Evidence

Execution note (2026-08-30): the isolation implementation and official profiles
are complete at behavior commit `4b9132a`. The earlier task-level micro-commit
checkboxes above are retained as the approved plan history; the authoritative
observed results and operational procedure are in
`docs/superpowers/runbooks/backend-release-matrix.md`.

**Files:**

- Create: `docs/superpowers/runbooks/backend-release-matrix.md`
- Modify: `plan.md`
- Modify: `docs/portfolio-log.md`
- Modify: `docs/superpowers/runbooks/session-handoff.md`

- [x] **Step 1: Verify the implementation tree and shared service before release**

```powershell
git status --short
uv lock --check
git diff --check
```

Expected: clean tracked worktree and a healthy externally provided shared container matching the exact approved contract. If the container is absent/mismatched, stop with `preflight_refused`; do not start or replace it.

- [x] **Step 2: Run the official controller profiles**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile settings-diagnostic
uv run --locked python scripts/backend_release_matrix.py --profile postgres
uv run --locked python scripts/backend_release_matrix.py --profile compatibility
uv run --locked python scripts/backend_release_matrix.py --profile non-slack
uv run --locked python scripts/backend_release_matrix.py --profile full
```

Required results:

- `settings-diagnostic`: exact six selected nodes pass, zero skip/xfail/error.
- `postgres`: exact canonical/child union, every release-critical node passes, zero skip/xfail/error.
- `compatibility`: six disjoint child sets equal canonical union and all pass.
- `non-slack`: canonical minus exact Slack ten, all selected tests pass, zero skip/xfail/error.
- `full`: exact Slack ten are the only failures, no error/skip/xfail, and `deferred_baseline_match=true`.
- every profile: `release_proof=true`, no live provider, balanced lease events, pre-drop lease-prefix zero, sentinel present, and outer `0:0:0:0`.

Any changed Slack node, missing module, collection mismatch, optional skip, new failure, residual resource, or cleanup uncertainty stops the release. Do not edit the manifest to make a run pass.

- [x] **Step 3: Run final code-quality gates**

```powershell
uv run --no-cache --locked pytest backend/tests/test_release_contracts.py backend/tests/test_release_bootstrap.py backend/tests/test_release_evidence_plugin.py backend/tests/test_postgres_isolation.py backend/tests/test_backend_release_matrix.py -q
uv run --no-cache --locked ruff check backend/tests scripts/backend_release_matrix.py
uv run --no-cache --locked python -m compileall -q backend/tests/release_contracts.py backend/tests/release_bootstrap.py backend/tests/release_evidence_plugin.py backend/tests/postgres_isolation.py scripts/backend_release_matrix.py
uv lock --check
git diff --check
```

Expected: all selected controller/helper tests pass and static gates are clean. Do not represent expected counts from this plan as observed output.

- [x] **Step 4: Write the operational runbook and product truth**

`docs/superpowers/runbooks/backend-release-matrix.md` must document:

- prerequisites and exact five official commands;
- `--child-id` as non-release development mode;
- bounded result codes and safe aggregate fields;
- container refusal behavior;
- schema/database/role ownership and cleanup precedence;
- how to handle residual resources without automatic adoption/deletion;
- exact Slack-baseline human gate;
- no-live-provider and artifact-retention policy.

Update `plan.md`, `docs/portfolio-log.md`, and
`docs/superpowers/runbooks/session-handoff.md` with only the actual commit SHA,
actual isolation-profile counts/outcomes, and cleanup result. The completed
follow-on Task 16 Steps 8–11 may now be recorded as observed. State explicitly
that the paid Terra/Mini gates remain pending and rollout is disabled; after
that authorization decision, Deliverable D planning precedes E and Slack stays
last.

- [x] **Step 5: Commit verification evidence separately**

```powershell
git add docs/superpowers/runbooks/backend-release-matrix.md plan.md docs/portfolio-log.md docs/superpowers/runbooks/session-handoff.md
git commit -m "docs: record backend release matrix verification"
git status --short
```

Expected: the documentation commit succeeds after all behavior-slice commits and the worktree is clean. Do not squash the reviewed implementation commits into this evidence commit.

## Final Implementation Review Checklist

- [ ] Every task has fresh RED and GREEN evidence from this implementation session.
- [ ] The controller, bootstrap, plugin, and lease helper are test-only and no production/Alembic file changed.
- [ ] The controller imports no application Settings/session/init module and invokes no unguarded app bootstrap.
- [ ] Parent `.env`, credentials, provider flags, fingerprint overrides, and pytest controls cannot enter a child.
- [ ] All release pytest processes are serial with plugin autoload disabled and exactly one explicit plugin.
- [ ] Exact profile child sets, node partitions, union, native exit, and sidecar identities are authoritative; counts alone cannot pass.
- [ ] Safe parameter ids preserve every function's case count and do not change Slack node ids.
- [ ] Every PostgreSQL consumer uses a leased URL; `current_schema()` is exact for SQLAlchemy, Alembic, LangGraph, and pgvector.
- [ ] No lease adopts/deletes a pre-existing schema; no outer controller adopts/deletes a pre-existing role/database/container/volume.
- [ ] Engines, sessions, pools, and savers close before lease exit.
- [ ] Review V2 fixture changed only after the exact evidence-boundary RED and now creates real relational C.5 provenance.
- [ ] Settings diagnostics pass without changing their assertions.
- [ ] PostgreSQL release-critical coverage has zero skip; non-Slack has no exclusion beyond Slack ten; full has no failure beyond Slack ten.
- [ ] Timeout and every failure path are bounded; cleanup continues and cleanup failure has highest precedence.
- [ ] Sidecars and aggregate reports contain no raw output, traceback, path, URL, DSN, resource name, source content, prompt, model output, or secret.
- [ ] Raw temp artifacts are deleted by exact path and final exact/run-prefix database/role counts are `0:0:0:0`.
- [ ] No live provider/connector call, paid evaluation, rollout enablement, Slack recovery, Deliverable D/E work, push, merge, or PR occurs under this plan.

## Execution Handoff

Approval of this document completes the **planning** checkpoint only. The immediate product implementation remains C.5 Task 6, followed by Tasks 7–15. At Task 16 entry, select one execution mode and explicitly authorize actual code changes:

1. **Subagent-Driven Development (recommended):** invoke `superpowers:subagent-driven-development` and execute one fresh worker plus spec/code review per task.
2. **Inline Plan Execution:** invoke `superpowers:executing-plans` and execute the same RED/GREEN/commit checkpoints sequentially.

Neither plan approval nor a focused `--child-id` run authorizes paid provider calls, rollout, Slack work, push, merge, or PR creation.

At Task 16, finish infrastructure Tasks 1–11 before C.5 Step 4. Execute this plan's Task 12 as the authoritative realization of C.5 Steps 4–7, then continue the remaining C.5 static/frontend/rollback/documentation gates.
