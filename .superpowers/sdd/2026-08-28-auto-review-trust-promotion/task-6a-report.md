# Task 6A RAG/orchestrator revocable trusted-serving report

## Status

GREEN. Task 6A is implemented and verified without modifying any Task 6B-owned file. Slack remains deferred and legacy-dedupe-only.

## What was implemented

- Added one fail-closed trusted-serving authority for raw chunks and explicit/legacy trusted knowledge. It verifies human-only raw approval, current Source signature or exact current verified parser revision, current document/parser policy identity, active provenance, validation, audit quarantine/correction, permission, and tombstones.
- Added exact auto-review revoke, replay, immutable assessment, audit transition, source invalidation, bounded recovery/status/repair, and fixed-output admin reconciliation services.
- Added exact vector deletion and permission narrowing contracts, transaction-aware in-memory mutations, conditional tombstone-safe pgvector writes, pre-ranking tombstone/provenance/audit filters, and post-provider eligibility checks.
- Added the shared generation barrier and exact per-document PostgreSQL advisory lock contract used by production reindex and revoke mutations.
- Added actor-aware trusted-knowledge and review-evidence projections across Knowledge, Dashboard, Projects, Review, Todo, Notification, Integration, Search, Ask, Assistant, and orchestration boundaries.
- Added exact server-only RAG dependency snapshots and atomic assistant dependency persistence. Every later assistant read/context/summary/email projection rechecks the selected approval effect, complete evidence-child set, source/parser identity, permission, and serving-content hash; stale evidence becomes bounded `evidence_unavailable`.
- Kept persisted ORM permission values immutable during read projection and rejected unknown/stale vector identities and stale vector content before answer generation.
- Explicitly excluded Slack sources from the C.5 raw/trusted serving authority without changing Slack implementation code.
- Accepted either the authoritative `server-source-content:v1` signature or exactly one current verified `DocumentParserRun.revision_id` as the canonical evidence version identity. PostgreSQL pre-ranking SQL and ORM eligibility use the same rule.

## Files changed

Created:

- `backend/app/admin/auto_review_source_reconciliation.py`
- `backend/app/knowledge/trusted_serving_eligibility.py`
- `backend/app/rag/serving_locks.py`
- `backend/app/review/auto_review_audit_transitions.py`
- `backend/app/review/auto_review_revoke.py`
- `backend/app/review/auto_review_source_reconciliation.py`
- `backend/app/review/evidence_visibility.py`
- `backend/tests/test_auto_review_revocation.py`
- `backend/tests/test_auto_review_source_reconciliation.py`
- `backend/tests/test_auto_review_source_reconciliation_admin.py`
- `backend/tests/test_review_evidence_visibility.py`

Modified:

- `backend/app/agents/rag_orchestrator_agent/agent.py`
- `backend/app/agents/rag_orchestrator_agent/service.py`
- `backend/app/api/v1/assistant.py`
- `backend/app/api/v1/dashboard.py`
- `backend/app/api/v1/integrations.py`
- `backend/app/api/v1/knowledge.py`
- `backend/app/api/v1/notifications.py`
- `backend/app/api/v1/projects.py`
- `backend/app/api/v1/review.py`
- `backend/app/api/v1/search.py`
- `backend/app/api/v1/todos.py`
- `backend/app/assistant/service.py`
- `backend/app/main.py`
- `backend/app/projects/service.py`
- `backend/app/rag/indexing.py`
- `backend/app/rag/pgvector_store.py`
- `backend/app/rag/reindexing.py`
- `backend/app/rag/vector_store.py`
- `backend/tests/test_ask_api.py`
- `backend/tests/test_assistant_api.py`
- `backend/tests/test_assistant_service.py`
- `backend/tests/test_knowledge_api.py`
- `backend/tests/test_pgvector_integration.py`
- `backend/tests/test_pgvector_store.py`
- `backend/tests/test_rag_indexing.py`
- `backend/tests/test_rag_orchestrator_service.py`
- `backend/tests/test_search_permissions.py`
- `backend/tests/test_search_retrieval_backend.py`
- `backend/tests/test_todos_api.py`

The remaining Task 6A-owned files already contained the required Tasks 1-5 contracts at the assigned base and needed no diff. No Task 6B-owned source or test file changed.

## TDD RED evidence

The final self-review behavior test was added before its production fix and run exactly as follows (both database variables were pinned to the same isolated SQLite URL):

```powershell
$env:UV_CACHE_DIR='.tmp/task6a/uv-cache'; $env:PARAWORKS_DATABASE_URL='sqlite:///./.tmp/task6a/selected-effect-red.sqlite'; $env:DATABASE_URL=$env:PARAWORKS_DATABASE_URL; uv run --locked pytest backend/tests/test_assistant_service.py::test_assistant_dependency_rechecks_the_selected_effect_even_when_shared_provenance_remains -q
```

Observed RED: `1 failed`; `KeyError: 'status'` at the assertion requiring `evidence_unavailable`. The target still had valid shared human provenance, so target-level eligibility remained true, but the exact auto-policy effect stored with the assistant answer had lost its validation authority. The failure proved the selected effect itself was not being rechecked.

The minimal fix added `TrustedServingEligibilityService.approval_link_is_live`, made RAG snapshot selection skip invalid effects, and made assistant projection recheck the stored exact effect. Its focused GREEN command was:

```powershell
$env:UV_CACHE_DIR='.tmp/task6a/uv-cache'; $env:PARAWORKS_DATABASE_URL='sqlite:///./.tmp/task6a/selected-effect-green.sqlite'; $env:DATABASE_URL=$env:PARAWORKS_DATABASE_URL; uv run --locked ruff check backend/app/knowledge/trusted_serving_eligibility.py backend/app/agents/rag_orchestrator_agent/service.py backend/app/assistant/service.py backend/tests/test_assistant_service.py; uv run --locked pytest backend/tests/test_assistant_service.py::test_assistant_dependency_rechecks_the_selected_effect_even_when_shared_provenance_remains backend/tests/test_assistant_service.py::test_rag_answer_persists_complete_exact_dependencies_with_message_atomically backend/tests/test_assistant_service.py::test_source_changes_between_rag_retrieval_and_message_commit_fail_closed_without_raw_answer_persistence -q
```

Observed GREEN: `All checks passed!`; `3 passed in 0.23s`.

Other meaningful REDs captured before their corresponding minimal fixes included:

- writer/tombstone focused set: six behavioral failures before exact `delete_many`, rollback-safe in-memory mutations, and tombstone skips existed;
- serving-lock focused set: four `ModuleNotFoundError` failures before `serving_locks.py` existed;
- `test_search_pgvector_candidate_without_live_identity_is_dropped`: returned `chunk:999999` instead of `[]`;
- `test_search_pgvector_candidate_with_stale_content_for_live_identity_is_dropped`: returned stale bytes instead of `[]`;
- `test_crash_after_source_document_commit_before_reconciliation_is_fail_closed_and_recoverable`: expected `stale_count == 1`, observed `0`;
- `test_human_approved_deferred_slack_chunk_never_gains_trusted_serving_authority`: returned a Slack chunk instead of excluding it;
- `test_knowledge_visibility_projection_does_not_mutate_persisted_permission`: read projection mutated persisted permission from `public` to `internal`;
- `test_assistant_tool_middleware_redacts_current_response_after_dependency_revoke`: current response still exposed a raw source link;
- assistant dependency commit race: expected `ValueError`, but no exception was raised;
- `test_pgvector_search_excludes_critical_audit_correction_before_ranking`: the pre-ranking SQL omitted audit corrections;
- `test_current_verified_parser_revision_is_valid_explicit_evidence_identity`: exact current parser revision was incorrectly ineligible.

Each was rerun GREEN in its focused module before the broad gates below.

## Final GREEN commands and results

Focused cross-boundary regression gate, with both database variables pinned to one isolated SQLite URL:

```powershell
uv run --locked pytest backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_revocation.py backend/tests/test_review_evidence_visibility.py backend/tests/test_pgvector_store.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_search_retrieval_backend.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_knowledge_api.py backend/tests/test_project_memory_api.py backend/tests/test_dashboard_api.py -q
```

Observed before the final exact-effect regression was added: `124 passed in 7.85s`; the exact-effect focused command above then passed `3` tests, and the final Step 7 broad gate includes the new regression.

Original Task 6A Step 7 pytest command from the brief, with `PARAWORKS_DATABASE_URL` and `DATABASE_URL` pinned to the same isolated SQLite URL:

```powershell
uv run --locked pytest backend/tests/test_auto_review_revocation.py backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_source_reconciliation_admin.py backend/tests/test_review_evidence_visibility.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_integration.py backend/tests/test_knowledge_api.py backend/tests/test_dashboard_api.py backend/tests/test_review.py backend/tests/test_todos_api.py backend/tests/test_notifications_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py backend/tests/test_ask_api.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_email_agent.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_project_memory_api.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_agent_runtime_lifespan.py -q
```

Observed baseline-equivalent result: `237 passed, 4 failed, 1 skipped in 17.89s`. The skip was the explicit pgvector integration opt-in; it was executed in the PostgreSQL gate below. The exact failures were:

1. `backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_runs_real_agent_services`
2. `backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_skips_agents_that_exceed_cost_budget`
3. `backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_uses_cache_when_evidence_is_unchanged`
4. `backend/tests/test_orchestration_api.py::test_company_memory_orchestration_api_runs_agent_services`

All four are members of the approved exact ten Slack deselections in `docs/superpowers/plans/2026-08-29-whole-suite-postgresql-isolation.md` lines 297-310. No failure was outside the approved list, so Slack code was not changed.

The isolated PostgreSQL/pgvector gate used healthy tracked container `paraworks-postgres` at `127.0.0.1:55432`. Exact owned identities were `paraworks_task6a_database_test` and `paraworks_task6a_role_test`; preflight catalog counts were `0` and `0`. `PARAWORKS_DATABASE_URL`, `DATABASE_URL`, and `PARAWORKS_PGVECTOR_TEST_DATABASE_URL` were all pinned to that same one-time isolated URL. The original Step 7 selector list was run with only the approved exact ten `--deselect=<nodeid>` arguments, in approved order.

Observed result: `238 passed, 4 deselected in 18.69s`. Only four of the ten approved nodes belonged to the Step 7 selected modules, hence four collected deselections. There were zero failures, errors, or skips. The exact database was dropped, then the exact role was dropped; postflight catalog counts were `0` and `0`.

Exact Task 6A Step 7 Ruff selector command from the brief:

```powershell
uv run --locked ruff check backend/app/review/auto_review_revoke.py backend/app/review/auto_review_source_reconciliation.py backend/app/review/auto_review_audit_transitions.py backend/app/review/evidence_visibility.py backend/app/admin/auto_review_source_reconciliation.py backend/app/knowledge/trusted_serving_eligibility.py backend/app/rag/serving_locks.py backend/app/rag/indexing.py backend/app/rag/reindexing.py backend/app/rag/vector_store.py backend/app/rag/pgvector_store.py backend/app/models/vector_index.py backend/app/api/v1/knowledge.py backend/app/api/v1/dashboard.py backend/app/api/v1/review.py backend/app/api/v1/todos.py backend/app/api/v1/notifications.py backend/app/api/v1/integrations.py backend/app/api/v1/projects.py backend/app/api/v1/search.py backend/app/api/v1/ask.py backend/app/api/v1/assistant.py backend/app/assistant/service.py backend/app/agent_runtime/company_memory.py backend/app/projects/service.py backend/app/agents/rag_orchestrator_agent/agent.py backend/app/agents/rag_orchestrator_agent/service.py backend/app/main.py backend/tests/test_auto_review_revocation.py backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_source_reconciliation_admin.py backend/tests/test_review_evidence_visibility.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_integration.py backend/tests/test_knowledge_api.py backend/tests/test_dashboard_api.py backend/tests/test_review.py backend/tests/test_todos_api.py backend/tests/test_notifications_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py backend/tests/test_ask_api.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_email_agent.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_project_memory_api.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_agent_runtime_lifespan.py
git diff --check
```

Observed: `All checks passed!`; `git diff --check` exit `0` (only Git's existing LF-to-CRLF checkout notices).

## Skipped or environment-only attempts

- The baseline SQLite command intentionally skipped `test_pgvector_reindex_path_with_fake_embedding` because `PARAWORKS_PGVECTOR_TEST_DATABASE_URL` was absent. The final isolated PostgreSQL gate ran it successfully, so no required test remains skipped.
- Before explicit database pinning, one focused attempt produced `56` setup errors and a later indexing/pgvector attempt produced `15` setup errors because inherited `.env` settings targeted unavailable `localhost:5432`. The same suites passed after pinning both database variables; these were environment errors, not product-code failures.
- An initial isolated-role pgvector attempt could not create the extension because extension installation is an owner/admin operation. The extension was created inside only the exact isolated database by the tracked local development admin, the test role ran the suite, and the database/role were then removed with catalog `0/0` proof.
- One sandboxed uv process launch was refused with Windows `Access is denied` for uv's managed Python outside the workspace. The same locked commands were rerun through the approved uv execution boundary and passed.

## Self-review findings

- Found and fixed exact-effect drift: target-level eligibility alone was insufficient when shared provenance kept a target live. Snapshot selection and assistant replay now validate the deterministic selected effect itself.
- Found and fixed ORM read-projection mutation in Knowledge/Projects/Dashboard; actor-effective permissions are projected through wrappers and persisted canonical permissions remain unchanged.
- Found and fixed stale candidate identity/content fallback at the final Search/RAG boundary; unknown ids and content-hash mismatches are dropped rather than serialized.
- Found and fixed current-response assistant leakage; all evidence-shaped public serialization paths now require actor/database revalidation.
- Found and fixed PostgreSQL/ORM semantic drift for audit corrections and current parser revision identity.
- Confirmed no Task 6B-owned file changed and no live provider, connector, OAuth, embedding, Slack, or paid call ran.

## Concerns and Task 6B interface pressure

No Task 6A blocker remains. Task 6B must preserve the existing shared contract exactly: compute and store the server-owned signature, maintain exact current `Document`/`DocumentVersion`/`DocumentParserRun` pointers and parser policy identities, propagate permission-only state changes to current chunks, and invoke synchronous reconciliation after committing source state. Task 6A deliberately does not synthesize missing/ambiguous source identity and fails closed until Task 6B supplies it.

Slack remains intentionally outside C.5 authority. The four baseline failures above are unchanged approved deferred-Slack nodes; no Slack client, data, OAuth, or feature work was added.
