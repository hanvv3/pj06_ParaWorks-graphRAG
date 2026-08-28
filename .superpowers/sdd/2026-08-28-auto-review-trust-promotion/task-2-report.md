# Task 2 Report: C.5 Persistence with Enforced Provenance Ownership

## Outcome

Task 2 is complete in feature commit `d7cbd15e3c7817bfe897ff11d65c21a77af3f161`
(`feat: persist auto review trust state`). The slice preserves V2.0 nullable
compatibility while adding the C.5 persistence schema, source/parser authority,
Assistant evidence dependencies, append-only control ledgers, PostgreSQL
deferred guards, fixed keyed-mutation lock ordering, startup key bootstrap, and
empty-only downgrade behavior. Slack remains C.5-ineligible; its only contract
change is the additive trailing nullable `SourceEvent.semantic_timestamp_raw`.

## Files

Production:

- `backend/app/models/auto_review.py`
- `backend/app/models/__init__.py`
- `backend/app/models/agent_runs.py`
- `backend/app/models/agent_workflows.py`
- `backend/app/models/assistant.py`
- `backend/app/models/enums.py`
- `backend/app/models/review.py`
- `backend/app/models/source.py`
- `backend/app/connectors/base.py`
- `backend/app/agent_runtime/keyed_mutation_guard.py`
- `backend/app/admin/auto_review_keys.py`
- `backend/app/admin/data_reset.py`
- `backend/app/main.py`
- `backend/migrations/versions/7c5a2e9f4b10_add_auto_review_trust_promotion.py`

Tests:

- `backend/tests/test_auto_review_migration.py`
- `backend/tests/test_agent_runtime_migration.py`
- `backend/tests/test_db_schema_operations.py`
- `backend/tests/test_models.py`
- `backend/tests/test_assistant_models.py`
- `backend/tests/test_data_reset.py`
- `backend/tests/test_keyed_mutation_guard.py`
- `backend/tests/test_auto_review_key_bootstrap.py`
- `backend/tests/test_agent_runtime_lifespan.py`
- `backend/tests/test_connector_ingestion_contract.py`

## TDD Evidence

### RED

The named focused command was run after adding the Task 2 tests and before any
Task 2 production implementation. Collection failed only on the absent Task 2
surface:

- `AutoReviewExtractionCall` was missing.
- `AssistantMessageEvidenceDependency` was missing.
- `AutoReviewRuntimeKeyState` was missing.
- `backend.app.agent_runtime.keyed_mutation_guard` was missing.
- `backend.app.admin.auto_review_keys` was missing.

A later PostgreSQL-specific TDD cycle added the gapless provider-event test and
observed `Failed: DID NOT RAISE DBAPIError` for an illegal sequence-3 insert
against a sequence-1 aggregate before adding the deferred guard.

### GREEN

Final exact focused command, with PostgreSQL-only tests enabled through an
isolated local Docker database on port 55432:

```text
uv run --locked pytest backend/tests/test_auto_review_migration.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_schema_operations.py backend/tests/test_models.py backend/tests/test_assistant_models.py backend/tests/test_data_reset.py backend/tests/test_keyed_mutation_guard.py backend/tests/test_auto_review_key_bootstrap.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_connector_ingestion_contract.py -q
```

Result: `67 passed, 21 warnings in 7.30s`.

All warnings are the existing Alembic `prepend_sys_path`/`path_separator`
deprecation warning. No test was skipped in this run. No live provider or
connector was called.

The focused PostgreSQL gapless-event test separately passed:
`1 passed, 1 warning in 0.70s`.

## Migration Evidence

- Revision: `7c5a2e9f4b10`
- Down revision: `2f6a8b9c0d1e`
- Fresh PostgreSQL upgrade from an empty database reached head successfully.
- Empty PostgreSQL downgrade removed the Task 2 schema and reported current
  revision `2f6a8b9c0d1e`.
- Populated PostgreSQL downgrade failed closed with:
  `RuntimeError: retained C.5 state in auto_review_provider_safety_states; schema downgrade refused`.
- The PostgreSQL cutover test proved an overlapping old-style workflow-owned
  ReviewItem commit is rejected, an item plus exact same-workflow evidence ref
  commits, and the schema-boundary marker cannot be deleted.
- The PostgreSQL event test proved an out-of-sequence event without the exact
  aggregate backpointer is rejected at transaction commit.

## Lint and Diff Verification

The exact named Ruff command was rerun after its automatic formatting pass:

```text
uv run --locked ruff check backend/app/models backend/app/connectors/base.py backend/app/agent_runtime/keyed_mutation_guard.py backend/app/admin/auto_review_keys.py backend/app/admin/data_reset.py backend/app/main.py backend/migrations/versions/7c5a2e9f4b10_add_auto_review_trust_promotion.py backend/tests/test_auto_review_migration.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_schema_operations.py backend/tests/test_models.py backend/tests/test_assistant_models.py backend/tests/test_data_reset.py backend/tests/test_keyed_mutation_guard.py backend/tests/test_auto_review_key_bootstrap.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_connector_ingestion_contract.py
```

Result: `All checks passed!`.

`git diff --cached --check` passed before the feature commit. The repository's
Windows checkout emitted LF-to-CRLF notices; these are line-ending notices, not
diff errors.

## Environment-Dependent Checks

Docker was available. The repository PostgreSQL/pgvector image was exercised
on alternate host port 55432 because the default port 5432 was already occupied.
Both PostgreSQL-only focused tests ran; there are no skipped
environment-dependent checks to report.

## Residual Risks

- SQLite intentionally relies on the brief-authorized service-equivalent lock
  checks; PostgreSQL remains the production enforcement path for deferred
  relational guards and advisory locks.
- The migration's PostgreSQL behavior was exercised on the repository's local
  `pgvector/pgvector:pg17` image. Other supported PostgreSQL deployment images
  should retain the same PL/pgSQL and deferrable-trigger behavior, but were not
  separately exercised.
- Task 2 only establishes persistence and guards. Task 3 and later services must
  consume the sole keyed-mutation guard API and must not issue independent
  advisory-lock SQL or reinterpret connector signatures as source authority.
