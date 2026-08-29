# Task 6 I2/I3 Recovery Fix Report

## Scope

- Base: `82654be22c53e7dae8ccb1fc8142f0791c77ab91`
- Branch: `codex/task6-fix-recovery`
- Commit message: `fix: make source recovery authoritative`
- Findings addressed: I2 current-document-pointer repair/readiness and I3
  startup/admin pgvector convergence only.
- Ask/reindex behavior, connector parser-metadata projection, canonical resolver,
  Slack, and unrelated refactors were not changed.

## Root cause

### I2

Pointer repair treated any non-null parser policy/version strings as authority.
It did not compare the parser name, MIME-derived registry policy, implementation
version, chunk-policy version, or exact run/version/chunk relation. Repair also
computed `remaining_count` only from the selected page, while `status()` ignored
null current-version pointers entirely. A bounded run could therefore return
`readiness=true` and CLI exit 0 with unscanned or ambiguous work retained.

### I3

Synchronous ingestion reconciliation could receive a vector writer, but startup
and the local admin CLI constructed `AutoReviewSourceReconciliationService`
without one. Logical revoke/narrowing could commit and report ready while a
stale or broader physical pgvector row remained.

## Implementation

- Added a registry-owned parser-policy lookup for persisted source type/MIME;
  event ingestion continues through the same registry function.
- Pointer repair now requires the exact server signature schema/signature,
  parser status/name/MIME/policy/implementation/chunk-policy tuple, exactly one
  matching parser run/version, and a complete contiguous chunk relation.
- Legacy, wrong-policy, incomplete, or multiply exact relations remain
  ambiguous and require resync; no pointer is guessed.
- `status()` now includes global null-pointer work and bounded ambiguity
  inspection. Repair performs an authoritative post-batch null-pointer count,
  so unscanned continuation keeps readiness false and CLI exit 3.
- Added one shared reconciliation-service factory. PostgreSQL sessions receive
  a same-session `PgVectorStore` configured with the server embedding dimension;
  SQLite smoke sessions remain writer-free. Startup and admin use this factory.
- Added a schema-isolated real pgvector recovery test that begins with a broad,
  stale physical row and proves startup recovery deletes it before reporting
  ready. It uses no embedding provider.

## Files

- `backend/app/ingestion/source_content_signature.py`
- `backend/app/review/auto_review_source_reconciliation.py`
- `backend/app/main.py`
- `backend/app/admin/auto_review_source_reconciliation.py`
- `backend/tests/test_auto_review_source_reconciliation.py`
- `backend/tests/test_auto_review_source_reconciliation_admin.py`
- `backend/tests/test_pgvector_integration.py`
- `.superpowers/sdd/2026-08-28-auto-review-trust-promotion/task-6-fix-recovery-report.md`

## TDD evidence

### RED

- Focused I2/I3 unit run: `9 failed, 1 passed`.
  - Wrong parser name/MIME/policy/implementation/chunk-policy identities were
    incorrectly repaired.
  - Missing chunk lineage was incorrectly repaired.
  - A second null pointer beyond `limit=1` was hidden by false readiness.
  - Status/CLI ignored ambiguous pointer work.
  - The PostgreSQL reconciliation factory did not exist.
- Real pgvector startup schedule against the old wiring: reconciliation returned
  ready, but the assertion failed with `physical_count == 1` instead of `0`.

### GREEN

- Final focused unit/CLI/lifespan/parser/vector/revoke gate:
  `91 passed in 5.06s`.
- Final PostgreSQL + pgvector integration gate:
  `11 passed in 6.99s`, zero skips.
- Focused Ruff: `All checks passed!`
- `git diff --check`: passed.

## PostgreSQL isolation and cleanup

- Pinned all required database URLs to the exact disposable database/role
  `task6fix_recovery_test` on `127.0.0.1:55432`.
- Both database and role names end in `_test`.
- Test uses a unique schema and drops it in `finally`.
- Final disposable database/role cleanup verification: `0|0`.

## Remaining issues

- No remaining I2/I3 issue was found in the requested scope.
- The independent reviewer subagent could not be started because all four team
  slots were occupied; the final diff was inspected directly and the parent
  agent retains the whole-task review boundary.
- Findings C1, I1, I4, I5, and all explicitly excluded domains remain owned by
  their separate recovery tasks.
