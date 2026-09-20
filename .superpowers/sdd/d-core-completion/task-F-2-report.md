# Task F-2 report — PostgreSQL/pgvector integration evidence

## Environment and cleanup

- Worktree/branch: `review-hitl-v2-design` / `codex/rag-orchestrator-agent`.
- Base inspected: `c4623146caf78932741440ff5d0d2bd68552d36a`.
- Disposable local PostgreSQL identity: database `paraworks_rag_test`, role
  `paraworks_rag_test`, pgvector `0.8.2`; no provider credentials or network
  provider calls were used.
- Final read-only cleanup check found no active non-observer database sessions
  and no `paraworks_c5t16_%` leased schemas. The twelve pre-existing
  `rag_task12_cost_%` schemas from interrupted earlier attempts were preserved
  rather than deleted outside a successful fixture cleanup path.

## Selectors and results

- Direct Alembic upgrade to head and `scripts/check_db_schema.py` passed after
  migration corrections. `check_db_schema.py --ensure-vector-schema
  --expect-app-schema` confirmed the vector and application schema.
- `backend/tests/test_rag_v2_migration.py::test_postgresql_fresh_schema_reaches_head`
  was RED for duplicate b5 check/legacy trigger setup and GREEN in `0.92s`.
- `backend/tests/test_pgvector_integration.py` was RED when its fixtures used
  the migrated `public` schema through `checkfirst`; after per-test leased
  schema isolation it was GREEN: `29 passed in 118.42s`. These tests use real
  PostgreSQL/pgvector with deterministic/refusing fake embedding models and
  cover fake embedding write/search, permission and hidden-match behavior,
  stale evidence/revocation, and canonical evidence snapshots.
- The required six-file command was run with all four mandated DB variables.
  Its first run was `34 failed, 284 passed, 13 warnings in 29.09s`; pgvector
  fixture failures were subsequently resolved; the completion evidence below
  records the final exact aggregate result.

## TDD changes

- RED: migration b5 attempted to recreate checks already emitted by the current
  schema migration. GREEN: inspect existing checks and add only missing ones.
- RED: a fresh isolated schema resolved tables in migrated `public` because
  `create_all(checkfirst=True)` searched the full path. GREEN: the initial
  migration creates its intended schema explicitly, and pgvector tests use
  per-test schema leases with explicit table creation.
- Corrected pgvector test fixtures so the stale guarded upsert reaches its
  relational guard and assistant citations carry their full evidence payload.

## Initial diagnosis and limitations

`backend/tests/test_rag_v2_costs_postgres.py` exposed a real authority
composition mismatch. The paid phase-2 assembler now fails closed unless it
uses the exact provider-safety `RagPostgresAdvisoryTransport` with the same
runtime-health authority as `RagPostgresDatabaseAuthority`; owner and evidence
operations remain on the database authority. This clears the original
`provider advisory connection authority changed` failure. The focused recovery
test then reaches a later lifecycle failure, `pinned PostgreSQL transaction did
not end`, so the exact suite is not yet green. Frontend lint/type/build,
controlled-fake Playwright, final product docs, and a local commit are not
claimed: F-2 acceptance is incomplete. Formal release remains NOT CLEAN.

## Final correction and completion evidence

- The recovery root cause was confirmed: constructing terminal data after
  `_commit` read expired ORM rows, reopening the pinned transaction. Finals and
  terminal are now built before commit; the focused recovery selector is green.
- Required exact real-PG command: **318 passed, 13 warnings**, no skips. It
  covers the supplied non-superuser role, same-DB/pinned transaction authority,
  cost actual-or-reserve preservation, C.5/advisory serialization, readiness,
  permission/hidden matches, evidence revoke, and native pgvector search.
  Embeddings/providers are controlled fakes; SQLite smoke is not in this F-2
  command.
- `alembic upgrade head`, `scripts/check_db_schema.py`, and
  `scripts/check_pgvector_dev.py --ensure-vector-schema --expect-app-schema`
  passed. Frontend lint/type/build passed. The 48-case controlled-fake Chromium
  run's two timing-sensitive initial failures passed in a fresh `--last-failed`
  retry; final Playwright state is `passed` with no failed tests.
- Cleanup counts are `leased_schemas=0` and `active_peer_sessions=0`.

**D functional baseline passed / E-1 may start / formal release deferred and
NOT CLEAN.** Actual-model quality, reviewer OAuth, 30-case publication, live
rollout, and capability P1 remain outside this evidence.
