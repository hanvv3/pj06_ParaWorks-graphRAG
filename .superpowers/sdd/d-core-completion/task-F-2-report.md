# Task F-2 report — PostgreSQL/pgvector integration evidence

## Environment and cleanup

- Worktree/branch: `review-hitl-v2-design` / `codex/rag-orchestrator-agent`.
- Base inspected: `c4623146caf78932741440ff5d0d2bd68552d36a`.
- Final correction code revision: `9200ea3edd02590ef29e9156c3ea7f3a8fc27bf6`;
  original F-2 evidence baseline commit: `a7a58f8`.
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

## Historical initial diagnosis (superseded)

Before the later corrections, `backend/tests/test_rag_v2_costs_postgres.py`
exposed the paid phase-2 authority mismatch and then the recovery transaction
failure. Those observations caused the fail-closed exact advisory-transport
and pre-commit terminal-data corrections recorded below. They are not the
current F-2 result; formal release remains NOT CLEAN.

## Final correction and completion evidence

- The recovery root cause was confirmed: constructing terminal data after
  `_commit` read expired ORM rows, reopening the pinned transaction. Finals and
  terminal are now built before commit; the focused recovery selector is green.
- Review-round composition RED->GREEN: `_postgres_finalizer` now binds and
  passes the bootstrap-bound advisory transport to paid safety. Its focused
  production composition selector is **2 passed in 4.54s**.
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
- Initial F-2 cleanup counts were `leased_schemas=0` and
  `active_peer_sessions=0`. A post-correction read-only check briefly observed
  one unowned lease/session; it cleared without termination or deletion. Final
  read-only ownership check is again `leased_schemas=0` and
  `active_peer_sessions=0`.

**D functional baseline passed / E-1 may start / formal release deferred and
NOT CLEAN.** Actual-model quality, reviewer OAuth, 30-case publication, live
rollout, and capability P1 remain outside this evidence.
