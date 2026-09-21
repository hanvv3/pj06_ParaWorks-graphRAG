# Backend Release Matrix Runbook

Updated: 2026-08-30

## Purpose

`scripts/checks/backend_release_matrix.py` is the authoritative non-paid backend
release verifier for C.5. It proves exact pytest coverage, PostgreSQL isolation,
the user-deferred Slack baseline, privacy-safe evidence, and cleanup. A focused
`--child-id` run is development evidence only and always reports
`release_proof=false`.

This controller never starts, replaces, or adopts Docker resources and never
calls an LLM, embedding provider, connector, or OAuth service.

## Prerequisites

- Run from the repository root with the locked `uv` environment.
- The existing `paraworks-postgres` container must already be healthy.
- Its image must be `pgvector/pgvector:pg17`, compose service must be
  `postgres`, and PostgreSQL must be bound only to `127.0.0.1:55432`.
- Port, image, health, compose label, server role, and database identity must
  all match. Any mismatch returns `preflight_refused`; do not start or replace
  the service from this runbook.
- Paid-provider credentials and authorization flags are removed from every
  child environment.

## Official commands

Run all five profiles serially:

```powershell
uv run --locked python scripts/checks/backend_release_matrix.py --profile settings-diagnostic
uv run --locked python scripts/checks/backend_release_matrix.py --profile postgres
uv run --locked python scripts/checks/backend_release_matrix.py --profile compatibility
uv run --locked python scripts/checks/backend_release_matrix.py --profile non-slack
uv run --locked python scripts/checks/backend_release_matrix.py --profile full
```

Observed on 2026-08-30 at behavior commit `4b9132a`:

| Profile | Collected | Selected result | Explicitly deselected | Errors/skips/xfails | Lease create/drop | Outcome |
|---|---:|---:|---:|---:|---:|---|
| settings-diagnostic | 6 | 6 passed | 0 | 0/0/0 | 2/2 | passed |
| postgres | 394 | 394 passed | 0 | 0/0/0 | 10/10 | passed |
| compatibility | 1,595 | 1,591 passed | 4 Slack | 0/0/0 | 91/91 | passed |
| non-slack | 2,036 | 2,026 passed | 10 Slack | 0/0/0 | 150/150 | passed |
| full | 2,036 | 2,026 passed, 10 failed | 0 | 0/0/0 | 150/150 | passed baseline |

For `full`, the ten failures exactly equal the frozen Slack manifest and
`unexpected_nodeids` is empty. Every profile reported `release_proof=true`,
`no_live_provider=true`, and post-cleanup database/role counts `0/0`.

## Evidence contract

The controller prints one aggregate JSON object. Safe fields include profile,
bounded outcome, collection/selection/deselection counts, pass/fail/error/skip/
xfail counts, sidecar counts, lease counts, unexpected node ids, ownership
booleans, cleanup counts, and `release_proof`.

It does not print captured pytest output, traceback, DSN, password, temporary
resource name, filesystem artifact path, source content, prompt, model output,
credential, or HMAC material. Sidecars are written only inside the exact
temporary run directory, validated by invocation and content hashes, and
deleted when the controller exits.

Bounded outcomes are `passed`, `preflight_refused`, `collection_mismatch`,
`evidence_refused`, `verification_failed`, `baseline_mismatch`, and
`cleanup_failed`. Cleanup failure has highest precedence.

## Isolation and cleanup

Each run creates a random controller-owned `_test` role and `_test` database
only after confirming both names do not exist. It creates the `vector`
extension, then gives verification modules serial, uniquely named schema
leases. Ordinary application tests receive a separate temporary SQLite path.

Every lease must have exactly one `created` event followed by one `dropped`
event. The controller terminates only connections to its exact owned database,
drops only that database and role, and verifies both catalog counts are zero.
It never adopts or deletes a pre-existing role, database, schema, container,
volume, or unrelated process.

If cleanup reports a residual resource, stop. Inspect the bounded aggregate and
the exact controller code; do not broaden a delete command or rerun with a
guessed resource name. A later run creates new random identities and must not
adopt the residual resource.

## Slack and paid-gate boundaries

The exact ten Slack node ids in `backend/tests/release_contracts.py` are a
human-approved deferred baseline because the original Slack data source no
longer exists. Do not add, rename, or suppress a node to make a profile pass.
Compatibility currently intersects that list in four nodes; non-Slack removes
all ten; full must fail on exactly those ten and no others.

These profiles do not authorize the paid Terra validation gate or the distinct
paid Mini extraction gate. Both require explicit user authorization, sanitized
fixtures, and their own aggregate-only CLI commands. Until both pass, keep
auto-review rollout `disabled`.
