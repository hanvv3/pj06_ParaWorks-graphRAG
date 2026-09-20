# C-1 independent answer-cache storage

C-1 adds storage only. No API, answer graph, provider transport, finalization,
Review, permission or cost policy is changed. `create_answer_cache` defaults to
disabled; disabled and SQLite instances are `NullAnswerCache` without opening
a connection or writing schema. There is no rollout flag activation.

## Internal contract for C-2

`build_answer_cache_key` accepts exact `SecurityScope`, E's
`PreparedModelInfluenceSet` and `AnswerCacheVersions`. It authenticates the
existing prepared v3 signature and every ordered graph path v1. Every model
influence is included, even evidence not selected for citation. Key material
binds exact scope/principal/workspace, prepared rendered input HMAC, all evidence
identity/version references and graph paths, corpus/index/readiness, prompt,
model configuration, output schema, runtime policy, retrieval/graph policy,
seed/effective backend, and fingerprint key version/material.

`put(key, answer=..., slots=...)` authenticates key and all current slot
observations, then recomputes `ValidatedAnswerBlocks` through the real
`RagAnswerOutputValidator`. Inject the current answer policy's validator/signer;
an unchecked dict or constructed/altered dataclass is not eligibility.
Empty/no-match/insufficient answers are not stored. Stored answer JSON contains
only selected text/slot/support blocks and the required null insufficient reason.
There is no assembled-answer duplicate, citation, URL/snippet metadata, prompt,
question or old provider receipt.

`get(key, slots=...)` returns `AnswerCacheHit` or a safe miss. The hit contains
newly validated blocks, key/scope/value HMACs and signed creation/expiry seconds.
The signature binds exact answer bytes, complete dependencies, key/scope and
timestamps. C-2 can preserve those identities for explicit cache-hit accounting;
this carrier is **not** evidence approval or provider execution authority.
Current PostgreSQL permission, canonical evidence and relation reads still belong
to E's preparation/finalization boundaries. Their failures must propagate closed.
After retrieval, C-2 must revalidate all influences and paths before publication,
rebuild citations, create a new run/audit and record actual query-embedding cost
plus zero generation cost. C-1 does not implement those operations.

## Storage and migration

One table, `rag_answer_cache_entries`: key HMAC primary key, scope HMAC,
canonical dependency JSON, selected answer JSON, created/expiry epoch seconds,
and value HMAC. Answer text is sensitive; reads require the exact authenticated
scope/key. Dedicated short transactions avoid altering a caller's authority
transaction. PostgreSQL statement/lock timeouts are bounded. Store SQL/connection
faults are misses/unsuccessful writes; no raw value is logged by this module.

The fixed TTL is 3,600 seconds; database constraints cap all entries at 24 hours.
Reads and same-key unexpired writes never extend retention. Expiry is checked
again after I/O/validation. A fresh validated write can replace an expired entry.
`cleanup(limit=100)` deletes at most 1–1,000 expired entries using ordered
`FOR UPDATE SKIP LOCKED`; scheduling cleanup is an explicit C-2/operator duty.

Alembic `a6b7c8d9e0f1` follows `d7a8b9c0d1e2`. It handles the historical baseline
which creates current metadata on empty databases. Downgrade drops only the cache
table/index and its expendable entries, preserving source and audit tables.

## Verification boundary

`backend/tests/test_rag_answer_cache.py` uses E's real canonical preparation,
the real answer validator and a fake signer. Actual leased PostgreSQL tests cover
scoped reads, mutation, expiry, bounded cleanup, database retention, full fresh
Alembic head upgrade/downgrade/re-upgrade and existing-schema forward creation.
Each lease is dropped after the test. A refused local connection proves safe
store failure; connection traps prove disabled/SQLite Null performs no I/O.

Focused result: **8 passed**, four existing Alembic path-separator deprecation
warnings. This is storage evidence, not answer-reuse, live model quality, cache
cost savings or production isolation/RBAC evidence. Formal release remains
NOT CLEAN and capability P1 is unchanged.
