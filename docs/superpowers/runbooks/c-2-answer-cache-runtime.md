# C-2 answer-cache runtime integration

The default-off `rag_answer_cache_enabled` setting connects C-1 to the existing
LangGraph. SQLite remains Null/no-write. Public Ask/Search/Assistant payloads,
Review trust, provider readiness and paid-miss ceilings are unchanged.

## State and authority

Fresh retrieval, canonical permission filtering and full rendered preparation
precede lookup. The runtime key includes the exact principal/scope, all ordered
model influences and E graph paths, input HMAC, Assistant retrieval-context HMAC,
model/prompt/output/cost-policy identity and actual seed/effective graph backend.

An authenticated hit supplies substantive validated blocks without provider
dispatch or a fabricated provider receipt. The existing durable admission and
prepared-input binding remain in place. `commit_answer_cache_pending` verifies
the hit, compares its scope/preparation to the parent, then uses the existing
owner-fenced pending transition to close generation at exact terminal zero.
Query embedding keeps its actual usage/charge and paid safety binding, if used.
The same transaction records a `rag_answer_cache_hit` audit. This record survives
a later refusal or authority outage; it is not publication authorization.

`PreparedRagFinalization.answer_cache_hit` selects `answer-cache-hit:v1`.
Finalization reconstructs the complete key from current request preparation,
reauthenticates the value and current expiry, and compares durable parent
identities. Its existing C.5/owner/canonical PostgreSQL transaction revalidates
every selected and nonselected influence and complete ordered graph path, then
rebuilds citations. Successful or redacted publication records a separate
`rag_answer_cache_finalized` audit. A failed authority read yields no product.
The cache result has its own `rag-result-cache-hit:v1` HMAC domain; ordinary
generation retains `rag-result:v1`. Assistant still stores substantive answers
as `rag_assembled`, with its existing exact-content and evidence checks.

Only successfully committed substantive generations are inserted. Cache I/O
faults are misses/unsuccessful writes. Canonical authority failures are never
caught as storage misses. No prompts, questions, citations or old receipts are
added to cache storage. C-1's table and TTL remain unchanged; no migration or
public DTO changes are needed. Existing pending recovery remains fail-closed
and cannot redispatch a cached generation.

## Cleanup and rollback

An enabled PostgreSQL request performs one bounded cleanup of at most 100
expired rows in a separate short transaction before admission. This traffic
cleanup is opportunistic, not a scheduling guarantee. Operators should also
schedule `python -m backend.scripts.cleanup_answer_cache --limit 100` with the
deployment's process environment, independently of traffic, at least hourly.
Repeat bounded batches until a successful exit reports `deleted_count=0`;
monitor database failures and backlog. The operator command uses strict cleanup:
database/DELETE failure exits 1, emits only a sanitized error on stderr and no
deleted count. Request cleanup remains best effort. The command can clean expired entries while the runtime flag is off,
does not activate it, and performs no SQLite I/O. No scheduler or flag was
activated by C-2. TTL rejects expired reads even during cleanup outages.

Setting the flag off restores fresh E retrieval/generation. Historical run and
audit rows are preserved. Rollback needs no table downgrade.

## Verification boundary

`test_rag_answer_cache_integration.py` runs the real graph and canonical
validators. The smaller cases use the existing fake synchronization ledger
fixtures with actual PostgreSQL cache I/O. The production-composition cases use
real PostgreSQL application/cost/cache/canonical rows, dedicated advisory
connections, provider safety and finalization, plus E's complete graph paths.
Only the external model and Neo4j traversal are fake in those cases. They use
E's isolated metadata schema plus real lexical scorer functions, not the full
migration-trigger schema; this is not migration, production RBAC, live model
quality or formal release acceptance. C-1 retains separate migration evidence.

The actual composition path exposed a pre-existing committed-read mismatch:
dedicated bootstrap transports had already read their database identity before
registry capability lookup. Three narrowly owned connection loaders now end
only those validation reads before the registry's unchanged fresh-read gate.
Application/borrowed transactions and lock/owner policy are unchanged.

Exact fresh selectors/results and code revisions are recorded in the C-2 task
report and portfolio. Formal release remains NOT CLEAN; capability P1 remains
deferred. No live providers, rollout, secret files or remote push were used.
