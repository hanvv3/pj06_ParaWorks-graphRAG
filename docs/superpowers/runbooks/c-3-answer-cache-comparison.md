# C-3 measured generation reuse and rollback

`backend/tests/test_rag_answer_cache_comparison.py` reuses C-2's actual PostgreSQL
composition fixture. It executes real LangGraph retrieval, canonical authority,
cost/advisory connections, cache storage and finalization. The external model and
Neo4j traversal are fake. This is a keyword-seed + graph-path comparison, not a
live Neo4j or pgvector benchmark, model-quality evaluation or provider bill.

## Reproduce and measurement boundary

Supply a disposable local `PARAWORKS_TEST_POSTGRES_URL` through the process
environment. Run from this worktree using the working interpreter:

```text
.venv-task4-r3-review/Scripts/python.exe -m pytest backend/tests/test_rag_answer_cache_comparison.py -q -s --tb=short --basetemp=.tmp/c3-new-run -o cache_dir=.tmp/c3-new-cache
```

Use fresh temporary names. The fixture leases its own schema and disposes/drops
it on exit. No credential file, rollout flag, scheduler or paid provider is used.
The `C3_MEASUREMENTS` JSON contains all request samples, run/cache-audit IDs,
component usage, retrieval counts, cursor counts/time and wall time.

Three rounds execute two cache-off baseline requests and one cold/warm pair.
Only cache entries are cleared between rounds, outside the timed region. Corpus,
exact principal (including within each pair), permissions, question, prompt,
model and retrieval budgets stay fixed. Both baseline and cached groups use the
same effective `neo4j` backend. SQL corpus/scorer setup, provider-safety bootstrap,
fake path construction and post-request verification reads are outside timing.
No retrieval result or final answer is precomputed for the measured request.

Wall time starts before request-service assembly and includes graph construction,
text preparation, fresh retrieval, cache cleanup/lookup, generation or reuse,
durable finalization and service teardown. Engine-class SQLAlchemy cursor hooks
cover application and dedicated cost/advisory/finalizer engines. SQL count includes
SET and cursor executions; it excludes driver-level BEGIN/COMMIT, connection
handshakes and row fetching. Those remain in wall time. SQL milliseconds measure
cursor execution only, not total DB or network time. p50/p95 use nearest rank;
three cold and three warm samples cannot establish a latency SLO.

## Results

| Group | n | Gen calls | Hit rate | Simulated USD | Wall p50 / p95 ms | SQL count p50 / p95 | SQL p50 / p95 ms |
|---|---:|---:|---:|---:|---|---|---|
| baseline | 6 | 6 | 0% | 0.000180 | 5937.132 / 6333.62 | 3205 / 3205 | 1730.405 / 1766.479 |
| cold | 3 | 3 | 0% | 0.000090 | 6242.096 / 6253.183 | 3214 / 3214 | 1769.076 / 1774.069 |
| warm | 3 | 0 | 100% | 0.000000 | 7271.826 / 7625.794 | 2696 / 2696 | 1439.778 / 1443.385 |
| cold + warm | 6 | 3 | 50% | 0.000090 | 6253.183 / 7625.794 | 2696 / 3214 | 1443.385 / 1774.069 |

For six identical requests, caching reduced generation calls 6→3 and simulated
cost USD 0.000180→0.000090 (50%). Warm requests individually skipped generation.
All 16 requests made two real keyword retriever calls (initial retrieval and
finalization retrieval); query embedding dispatches stayed zero. Citation HMACs
matched across each baseline/cold/warm comparison.

Warm wall time was **slower** here despite fewer SQL cursor executions. The fake
provider has no remote network or generation wait, and the request includes
safety/bootstrap/advisory and finalization overhead. These samples establish no
live latency improvement, performance SLO, bottleneck attribution or need for
Redis L2. No cache-only timing breakdown was collected.

Raw samples from the final execution (2026-09-21 local date):

| Request / run | Cache audit IDs | Backend | Hit | Gen calls | Wall ms | SQL count | SQL ms |
|---|---|---|---:|---:|---:|---:|---:|
| baseline / 1 | — | neo4j | 0 | 1 | 6006.639 | 3205 | 1753.755 |
| baseline / 2 | — | neo4j | 0 | 1 | 5937.132 | 3205 | 1730.405 |
| cold / 3 | — | neo4j | 0 | 1 | 5961.552 | 3214 | 1741.196 |
| warm / 4 | 1, 2 | neo4j | 1 | 0 | 7271.826 | 2696 | 1443.385 |
| baseline / 5 | — | neo4j | 0 | 1 | 6333.62 | 3205 | 1745.748 |
| baseline / 6 | — | neo4j | 0 | 1 | 6027.223 | 3205 | 1766.479 |
| cold / 7 | — | neo4j | 0 | 1 | 6242.096 | 3214 | 1774.069 |
| warm / 8 | 3, 4 | neo4j | 1 | 0 | 7205.689 | 2696 | 1430.251 |
| baseline / 9 | — | neo4j | 0 | 1 | 5746.403 | 3205 | 1704.429 |
| baseline / 10 | — | neo4j | 0 | 1 | 5810.853 | 3205 | 1705.058 |
| cold / 11 | — | neo4j | 0 | 1 | 6253.183 | 3214 | 1769.076 |
| warm / 12 | 5, 6 | neo4j | 1 | 0 | 7625.794 | 2696 | 1439.778 |
| cache_off_rollback / 13 | — | neo4j | 0 | 1 | 5755.607 | 3205 | 1678.616 |
| graph_off_cold / 14 | — | deterministic_lexical | 0 | 1 | 5170.685 | 2247 | 1210.039 |
| graph_off_warm / 15 | 7, 8 | deterministic_lexical | 1 | 0 | 6454.226 | 1848 | 978.475 |
| graph_restored_warm / 16 | 9, 10 | neo4j | 1 | 0 | 7301.417 | 2696 | 1443.021 |

Fake generation reports 10 input / 5 output tokens, priced by the existing runtime
policy. A cold/baseline generation is a simulated USD 0.000030; a warm generation
is undispatched with zero usage/charge. `charge_basis=actual` refers to parsed fake
usage in the durable ledger, not a real provider charge. Keyword seed performs no
query embedding, so this comparison proves no embedding saving. C-2's unchanged
paid-embedding case separately verifies a fresh fake embedding call and its
USD 0.000001 simulated charge on the warm path.

Every request has a fresh durable final AgentRun. Every hit has distinct
`rag_answer_cache_hit` and `rag_answer_cache_finalized` AuditLog rows; the reported
cache-specific audit list is empty on cold/baseline requests and says nothing
about other audit mechanisms. Baseline/cold/warm citation HMACs must match.

## Rollback and inherited safety evidence

Final combined comparison, C-2 PG integration, API delivery, Assistant evidence/
delivery and secret-hygiene selection: **113 passed in 339.78s**, zero failures,
skips or warnings. Test files were unchanged during/after this execution.

After warming, cache-off executes a fresh graph generation. Graph-off with cache
enabled misses the graph entry, returns `deterministic_lexical`, then warms its
own key. Restoring graph returns its original warm entry. The two keys coexist,
and every pre-rollback audit row retains its action, target and metadata.

C-3 reruns the directly affected C-2 integration tests, including same-role
different principals, Assistant context/input order, store safe misses, exact
expiry at finalization, selected/nonselected influence revocation, authority
failure, real-PG edge/node drift and newly added evidence. C-1 key/TTL tests are
unchanged: their previously recorded 8-pass evidence covers principal/workspace/
permission and prompt/model/output/policy/key/backend identities, non-sliding
TTL, expiry during read, unavailable store and full cache migration rollback.
Those C-1 results are inherited evidence, not a new C-3 execution.

The PG composition schema uses E/C-2 metadata plus real lexical SQL scorers;
it does not include the full migration-trigger deployment. Production RBAC,
actual-model quality/billing, pgvector full-path timing and deployment-scale
concurrency remain unmeasured. All runtime defaults stay off. Formal release
remains NOT CLEAN and capability P1 remains deferred. After independent C-3
review, the next task is Slack synthetic S-1.
