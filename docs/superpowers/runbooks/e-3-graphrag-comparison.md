# E-3 GraphRAG comparison and rollback evidence

This is the cache-off fixed-corpus prototype measurement, not a production
quality or rollout gate. It uses E-1's `graph_projection_fixtures.py`, principal
`trusted-actor`, candidate cap 50, visible cap 5, fixed 1536-dimensional test
vectors, and no embedding or answer-model provider calls.

## Fixed-corpus result

The pgvector control is an actual E-1-equivalent fixed-vector ranked query;
its recorded ordered IDs are asserted. E-2's actual
`Neo4jEvidenceRetriever(db, settings, graph_store, seed_retriever)` then uses an
official-driver Neo4j store plus that actual ranked seed window. Public-source
metrics deduplicate only in this harness: `history_event:1` and `chunk:1` both
represent `gmail:trusted-1`; ordered serving IDs remain separate.

| Case | pgvector control IDs | graph ordered IDs | public source precision / recall |
|---|---|---|---|
| relation | `history_event:1`, `chunk:1` | `history_event:1`, `chunk:1`, `chunk:2` | 1.0 / 1.0 (control recall 0.5) |
| single | `chunk:3` | `chunk:3` | 1.0 / 1.0 |
| absent | none | none | no expected/retrieved sources; no fabricated evidence |
| restricted | `chunk:1` | `chunk:1` | 1.0 / 1.0 |
| revoke | `chunk:1` | `chunk:1` | 1.0 / 1.0 |

Relation coverage therefore adds the missing second approved source without
increasing the candidate or visible-evidence budget. Fixed vectors and controlled
seeds demonstrate retrieval behavior only; they do not measure live embedding,
model-answer faithfulness, citation quality, or paid cost.

## Timing sample

One standalone five-case run measured the retrieval portion after fixture setup
using `perf_counter_ns`; `n=5`, one measurement per case, no provider calls.
Raw `(relation, single, absent, restricted, revoke)` milliseconds were pgvector
baseline `(437.342, 59.468, 48.151, 232.486, 58.171)`, graph enrichment
overhead—using that precomputed seed—`(176.653, 5.815, 0.233, 5.990, 5.612)`,
derived graph total `(613.994, 65.282, 48.384, 238.476, 63.784)`, and projection
synchronization `(167.348, 161.109, 173.114, 111.084, 89.656)`. Nearest-rank
p50/p95 are pgvector `59.468/437.342`, enrichment overhead `5.815/176.653`,
derived total `65.282/613.994`, and sync `161.109/173.114`.
Every completed scope reported generation lag `0`; this tiny cold-fixture sample
is observability evidence, not an SLO or production latency claim.

## Repeatable checks

Run with process-only `PARAWORKS_TEST_POSTGRES_URL` and `PARAWORKS_TEST_NEO4J_*`
values:

```text
.venv-task4-r3-review/Scripts/python.exe -m pytest backend/tests/test_graph_projection_baseline.py backend/tests/test_neo4j_retriever.py backend/tests/test_graph_retrieval_comparison.py -q
```

`test_graph_retrieval_comparison.py` imports its own `lexical_pg` fixture and
therefore runs standalone. It verifies all five cases, actual default-off
composition, plus fake-store unavailable/stale fallback and receipt preservation.
Those fake failure modes are distinct from E-2's existing actual-driver
composition/outage coverage. When `PARAWORKS_TEST_NEO4J_RESTART_PAUSE=1`, it recreates the official
driver/store after the disposable Neo4j restart. Each test leases a PostgreSQL
schema and deletes its unique Neo4j scope.

The measured run used `pgvector/pgvector:pg17` and `neo4j:2026.08.1`; the eight
comparison/rollback tests passed in 5.79s. Review R1 initially reproduced eight
standalone setup errors because this module omitted `lexical_pg`; Ruff's unused
import autofix removed the first correction. The explicit self-alias import is
the R2 GREEN correction. The controller-coordinated restart
recovery passed in 25.74s, including test-only Neo4j readiness polling. Projection lag was
zero for the completed relation scope. A separate controlled PostgreSQL restart
reconstructed a fresh engine/session and reproduced the same pgvector control and
graph result (**1 passed, 7 deselected in 16.34s**); lease cleanup reconnects once
only for this disposable test outage. No paid providers, cache, rollout flag,
`.env` change, or deployment RBAC claim is included.

R2 code revision: `d307a2804c3aa9c8f955de386df8dd5ff12ad7d8`.
