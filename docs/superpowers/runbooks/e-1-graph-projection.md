# E-1 canonical graph projection

The graph is a rebuildable, disabled-by-default PostgreSQL projection. No API,
runtime backend, trust policy, Review Queue or provider budget has changed.
`reconcile_graph_step` is the internal admin-job entry point: call one step in
one fresh SQLAlchemy transaction, commit it, and schedule another until complete.
Supply an explicit validated `SecurityScope`; the existing scope fingerprint is
the graph namespace, including principal, workspace, constraints and permissions.
E-2 must only consume a complete projection of the current generation for the
same scope and must independently revalidate every dependency against PG.

## Relations and E-2 carrier

Only `SUPPORTED_BY` is allowed: an explicit live approval's selected canonical
evidence link connects approved knowledge to the lowest-id current source chunk
whose URL/snippet exactly matches the review evidence pair. All children must
resolve and be visible. Legacy human knowledge can be a seed, but has no inferred
edge. Project similarity, shared names, AI relations and graph text are excluded.

`GraphNodeDependency` preserves canonical serving ID/version, effective permission,
public source ID, scope fingerprint and the complete text-free serving version
envelope. `GraphEdgeDependency` preserves ordered endpoints, approval/review/child
IDs, exact approval provenance, child source signature, endpoint versions and a
stable SHA-256 version over these references. The hash is change detection, never
permission/approval authority. `GraphPathDependency` v1 is ordered node/edge/node,
one hop; E-2 must reconstruct that same relationship in PostgreSQL before provider
transmission and final exposure, including all influencing paths. Current
`RetrievalResult`, graph state and finalization do not yet carry it; E-2 owns those
consumer changes. Public citation/response contracts stay unchanged.

## Fixed synthetic corpus and cache-off baseline

Fixture: `backend/tests/graph_projection_fixtures.py`. Principal `trusted-actor`,
workspace `workspace-a`, public/internal permissions; no source/project constraints.
All IDs below are local synthetic fixture IDs, not production data.

| Identity | Fixed canonical reference |
|---|---|
| `history_event:1` | human approval 1, review 1, evidence links 1 and 2 |
| `chunk:1`, `gmail:trusted-1` | source/document/version/parser 1, signature hex `1` padded to 64 |
| `chunk:2`, `gmail:trusted-2` | source/document/version/parser 2, signature hex `2` padded to 64 |
| `chunk:3`, `gmail:trusted-3` | source/document/version/parser 3, signature hex `3` padded to 64 |

The expected relationship evidence is source 1 + source 2; both approved
`SUPPORTED_BY` links remain necessary even though the decision cites source 1.
Single-evidence expects only source 3; absent expects nothing. Restricted changes
source/chunk 2 to restricted and excludes the approved decision and source 2;
revoke deactivates approval 1. Both keep source 1 as an independent observation,
without exposing the decision as an approved relation answer.

Before projection implementation, actual PostgreSQL/pgvector with 1536-dimensional
fixed basis vectors and the existing canonical search adapter returned:

| Question | Cache-off pgvector serving IDs |
|---|---|
| relation | `history_event:1`, `chunk:1` (supporting source 2 missed) |
| single | `chunk:3` |
| absent | empty |
| restricted | `chunk:1` |
| revoke | `chunk:1` |

Candidate cap 50, visible/evidence cap 5, relevance threshold 0.25; embeddings are
precomputed deterministic test vectors, provider/embedding calls 0, no answer model
or cache. This establishes fixture retrieval behavior, not real embedding/model
quality. E-2/E-3 must reuse these budgets/corpus and compare graph retrieval before
claiming improvement. The initial graph policy is at most 2 reused seeds, 1 hop,
50 candidates, 5 visible evidence items, 1 second traversal. E-1 only defines the
policy; E-2 implements traversal and measures quality.

## Bounded reconciliation and operational boundary

Read at most 100 lexical serving rows per page with a stable integer cursor and
shared PG corpus generation lock. Canonical resolvers recheck current visibility
and provenance. Sources with more than 100 current chunks or knowledge with more
than 50 active approval/evidence children are conservatively omitted before the
unbounded canonical resolvers run; this is a prototype projection limit, not a
new trusted-knowledge rule. Full generation scans plus bounded stale edge/node
sweeps discover deletion even without updated timestamps/tombstones. Existing
canonical tombstone/revoke/supersede eligibility remains authoritative.

Each Neo4j transaction locks one unique scope state, checks generation and cursor,
upserts stable node/edge identities, and advances cursor atomically. Retrying a
committed page is idempotent. A new generation restarts at cursor zero; older
generations and out-of-order cursor writes are rejected. Sweeps delete at most
100 edges or isolated nodes per transaction. Stored counts avoid full graph
count scans; generation lag uses the last fully swept generation and elapsed
scan time is reported separately. Incomplete generations cannot serve E-2.

PG statements have a 5-second timeout and 2-second lock timeout; Neo4j transaction
callbacks have a 5-second timeout. Configure supplied official drivers with short
connection/acquisition timeouts and bounded retry time for the hosting job. The
step interface intentionally does not create a driver or manage credentials.
Graph failure propagates as a sanitized internal error. Existing RAG never
imports/calls this job and remains independent of Neo4j availability.

Provision unique `PwProjection(scope_id)` and
`PwEvidence(scope_id,document_id)` constraints using separate schema-admin
credentials through `ensure_schema`; routine workers need only the projection
labels/relation. Neo4j Community disposable tests use its available administrator
account and are not evidence of production fine-grained RBAC. No credentials or
`.env` changes are committed.

Official SDK `neo4j` 6.3.1 is locked, with no `neo4j-graphrag`, LLM or second vector
framework. Official references checked at implementation:
[driver compatibility](https://neo4j.com/docs/python-manual/current/install/),
[transaction API](https://neo4j.com/docs/api/python-driver/current/api.html), and
[psycopg list/array adaptation](https://www.psycopg.org/psycopg3/docs/basic/adapt.html#lists-adaptation).
The actual baseline exposed tuple array bindings in pgvector; its three SQL
array parameters now bind lists. The keyword equivalent remains a specific E-2
fallback check, not a verified fix in this slice.

## Verification commands

Use `.venv-task4-r3-review/Scripts/python.exe -m pytest` with fresh `.tmp` basetemp
and cache directory. Set disposable-only connection environment variables:
`PARAWORKS_TEST_POSTGRES_URL`, `PARAWORKS_TEST_NEO4J_URI`,
`PARAWORKS_TEST_NEO4J_USER`, `PARAWORKS_TEST_NEO4J_PASSWORD`.
Run `test_graph_projection.py`, `test_graph_projection_baseline.py`, and
`test_graph_projection_neo4j.py`. Missing DB settings skip integration evidence;
they are not a pass. PG tests lease/drop unique schemas and Neo4j tests clean only
their unique scope namespaces. Optional `PARAWORKS_TEST_NEO4J_RESTART_PAUSE=1`
with pytest `-s` pauses after a committed page: restart the exact disposable
server, resume, then verify persisted cursor, full convergence and cleanup.

Formal release remains NOT CLEAN/deferred; paid model quality and E-2/E-3 adapter,
full-path finalization, outage fallback and real retrieval improvement are pending.
