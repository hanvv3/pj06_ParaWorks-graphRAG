# E-2 permission-preserving relationship retrieval

`rag_graph_enrichment_enabled` is frozen and defaults to false. When enabled,
the PostgreSQL runtime and its fresh finalizer wrap the configured keyword or
pgvector Runnable with `Neo4jEvidenceRetriever`. Supply `rag_neo4j_uri`,
`rag_neo4j_username`, `rag_neo4j_password`, and optionally `rag_neo4j_database`
through process configuration/secret management. Never commit credentials.
The SQLite smoke runtime rejects enabled graph configuration; its default
provider-free path is unchanged. No route, public citation, Review rule, or paid
budget changed. Turning the flag off restores the prior retriever assembly.

## Retrieval and budget

The configured backend remains the seed backend (`keyword`/`pgvector`) so the
existing paid embedding preflight, reservation, receipt and fallback remain
authoritative. Effective backend becomes `neo4j` only when an exactly validated
relationship adds evidence. No classifier, model, embedding or cache is added.
One invocation reuses the seed result once; graph failure never invokes it again.

The graph query is fixed one-hop `SUPPORTED_BY`, parameterized by scope, current
generation, at most two seeds, permissions and remaining candidate budget.
Only complete/current projections for the exact principal/scope are consumed.
The transaction timeout is one second; production drivers have one-second
connection/acquisition timeouts and zero automatic retry duration. Drivers close
after each traversal. Graph rows include identifiers/provenance, never model text.

The combined trace has at most 50 candidate slots, with graph path proposals
charged conservatively against `50 - seed.candidate_window_count`; this is not
50 graph candidates after 50 seed candidates. Visible/evidence count is capped
at five. Permission-filtered graph paths do not produce public edge counts;
the seed's bounded hidden-match count is retained. Outage, stale/incomplete
projection and no improvement reuse seed evidence under these same caps.
Authority read errors propagate closed instead of allowing graph data through.

## Internal contract and consumers

`RetrievalResult.graph_paths` is an ordered tuple of the E-1
`GraphPathDependency` v1, default empty for ordinary keyword/pgvector results.
`graph_policy_version` distinguishes enabled enrichment from the disabled seed
path, including fallback. Graph candidate windows have their own v1 HMAC domain.
Final hidden-membership HMAC is v2 and includes that policy identity.

Graph state already carries `RetrievalResult`; it now forwards all influencing
paths, including evidence not selected as citations, to
`PreparedModelInfluenceSet.graph_paths` and its request scope. The prepared-set
fingerprint is v3 and binds ordered path payloads. The public answer schema,
child observation fingerprint and persisted citation shape do not change.
There is no migration or durable cache; old in-flight v2 prepared sets fail
authentication rather than silently dropping dependencies.

At preparation, at the actual C.5 provider-send boundary, and at final exposure,
PostgreSQL re-runs the bounded E-1 canonical reconstruction for the path's approved
knowledge IDs and compares every ordered node and edge, including exact approval,
review pair, child link/version/signature and effective permission. The graph
cannot create citations or trust. The existing corpus/provider barrier remains
in force; canonical read failures are not graph availability failures.

The ledger retains the immutable graph prepared set in request-local binding
after the normal durable answer-budget commit. Before send it authenticates the
same aggregate and rendered-input HMAC against the durable parent, scope
fingerprint and existing policy verifier under the C.5 lock. Missing/changed
bindings fail closed. This is a dependency carrier, not a new execution authority.
`PreparedRagFinalization` requires identical retrieval/prepared paths, compares
fresh retrieval dependencies, then invokes canonical dependency revalidation.
Changed unselected edges suppress the whole generated answer and selected
citations while preserving the original actual-or-reserve cost accounting.

Future D.1 must reuse graph policy plus ordered path identity for cache keys,
pre-send checks and final exposure; a citation-only cache dependency is inadequate.

## Verification scope

`backend/tests/test_neo4j_retriever.py` covers canonical relationship enrichment,
restriction/revoke/delete/version/scope rejection, authority errors, shared bounds,
receipt reuse, production composition/default off, actual LangGraph fake-provider
normal/pre-send/post-send outcomes and missing/changed binding rejection.
Those full graph tests use the existing SQLite synchronization test ports with
real cost rows/model preparation; they are not PostgreSQL synchronization evidence.

Separate leased PostgreSQL tests use the actual lexical scorer migration,
canonical reconstruction, provider-free final influence validation and disposable
Neo4j official-driver traversal. Each drops its unique PG schema and graph scope.
The actual PG fallback reproduced psycopg's malformed tuple-array literal; only
the SQL execution boundary now adapts typed array binds as lists.

Set the process-only disposable connection variables documented in the E-1
runbook and run the E-2 test plus existing projection, default-runtime, graph,
cost-policy/ledger/provider, finalization and keyword/pgvector suites with a fresh
`.tmp` basetemp/cache. Missing DB settings skip, not prove, integration coverage.
No live provider calls, restart exercise or deployment RBAC claim is part of E-2.
E-3 owns fixed-corpus comparison/publication. Formal release remains NOT CLEAN.
