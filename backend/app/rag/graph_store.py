"""Official Neo4j driver boundary for a disposable, rebuildable projection.

Provision constraints separately with schema-admin credentials. Normal workers
only require read/write on PwProjection, PwEvidence and SUPPORTED_BY. Community
test credentials do not demonstrate production least-privilege RBAC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from neo4j import unit_of_work


class GraphUnavailable(RuntimeError):  # noqa: N818 - internal adapter outcome
    pass


class StaleGraphGeneration(ValueError):  # noqa: N818 - internal fencing outcome
    pass


@dataclass(frozen=True, slots=True)
class GraphProjectionStatus:
    generation: int = -1
    cursor: int = 0
    scanned: bool = False
    complete: bool = False
    node_count: int = 0
    edge_count: int = 0
    generation_lag: int = 0
    lag_seconds: float = 0.0


class Neo4jGraphStore:
    def __init__(self, driver, *, database='neo4j'):
        self.driver = driver
        self.database = database

    def _transaction(self, callback, *, write=True):
        try:
            with self.driver.session(database=self.database) as session:
                method = session.execute_write if write else session.execute_read
                return method(unit_of_work(timeout=5.0)(callback))
        except StaleGraphGeneration:
            raise
        except Exception:
            raise GraphUnavailable('graph unavailable') from None

    def ensure_schema(self):
        def provision(tx):
            tx.run(
                'CREATE CONSTRAINT pw_projection_scope IF NOT EXISTS FOR (n:PwProjection) REQUIRE n.scope_id IS UNIQUE'
            ).consume()
            tx.run(
                'CREATE CONSTRAINT pw_evidence_identity IF NOT EXISTS FOR (n:PwEvidence) REQUIRE (n.scope_id, n.document_id) IS UNIQUE'
            ).consume()

        self._transaction(provision)

    def traverse(self, *, scope_id, generation, seeds, permissions, policy):
        from backend.app.rag.graph_projection import (
            GraphEdgeDependency,
            GraphNodeDependency,
            GraphPathDependency,
            GraphTraversalPolicy,
        )

        if type(policy) is not GraphTraversalPolicy:
            raise ValueError('graph traversal policy required')
        policy.__post_init__()
        if type(seeds) is not tuple or not 1 <= len(seeds) <= policy.seed_limit:
            raise ValueError('graph seed window outside bounds')

        def read(tx):
            rows = tx.run(
                """MATCH (s:PwProjection {scope_id:$scope})
                WHERE s.complete = true AND s.generation = $generation
                  AND s.completed_generation = $generation
                UNWIND $seeds AS seed
                MATCH (a:PwEvidence {scope_id:$scope, document_id:seed})
                      -[r:SUPPORTED_BY]-(b:PwEvidence {scope_id:$scope})
                WHERE r.scope_id = $scope AND a.generation = $generation
                  AND b.generation = $generation AND r.generation = $generation
                  AND a.permission IN $permissions AND b.permission IN $permissions
                  AND r.permission IN $permissions
                WITH DISTINCT startNode(r) AS left, r, endNode(r) AS right
                ORDER BY r.edge_id
                LIMIT $limit
                RETURN properties(left) AS left, properties(r) AS edge, properties(right) AS right
                """,
                scope=scope_id,
                generation=generation,
                seeds=list(seeds),
                permissions=list(permissions),
                limit=policy.candidate_limit,
            )

            def build(cls, value):
                return cls(**{f.name: value[f.name] for f in fields(cls)})

            return tuple(
                sorted(
                    (
                        GraphPathDependency(
                            (
                                build(GraphNodeDependency, r['left']),
                                build(GraphNodeDependency, r['right']),
                            ),
                            (build(GraphEdgeDependency, r['edge']),),
                        )
                        for r in rows
                    ),
                    key=lambda p: p.edges[0].edge_id,
                )
            )

        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(
                    unit_of_work(timeout=policy.timeout_seconds)(read)
                )
        except Exception:
            raise GraphUnavailable('graph unavailable') from None

    def status(self, scope_id, *, canonical_generation=None):
        def read(tx):
            record = tx.run(
                'MATCH (s:PwProjection {scope_id:$scope}) RETURN properties(s) AS state',
                scope=scope_id,
            ).single()
            if record is None:
                return GraphProjectionStatus(
                    generation_lag=max(0, canonical_generation or 0)
                )
            state = record['state']
            # Report stored bounded-work counts; no corpus-wide COUNT scan.
            generation = state['generation']
            lag = max(
                0,
                (
                    canonical_generation
                    if canonical_generation is not None
                    else generation
                )
                - state.get('completed_generation', 0),
            )
            age = tx.run(
                'MATCH (s:PwProjection {scope_id:$scope}) RETURN CASE WHEN s.complete THEN 0.0 ELSE (timestamp()-s.started_ms)/1000.0 END AS age',
                scope=scope_id,
            ).single()['age']
            return GraphProjectionStatus(
                generation,
                state['cursor'],
                state['scanned'],
                state['complete'],
                state.get('node_count', 0),
                state.get('edge_count', 0),
                lag,
                float(age),
            )

        return self._transaction(read, write=False)

    @staticmethod
    def _lock(tx, scope_id):
        return tx.run(
            'MERGE (s:PwProjection {scope_id:$scope}) ON CREATE SET s.generation=-1, s.cursor=0, s.scanned=false, s.complete=false, s.node_count=0, s.edge_count=0 SET s.fence=coalesce(s.fence,0)+1 RETURN properties(s) AS state',
            scope=scope_id,
        ).single()['state']

    def apply_page(self, page):
        if (
            len(page.nodes) > 5100
            or len(page.edges) > 5000
            or page.scanned_count > 100
            or page.generation < 0
            or page.cursor < page.start_cursor
            or any(node.scope_id != page.scope_id for node in page.nodes)
            or any(
                edge.scope_id != page.scope_id or edge.relation != 'SUPPORTED_BY'
                for edge in page.edges
            )
        ):
            raise ValueError('invalid graph projection page')

        def apply(tx):
            state = self._lock(tx, page.scope_id)
            if page.generation < state['generation']:
                raise StaleGraphGeneration('stale graph generation')
            if (
                page.generation == state['generation']
                and page.cursor <= state['cursor']
            ):
                # Replay after response loss; cursor and data committed together.
                return
            if page.generation > state['generation']:
                if page.start_cursor != 0:
                    raise StaleGraphGeneration('new generation requires initial cursor')
                tx.run(
                    'MATCH (s:PwProjection {scope_id:$scope}) SET s.generation=$generation, s.cursor=0, s.scanned=false, s.complete=false, s.started_ms=timestamp(), s.source_updated_at=$updated',
                    scope=page.scope_id,
                    generation=page.generation,
                    updated=page.source_updated_at,
                ).consume()
                state['cursor'] = 0
            if (
                page.start_cursor != state['cursor']
                or state.get('scanned')
                and page.generation == state['generation']
            ):
                raise StaleGraphGeneration('projection cursor conflict')
            node_result = tx.run(
                'UNWIND $nodes AS row MERGE (n:PwEvidence {scope_id:$scope, document_id:row.document_id}) ON CREATE SET n.new_marker=true WITH n,row,coalesce(n.new_marker,false) AS created SET n += row, n.generation=$generation REMOVE n.new_marker RETURN sum(CASE WHEN created THEN 1 ELSE 0 END) AS created',
                nodes=[asdict(n) for n in page.nodes],
                scope=page.scope_id,
                generation=page.generation,
            ).single()
            edge_result = tx.run(
                'UNWIND $edges AS row MATCH (a:PwEvidence {scope_id:$scope,document_id:row.from_id}), (b:PwEvidence {scope_id:$scope,document_id:row.to_id}) MERGE (a)-[r:SUPPORTED_BY {scope_id:$scope,edge_id:row.edge_id}]->(b) ON CREATE SET r.new_marker=true WITH r,row,coalesce(r.new_marker,false) AS created SET r += row, r.generation=$generation REMOVE r.new_marker RETURN sum(CASE WHEN created THEN 1 ELSE 0 END) AS created',
                edges=[asdict(e) for e in page.edges],
                scope=page.scope_id,
                generation=page.generation,
            ).single()
            tx.run(
                'MATCH (s:PwProjection {scope_id:$scope}) SET s.cursor=$cursor, s.scanned=$scanned, s.node_count=s.node_count+$nodes, s.edge_count=s.edge_count+$edges',
                scope=page.scope_id,
                cursor=page.cursor,
                scanned=page.complete,
                nodes=(node_result['created'] or 0),
                edges=(edge_result['created'] or 0),
            ).consume()

        self._transaction(apply)

    def sweep(self, scope_id, generation, *, batch_size=50):
        if type(batch_size) is not int or not 1 <= batch_size <= 100:
            raise ValueError('sweep batch outside bounds')

        def clean(tx):
            state = self._lock(tx, scope_id)
            if state['generation'] != generation or not state['scanned']:
                raise StaleGraphGeneration('sweep requires completed current scan')
            if state['complete']:
                return
            edges = tx.run(
                'MATCH ()-[r:SUPPORTED_BY {scope_id:$scope}]->() WHERE r.generation <> $generation WITH r LIMIT $limit DELETE r RETURN count(*) AS count',
                scope=scope_id,
                generation=generation,
                limit=batch_size,
            ).single()['count']
            # Delete nodes only once stale edges are drained: bounded work and
            # exact counters without unbounded DETACH DELETE fanout.
            nodes = 0
            if edges == 0:
                nodes = tx.run(
                    'MATCH (n:PwEvidence {scope_id:$scope}) WHERE n.generation <> $generation AND NOT (n)--() WITH n LIMIT $limit DELETE n RETURN count(*) AS count',
                    scope=scope_id,
                    generation=generation,
                    limit=batch_size,
                ).single()['count']
            complete = edges == 0 and nodes == 0
            tx.run(
                'MATCH (s:PwProjection {scope_id:$scope}) SET s.node_count=s.node_count-$nodes, s.edge_count=s.edge_count-$edges, s.complete=$complete, s.completed_generation=CASE WHEN $complete THEN s.generation ELSE s.completed_generation END, s.completed_ms=CASE WHEN $complete THEN timestamp() ELSE s.completed_ms END',
                scope=scope_id,
                nodes=nodes,
                edges=edges,
                complete=complete,
            ).consume()

        self._transaction(clean)
