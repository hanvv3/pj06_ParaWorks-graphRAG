import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.tests.graph_projection_fixtures import SCOPE, SETTINGS, seed_corpus
from backend.tests.postgres_isolation import lease_postgres_schema
from backend.tests.test_graph_projection import projection_module


class _CrashBeforeCursorDriver:
    """Inject a worker failure inside an actual Neo4j write transaction."""

    def __init__(self, driver):
        self.driver = driver

    def session(self, **kwargs):
        session = self.driver.session(**kwargs)

        class Transaction:
            def __init__(self, tx):
                self.tx = tx

            def run(self, query, **parameters):
                if 'SET s.cursor=$cursor' in query:
                    raise RuntimeError('simulated crash after graph writes')
                return self.tx.run(query, **parameters)

        class WrappedSession:
            def __enter__(self):
                session.__enter__()
                return self

            def __exit__(self, *args):
                return session.__exit__(*args)

            def execute_write(self, callback):
                return session.execute_write(lambda tx: callback(Transaction(tx)))

        return WrappedSession()


@pytest.fixture
def canonical_pg():
    url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('disposable PostgreSQL required')
    with lease_postgres_schema(
        url, run_id=uuid4().hex[:12], scope_name='e_graph'
    ) as lease:
        engine = create_engine(lease.database_url)
        try:
            Base.metadata.create_all(engine, checkfirst=False)
            with Session(engine) as db:
                yield db
        finally:
            engine.dispose()


def test_disposable_neo4j_restart_delete_revoke_and_generation_fences(canonical_pg):
    db_session = canonical_pg
    m = projection_module()
    uri = os.getenv('PARAWORKS_TEST_NEO4J_URI')
    if not uri:
        pytest.skip('disposable Neo4j required')
    from neo4j import GraphDatabase

    from backend.app.models import RagServingCorpusGeneration
    from backend.app.rag.graph_store import (
        GraphUnavailable,
        Neo4jGraphStore,
        StaleGraphGeneration,
    )
    from backend.app.rag.lexical_projection import refresh_rag_lexical_projections

    driver = GraphDatabase.driver(
        uri,
        auth=(
            os.environ['PARAWORKS_TEST_NEO4J_USER'],
            os.environ['PARAWORKS_TEST_NEO4J_PASSWORD'],
        ),
    )
    scope = replace(SCOPE, principal_subject='e-test-' + uuid4().hex)
    scope2 = replace(scope, principal_subject=scope.principal_subject + '-other')
    sources, chunks, target, approval = seed_corpus(db_session)
    store = Neo4jGraphStore(driver)
    store.ensure_schema()
    first = m.read_projection_page(
        db_session, settings=SETTINGS, scope=scope, cursor=0, batch_size=2
    )
    other = m.read_projection_page(
        db_session, settings=SETTINGS, scope=scope2, cursor=0, batch_size=100
    )
    try:
        with pytest.raises(GraphUnavailable):
            Neo4jGraphStore(_CrashBeforeCursorDriver(driver)).apply_page(first)
        assert store.status(first.scope_id).generation == -1
        assert (
            driver.execute_query(
                'MATCH (n:PwEvidence {scope_id:$scope}) RETURN count(n) AS count',
                scope=first.scope_id,
            ).records[0]['count']
            == 0
        )
        store.apply_page(first)
        if os.getenv('PARAWORKS_TEST_NEO4J_RESTART_PAUSE') == '1':
            print(
                f'RESTART_READY scope={first.scope_id} generation=1 cursor=2 nodes=2 edges=0',
                flush=True,
            )
            input('Restart only the disposable Neo4j container, then press Enter: ')
        # Restart the worker after a committed batch: durable cursor resumes.
        store = Neo4jGraphStore(driver)
        assert store.status(first.scope_id).cursor == first.cursor
        assert store.status(first.scope_id, canonical_generation=1).generation_lag == 1
        store.apply_page(first)  # delivery replay is idempotent
        with pytest.raises(StaleGraphGeneration):
            store.apply_page(
                replace(first, start_cursor=first.cursor + 1, cursor=first.cursor + 2)
            )
        for _ in range(12):
            result = m.reconcile_graph_step(
                db_session, settings=SETTINGS, scope=scope, store=store, batch_size=2
            )
            db_session.commit()
            if result.complete:
                break
        assert result.complete and result.node_count == 4 and result.edge_count == 2
        assert result.generation_lag == 0
        store.apply_page(other)
        store.sweep(other.scope_id, 1, batch_size=100)
        approval.active = False
        target.review_status = 'revoked'
        db_session.delete(chunks[2])
        generation = db_session.get(RagServingCorpusGeneration, 1)
        generation.corpus_generation = 2
        db_session.flush()
        refresh_rag_lexical_projections(
            db_session, settings=SETTINGS, corpus_generation=2
        )
        db_session.commit()
        for _ in range(12):
            result = m.reconcile_graph_step(
                db_session, settings=SETTINGS, scope=scope, store=store, batch_size=1
            )
            db_session.commit()
            if result.complete:
                break
        assert result.complete and result.node_count == 2 and result.edge_count == 0
        assert store.status(other.scope_id).node_count == 4
        with pytest.raises(StaleGraphGeneration):
            store.apply_page(first)
        # Endpoint same, source content changes: new canonical version replaces node.
        old_version = driver.execute_query(
            'MATCH (n:PwEvidence {scope_id:$s, document_id:"chunk:1"}) RETURN n.version AS v',
            s=first.scope_id,
        ).records[0]['v']
        sources[0].permission_level = chunks[0].permission_level = 'public'
        generation.corpus_generation = 3
        db_session.flush()
        refresh_rag_lexical_projections(
            db_session, settings=SETTINGS, corpus_generation=3
        )
        db_session.commit()
        for _ in range(12):
            result = m.reconcile_graph_step(
                db_session, settings=SETTINGS, scope=scope, store=store, batch_size=2
            )
            db_session.commit()
            if result.complete:
                break
        new_version = driver.execute_query(
            'MATCH (n:PwEvidence {scope_id:$s, document_id:"chunk:1"}) RETURN n.version AS v',
            s=first.scope_id,
        ).records[0]['v']
        assert result.complete and new_version != old_version
        # A new source revision invalidates the old parsed bundle even when no
        # deletion timestamp/tombstone changes. Reconciliation must remove it.
        sources[0].server_content_signature = 'f' * 64
        generation.corpus_generation = 4
        db_session.flush()
        refresh_rag_lexical_projections(
            db_session, settings=SETTINGS, corpus_generation=4
        )
        db_session.commit()
        for _ in range(12):
            result = m.reconcile_graph_step(
                db_session, settings=SETTINGS, scope=scope, store=store, batch_size=2
            )
            db_session.commit()
            if result.complete:
                break
        assert result.complete and result.node_count == 1 and result.edge_count == 0
    finally:
        driver.execute_query(
            'MATCH (n) WHERE (n:PwEvidence OR n:PwProjection) AND n.scope_id IN $scopes DETACH DELETE n',
            scopes=[first.scope_id, other.scope_id],
        )
        driver.close()
