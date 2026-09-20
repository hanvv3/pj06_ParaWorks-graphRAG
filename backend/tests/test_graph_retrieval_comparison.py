"""E-3 fixed-corpus GraphRAG comparison and recovery evidence.

This intentionally evaluates public-source coverage separately from ordered
serving IDs: approved knowledge and its raw source can share one public ID.
"""

import json
import os
import time
from dataclasses import replace
from uuid import uuid4

import pytest
from neo4j import GraphDatabase
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.rag.embeddings import validate_query_embedding_vector
from backend.app.rag.evaluation import evaluate_retrieval_matches
from backend.app.rag.graph_projection import reconcile_graph_step
from backend.app.rag.graph_store import Neo4jGraphStore
from backend.app.rag.indexing import build_rag_v2_index_documents
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.rag.retrieval import (
    QueryEmbeddingReceipt,
    RetrievalCandidate,
    RetrievalResult,
    SanitizedRetrievalTrace,
)
from backend.app.rag.search_store import SqlAlchemyPgVectorSearchStore
from backend.tests.graph_projection_fixtures import (
    CASES,
    SCOPE,
    SETTINGS,
    seed_corpus,
    vector,
)
from backend.tests.test_graph_projection_neo4j import canonical_pg  # noqa: F401
from backend.tests.test_neo4j_retriever import (
    adapter,
    request,
)
from backend.tests.test_neo4j_retriever import (
    lexical_pg as lexical_pg,
)


def _driver():
    uri = os.getenv('PARAWORKS_TEST_NEO4J_URI')
    if not uri:
        pytest.skip('disposable Neo4j required')
    return GraphDatabase.driver(
        uri,
        auth=(
            os.environ['PARAWORKS_TEST_NEO4J_USER'],
            os.environ['PARAWORKS_TEST_NEO4J_PASSWORD'],
        ),
        max_transaction_retry_time=0,
    )


def _complete_projection(db, store, scope):
    for _ in range(12):
        status = reconcile_graph_step(db, settings=SETTINGS, scope=scope, store=store)
        db.commit()
        if status.complete:
            return status
    pytest.fail('fixed corpus projection did not converge')


def _case_request(scope, case):
    """Same case-specific request object feeds both pgvector and graph arms."""
    return replace(request(scope), retrieval_query_text=CASES[case]['query'])


def _source_ids(result):
    # Trusted serving identity is intentionally `history_event:1`; it cites the
    # same public source as chunk:1.  Normalize this fixture-only alias here,
    # never in production ranking or RetrievalResult.
    public_ids = (
        'gmail:trusted-1'
        if candidate.evidence.serving_document_id == 'history_event:1'
        else candidate.evidence.public_source_id
        for candidate in result.visible
    )
    return tuple(dict.fromkeys(public_ids))


def _comparison_metrics(expected, result):
    return evaluate_retrieval_matches(
        expected_source_ids=expected,
        retrieved_source_ids=_source_ids(result),
        k=5,
    )


def _pgvector_seed(db, request_value, case, *, populate=True):
    """Run E-1's actual fixed-vector pgvector control before graph enrichment."""
    store = PgVectorStore(session=db, config=PgVectorConfig(embedding_dimensions=1536))
    if populate:
        store.ensure_schema()
        for document in build_rag_v2_index_documents(db, settings=SETTINGS):
            axis = {
                'history_event:1': 0,
                'chunk:1': 0,
                'chunk:2': 1,
                'chunk:3': 2,
            }[document.document_id]
            db.execute(
                text(
                    'INSERT INTO rag_vector_documents (document_id, text, source_url, source_snippet, permission_level, metadata_json, embedding) VALUES (:id, :body, :url, :snippet, :permission, CAST(:metadata AS jsonb), CAST(:embedding AS vector))'
                ),
                {
                    'id': document.document_id,
                    'body': document.text,
                    'url': document.source_url,
                    'snippet': document.source_snippet,
                    'permission': document.permission_level,
                    'metadata': json.dumps(document.metadata),
                    'embedding': json.dumps(vector(axis)),
                },
            )
        db.commit()
    started = time.perf_counter_ns()
    query = vector(2 if case == 'single' else 3 if case == 'absent' else 0)
    rows = SqlAlchemyPgVectorSearchStore(
        db=db,
        store=PgVectorStore(session=db, config=store.config, settings=SETTINGS),
        settings=SETTINGS,
    ).search(
        request_value,
        validate_query_embedding_vector(query, expected_dimensions=1536),
    )
    visible = tuple(
        RetrievalCandidate(
            evidence=row.evidence,
            relevance_score=row.relevance_score,
            matched_terms=row.matched_terms,
        )
        for row in rows
        if row.access.permission_visibility == 'visible'
    )[:5]
    assert (
        tuple(c.evidence.serving_document_id for c in visible)
        == CASES[case]['baseline_ids']
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return RetrievalResult(
        configured_backend='pgvector',
        effective_backend='pgvector',
        visible=visible,
        hidden_match_count=0,
        hidden_count_capped=False,
        top_candidate_window_hmac='a' * 64,
        query_embedding_receipt=None,
        trace=SanitizedRetrievalTrace(
            candidate_window_count=len(rows),
            visible_count=len(visible),
            hidden_match_count=0,
            provider_attempt_count=0,
            latency_ms=round(elapsed_ms),
            fallback_category=None,
        ),
    ), elapsed_ms


@pytest.mark.parametrize('case', tuple(CASES))
def test_e3_fixed_corpus_compares_ordered_ids_and_public_source_coverage(
    lexical_pg, case
):
    """Actual PG + official Neo4j driver, deterministic vectors and no provider."""
    db = lexical_pg
    # A unique principal isolates each disposable Neo4j namespace.  It retains
    # E-1's workspace, permission set and constraints, and is identical across
    # pgvector and graph arms of this one case.
    scope = replace(SCOPE, principal_subject=f'e3-{case}-{uuid4().hex}')
    case_request = _case_request(scope, case)
    seed_corpus(db, case)
    driver = _driver()
    store = Neo4jGraphStore(driver)
    scope_id = security_scope_fingerprint(scope, settings=SETTINGS)
    try:
        store.ensure_schema()
        sync_started = time.perf_counter_ns()
        status = _complete_projection(db, store, scope)
        sync_ms = (time.perf_counter_ns() - sync_started) / 1_000_000
        baseline, pgvector_ms = _pgvector_seed(db, case_request, case)
        from langchain_core.runnables import RunnableLambda

        graph_started = time.perf_counter_ns()
        result = adapter(db, store, RunnableLambda(lambda _: baseline)).invoke(
            case_request
        )
        graph_enrichment_overhead_ms = (
            time.perf_counter_ns() - graph_started
        ) / 1_000_000
        print(
            f'E3_TIMING case={case} n=1 pgvector_ms={pgvector_ms:.3f} '
            f'graph_enrichment_overhead_ms={graph_enrichment_overhead_ms:.3f} '
            f'graph_total_ms={pgvector_ms + graph_enrichment_overhead_ms:.3f} '
            f'sync_ms={sync_ms:.3f} '
            f'generation_lag={status.generation_lag}',
            flush=True,
        )
        expected = CASES[case]['expected_sources']
        metrics = _comparison_metrics(expected, result)
        ordered_ids = tuple(c.evidence.serving_document_id for c in result.visible)

        # Keep the E-1 pgvector ordering as the cache-off control.  The
        # production Runnable below is the E-2 graph-enriched retrieval path.
        if case == 'absent':
            assert not ordered_ids and not _source_ids(result)
            assert metrics.expected_count == metrics.retrieved_count == 0
        elif case == 'relation':
            assert ordered_ids == ('history_event:1', 'chunk:1', 'chunk:2')
            assert _source_ids(result) == ('gmail:trusted-1', 'gmail:trusted-2')
            assert metrics.precision_at_k == metrics.recall_at_k == 1.0
            assert status.generation_lag == 0
            if os.getenv('PARAWORKS_TEST_NEO4J_RESTART_PAUSE') == '1':
                print(
                    f'E3_RESTART_READY scope={scope_id} ordered={ordered_ids}',
                    flush=True,
                )
                input('Restart only disposable paraworks-e-neo4j, then press Enter: ')
                # Recreate both driver boundary objects: this proves E-2
                # retrieval recovery, not just E-1 projection cursor durability.
                driver.close()
                driver = _driver()
                store = Neo4jGraphStore(driver)
                recovered = None
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    recovered = adapter(
                        db, store, RunnableLambda(lambda _: baseline)
                    ).invoke(case_request)
                    if (
                        tuple(
                            candidate.evidence.serving_document_id
                            for candidate in recovered.visible
                        )
                        == ordered_ids
                    ):
                        break
                    time.sleep(0.25)
                assert recovered is not None
                assert (
                    tuple(
                        candidate.evidence.serving_document_id
                        for candidate in recovered.visible
                    )
                    == ordered_ids
                )
                assert _source_ids(recovered) == _source_ids(result)
            if os.getenv('PARAWORKS_TEST_POSTGRES_RESTART_PAUSE') == '1':
                print(
                    f'E3_POSTGRES_RESTART_READY scope={scope_id} ordered={ordered_ids}',
                    flush=True,
                )
                engine = db.get_bind()
                db.close()
                engine.dispose()
                input(
                    'Restart only disposable paraworks-e-postgres, then press Enter: '
                )
                # This is explicit fresh-engine/session reconstruction after an
                # outage, not a production transparent retry claim.
                engine.dispose()
                db = Session(engine)
                recovered_baseline, _ = _pgvector_seed(
                    db, case_request, case, populate=False
                )
                recovered = adapter(
                    db, store, RunnableLambda(lambda _: recovered_baseline)
                ).invoke(case_request)
                assert (
                    tuple(
                        candidate.evidence.serving_document_id
                        for candidate in recovered_baseline.visible
                    )
                    == CASES[case]['baseline_ids']
                )
                assert (
                    tuple(
                        candidate.evidence.serving_document_id
                        for candidate in recovered.visible
                    )
                    == ordered_ids
                )
        else:
            assert ordered_ids == CASES[case]['baseline_ids']
            assert _source_ids(result) == expected
            assert metrics.precision_at_k == metrics.recall_at_k == 1.0
    finally:
        with driver.session() as session:
            session.run(
                'MATCH (n {scope_id:$scope}) DETACH DELETE n', scope=scope_id
            ).consume()
        driver.close()
        db.close()


@pytest.mark.parametrize('mode', ('unavailable', 'stale'))
def test_e3_rollback_keeps_seed_path_and_embedding_receipt(lexical_pg, mode):
    """Fake failure boundary preserves an existing receipt; no live call is made."""
    from decimal import Decimal

    from langchain_core.runnables import RunnableLambda

    db = lexical_pg
    seed_corpus(db)
    # Rollback contract uses an already-computed seed result; the comparison
    # test above, not this failure-path unit, establishes its PGvector origin.
    from backend.app.rag.neo4j_retriever import _evidence

    evidence = _evidence(db, 'history_event:1', settings=SETTINGS, scope=SCOPE)
    assert evidence is not None
    original = RetrievalResult(
        configured_backend='pgvector',
        effective_backend='pgvector',
        visible=(
            RetrievalCandidate(
                evidence=evidence, relevance_score=1.0, matched_terms=()
            ),
        ),
        hidden_match_count=0,
        hidden_count_capped=False,
        top_candidate_window_hmac='a' * 64,
        query_embedding_receipt=None,
        trace=SanitizedRetrievalTrace(1, 1, 0, 0, 0, None),
    )
    receipt = QueryEmbeddingReceipt(
        True, 10, Decimal('0.000001'), 1, 'component_succeeded', 'a' * 64, 'b' * 64
    )
    original = replace(original, query_embedding_receipt=receipt)
    calls = []

    class Store:
        def traverse(self, **kwargs):
            calls.append(kwargs)
            if mode == 'unavailable':
                from backend.app.rag.graph_store import GraphUnavailable

                raise GraphUnavailable('disposable graph unavailable')
            return ()

    result = adapter(db, Store(), RunnableLambda(lambda value: original)).invoke(
        request()
    )
    assert result.visible == original.visible
    assert result.query_embedding_receipt is receipt
    assert result.top_candidate_window_hmac == original.top_candidate_window_hmac
    assert result.trace.provider_attempt_count == 0
    assert len(calls) == 1


def test_e3_default_off_composition_uses_unwrapped_seed(lexical_pg):
    """Config rollback is exercised through the production composition helper."""
    from langchain_core.runnables import RunnableLambda

    from backend.app.agent_runtime.rag_v2_composition import _graph_enrichment

    seed = RunnableLambda(lambda value: value)
    assert _graph_enrichment(lexical_pg, SETTINGS, seed) is seed
