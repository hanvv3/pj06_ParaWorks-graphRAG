import importlib
from dataclasses import asdict, replace

import pytest

from backend.tests.graph_projection_fixtures import SCOPE, SETTINGS, seed_corpus


def projection_module():
    assert importlib.util.find_spec('backend.app.rag.graph_projection'), (
        'canonical graph projection is missing'
    )
    return importlib.import_module('backend.app.rag.graph_projection')


def test_canonical_page_preserves_exact_approval_dependencies_without_text(db_session):
    m = projection_module()
    seed_corpus(db_session)
    page = m.read_projection_page(
        db_session, settings=SETTINGS, scope=SCOPE, cursor=0, batch_size=10
    )
    assert {n.document_id for n in page.nodes} == {
        'history_event:1',
        'chunk:1',
        'chunk:2',
        'chunk:3',
    }
    assert [
        (e.from_id, e.to_id, e.approval_id, e.evidence_link_id) for e in page.edges
    ] == [
        ('history_event:1', 'chunk:1', 1, 1),
        ('history_event:1', 'chunk:2', 1, 2),
    ]
    assert all(
        e.relation == 'SUPPORTED_BY' and e.permission == 'internal' for e in page.edges
    )
    assert all(e.scope_id == page.scope_id and len(e.version) == 64 for e in page.edges)
    payload = str(asdict(page))
    assert 'Exact evidence number' not in payload
    assert 'Canonical approved knowledge' not in payload
    assert 'embedding' not in payload
    assert page.complete and page.generation == 1


@pytest.mark.parametrize('case', ['restricted', 'revoke'])
def test_restricted_or_revoked_approval_never_projects_a_relation(db_session, case):
    m = projection_module()
    seed_corpus(db_session, case)
    page = m.read_projection_page(
        db_session, settings=SETTINGS, scope=SCOPE, cursor=0, batch_size=10
    )
    assert not page.edges
    assert 'history_event:1' not in {n.document_id for n in page.nodes}


def test_scope_and_page_bounds(db_session):
    m = projection_module()
    seed_corpus(db_session)
    page = m.read_projection_page(
        db_session, settings=SETTINGS, scope=SCOPE, cursor=0, batch_size=1
    )
    assert page.scanned_count == 1 and not page.complete
    other = m.read_projection_page(
        db_session,
        settings=SETTINGS,
        scope=replace(SCOPE, workspace_scope_id='other'),
        cursor=0,
        batch_size=10,
    )
    assert not other.edges and other.scope_id != page.scope_id
    with pytest.raises(ValueError):
        m.read_projection_page(
            db_session, settings=SETTINGS, scope=SCOPE, cursor=0, batch_size=101
        )


def test_path_dependency_rejects_disconnected_and_cross_scope_edges(db_session):
    m = projection_module()
    seed_corpus(db_session)
    page = m.read_projection_page(
        db_session, settings=SETTINGS, scope=SCOPE, cursor=0, batch_size=10
    )
    nodes = {n.document_id: n for n in page.nodes}
    edge = page.edges[0]
    path = m.GraphPathDependency(
        nodes=(nodes['history_event:1'], nodes['chunk:1']), edges=(edge,)
    )
    assert path.contract_version == 'rag-graph-path:v1'
    with pytest.raises(ValueError):
        m.GraphPathDependency(
            nodes=(nodes['history_event:1'], nodes['chunk:2']), edges=(edge,)
        )
    with pytest.raises(ValueError):
        m.GraphPathDependency(
            nodes=path.nodes, edges=(replace(edge, scope_id='other'),)
        )


@pytest.mark.parametrize('failure', [RuntimeError, ValueError])
def test_official_driver_failure_is_sanitized(failure):
    projection_module()
    from backend.app.rag.graph_store import GraphUnavailable, Neo4jGraphStore

    class BrokenDriver:
        def session(self, **kwargs):
            raise failure('password=private raw source content')

    with pytest.raises(GraphUnavailable, match='graph unavailable') as error:
        Neo4jGraphStore(BrokenDriver()).status('scope')
    assert 'private' not in str(error.value)


def test_driver_write_failure_does_not_acknowledge_cursor(db_session):
    m = projection_module()
    from backend.app.rag.graph_store import GraphUnavailable, Neo4jGraphStore

    seed_corpus(db_session)
    page = m.read_projection_page(
        db_session, settings=SETTINGS, scope=SCOPE, batch_size=2
    )
    calls = []

    class Result:
        def single(self):
            return {'state': {'generation': -1, 'cursor': 0, 'scanned': False}}

        def consume(self):
            pass

    class Transaction:
        def run(self, query, **parameters):
            calls.append((query, parameters))
            if 'UNWIND $nodes' in query:
                raise OSError('simulated connection lost during write')
            return Result()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute_write(self, callback):
            return callback(Transaction())

    class Driver:
        def session(self, **kwargs):
            return Session()

    with pytest.raises(GraphUnavailable):
        Neo4jGraphStore(Driver()).apply_page(page)
    # No acknowledgement query ran after the failed data write. Actual driver
    # transaction atomicity/replay is separately checked against Neo4j.
    assert not any('SET s.cursor=$cursor' in query for query, _ in calls)
    node_parameters = calls[-1][1]
    assert node_parameters['scope'] == page.scope_id
    assert node_parameters['generation'] == 1
    assert len(node_parameters['nodes']) == 2


def test_traversal_policy_cannot_expand_initial_search_budget():
    m = projection_module()
    for kwargs in (
        {'seed_limit': 3},
        {'max_hops': 2},
        {'candidate_limit': 51},
        {'timeout_seconds': 2},
        {'visible_limit': 8},
    ):
        with pytest.raises(ValueError):
            m.GraphTraversalPolicy(**kwargs)


def test_oversized_source_bundle_is_omitted_before_canonical_resolver(
    db_session, monkeypatch
):
    m = projection_module()
    from backend.app.models import DocumentChunk

    sources, chunks, target, approval = seed_corpus(db_session)
    for i in range(1, 101):
        db_session.add(
            DocumentChunk(
                version_id=chunks[0].version_id,
                source_id=sources[0].id,
                parser_run_id=chunks[0].parser_run_id,
                chunk_index=i,
                text='oversized',
                source_snippet='oversized',
                permission_level='internal',
                metadata_={},
            )
        )
    db_session.flush()
    original = m.CanonicalSourceObservationResolver.resolve_projection_for_scope_strict

    def guarded(self, chunk_id, **kwargs):
        assert chunk_id != 1, 'oversized source reached unbounded resolver'
        return original(self, chunk_id, **kwargs)

    monkeypatch.setattr(
        m.CanonicalSourceObservationResolver,
        'resolve_projection_for_scope_strict',
        guarded,
    )
    page = m.read_projection_page(db_session, settings=SETTINGS, scope=SCOPE)
    assert 'chunk:1' not in {node.document_id for node in page.nodes}
