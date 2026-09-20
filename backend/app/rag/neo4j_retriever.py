"""One-hop enrichment of an existing, paid-or-free seed retrieval.

Graph bytes never supply model text, citations, approval or permission authority.
Only the canonical E-1 PostgreSQL reconstruction can authorize a returned path.
"""

import json
from dataclasses import asdict, replace
from time import perf_counter_ns

from langchain_core.runnables import Runnable
from sqlalchemy import select

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_identity import (
    security_scope_fingerprint,
    verify_serialized_security_scope_fingerprint,
)
from backend.app.models import RagServingCorpusGeneration
from backend.app.rag.graph_projection import (
    GraphPathDependency,
    GraphTraversalPolicy,
    read_projection_page,
)
from backend.app.rag.graph_store import GraphUnavailable
from backend.app.rag.retrieval import (
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
)
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver

GRAPH_POLICY_VERSION = 'rag-graph-enrichment:v1'


def graph_path_payload(paths):
    return [json.dumps(asdict(p), sort_keys=True, separators=(',', ':')) for p in paths]


def validate_graph_paths(db, paths, *, settings, scope):
    """Exact ordered reconstruction, including unselected influencing edges.

    Invalid/stale data returns False; authority/SQL failures propagate closed.
    Call inside the existing provider/finalization corpus transaction fence.
    """
    if type(paths) is not tuple or len(paths) > 50:
        return False
    if not paths:
        return True
    scope_id = security_scope_fingerprint(scope, settings=settings)
    for path in paths:
        if type(path) is not GraphPathDependency:
            return False
        try:
            path.__post_init__()
        except (ValueError, TypeError, AttributeError):
            return False
        if any(n.scope_id != scope_id for n in path.nodes):
            return False
    page = read_projection_page(
        db,
        settings=settings,
        scope=scope,
        document_ids=tuple(sorted({p.nodes[0].document_id for p in paths})),
    )
    nodes = {n.document_id: n for n in page.nodes}
    edges = {e.edge_id: e for e in page.edges}
    return all(
        all(nodes.get(n.document_id) == n for n in p.nodes)
        and all(edges.get(e.edge_id) == e for e in p.edges)
        for p in paths
    )


def _evidence(db, document_id, *, settings, scope):
    kind, identifier = document_id.split(':')
    if kind == 'chunk':
        row = CanonicalSourceObservationResolver(
            db=db, settings=settings
        ).resolve_projection_for_scope_strict(int(identifier), scope=scope)
    else:
        row = TrustedServingEnvelopeResolver(
            db=db, settings=settings
        ).resolve_for_scope_strict(kind, int(identifier), scope=scope)
    return row.evidence if row is not None else None


class Neo4jEvidenceRetriever(Runnable[RetrievalRequest, RetrievalResult]):
    def __init__(self, *, db, settings, graph_store, seed_retriever, policy=None):
        self.db, self.settings = db, settings
        self.graph_store, self.seed_retriever = graph_store, seed_retriever
        self.policy = policy or GraphTraversalPolicy()

    def invoke(self, input, config=None):
        started = perf_counter_ns()
        verify_serialized_security_scope_fingerprint(
            input.security_scope,
            serialized_fingerprint=input.security_scope_fingerprint,
            settings=self.settings,
        )
        seed = self.seed_retriever.invoke(input, config=config)
        # A total candidate budget, not fifty more candidates after seed search.
        remaining = (
            min(self.policy.candidate_limit, input.candidate_scan_limit)
            - seed.trace.candidate_window_count
        )
        visible = list(seed.visible)
        visible_cap = min(input.visible_limit, self.policy.visible_limit)

        def fallback(category):
            return replace(
                seed,
                graph_paths=(),
                graph_policy_version=GRAPH_POLICY_VERSION,
                trace=replace(
                    seed.trace,
                    visible_count=len(seed.visible),
                    fallback_category=category,
                    latency_ms=max(0, (perf_counter_ns() - started) // 1_000_000),
                ),
            )

        # Enrichment must never evict valid seed evidence (ask allows eight).
        # If seeds already fill the graph's five-item cap, retain them exactly.
        if not visible or remaining <= 0 or len(visible) >= visible_cap:
            return fallback('graph_no_benefit')
        generation = self.db.scalar(
            select(RagServingCorpusGeneration.corpus_generation).where(
                RagServingCorpusGeneration.id == 1
            )
        )
        if generation is None:
            raise RuntimeError('canonical corpus generation unavailable')
        seeds = tuple(
            c.evidence.serving_document_id for c in visible[: self.policy.seed_limit]
        )
        try:
            paths = self.graph_store.traverse(
                scope_id=input.security_scope_fingerprint,
                generation=generation,
                seeds=seeds,
                permissions=input.security_scope.allowed_permission_levels,
                policy=replace(self.policy, candidate_limit=remaining),
            )
        except GraphUnavailable:
            return fallback('graph_unavailable')
        if type(paths) is not tuple or len(paths) > remaining:
            return fallback('graph_invalid')
        accepted = []
        ids = {c.evidence.serving_document_id for c in visible}
        for path in paths:
            if not validate_graph_paths(
                self.db, (path,), settings=self.settings, scope=input.security_scope
            ):
                continue
            if not any(n.document_id in seeds for n in path.nodes):
                continue
            added = False
            for node in path.nodes:
                if node.document_id in ids or len(visible) >= visible_cap:
                    continue
                evidence = _evidence(
                    self.db,
                    node.document_id,
                    settings=self.settings,
                    scope=input.security_scope,
                )
                if evidence is None:
                    continue
                visible.append(
                    RetrievalCandidate(
                        evidence=evidence,
                        relevance_score=visible[0].relevance_score,
                        matched_terms=(),
                    )
                )
                ids.add(node.document_id)
                added = True
            if added:
                accepted.append(path)
        if not accepted:
            return fallback('graph_no_benefit')
        secret, _ = fingerprint_secret_bytes(self.settings)
        window_hmac = keyed_fingerprint(
            {
                'seed': seed.top_candidate_window_hmac,
                'paths': graph_path_payload(accepted),
                'graph_policy': GRAPH_POLICY_VERSION,
            },
            secret=secret,
            schema_version='rag-graph-candidate-window:v1',
            policy_version=GRAPH_POLICY_VERSION,
        )
        return replace(
            seed,
            effective_backend='neo4j',
            visible=tuple(visible),
            graph_paths=tuple(accepted),
            graph_policy_version=GRAPH_POLICY_VERSION,
            top_candidate_window_hmac=window_hmac,
            trace=replace(
                seed.trace,
                visible_count=len(visible),
                candidate_window_count=seed.trace.candidate_window_count + len(paths),
                latency_ms=max(0, (perf_counter_ns() - started) // 1_000_000),
            ),
        )
