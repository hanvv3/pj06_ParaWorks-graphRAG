"""Bounded canonical projection. Neo4j is never serving/permission authority.

Only explicit approved SUPPORTED_BY links are projected. A caller must commit
the PostgreSQL transaction after each step; its shared corpus row lock keeps
normal canonical writers from changing a page while Neo4j acknowledges it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from time import monotonic

from sqlalchemy import select, text

from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.ingestion.source_authority import resolve_exact_source_authority
from backend.app.models import (
    Document,
    DocumentChunk,
    RagLexicalServingProjection,
    RagServingCorpusGeneration,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)
from backend.app.rag.serving_contracts import (
    ExplicitApprovalProvenance,
    strictest_permission,
)
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver


@dataclass(frozen=True, slots=True)
class GraphTraversalPolicy:
    """E-2 must reuse seeds/embedding and retain the D candidate/evidence caps."""

    seed_limit: int = 2
    max_hops: int = 1
    candidate_limit: int = 50
    visible_limit: int = 5
    timeout_seconds: float = 1.0

    def __post_init__(self):
        if (
            any(
                type(value) is not int or not 1 <= value <= maximum
                for value, maximum in (
                    (self.seed_limit, 2),
                    (self.max_hops, 1),
                    (self.candidate_limit, 50),
                    (self.visible_limit, 5),
                )
            )
            or not 0 < self.timeout_seconds <= 1.0
        ):
            raise ValueError('graph traversal policy exceeds initial bounds')


@dataclass(frozen=True, slots=True)
class GraphNodeDependency:
    scope_id: str
    document_id: str
    version: str
    permission: str
    source_id: str
    canonical_ref: str


@dataclass(frozen=True, slots=True)
class GraphEdgeDependency:
    scope_id: str
    edge_id: str
    relation: str
    from_id: str
    to_id: str
    version: str
    permission: str
    approval_id: int
    evidence_link_id: int
    review_item_id: int
    canonical_ref: str


@dataclass(frozen=True, slots=True)
class GraphPathDependency:
    nodes: tuple[GraphNodeDependency, ...]
    edges: tuple[GraphEdgeDependency, ...]
    contract_version: str = 'rag-graph-path:v1'

    def __post_init__(self):
        if (
            self.contract_version != 'rag-graph-path:v1'
            or len(self.edges) != 1
            or len(self.nodes) != 2
        ):
            raise ValueError('unsupported graph path')
        left, right = self.nodes
        edge = self.edges[0]
        if (
            left.scope_id != right.scope_id
            or edge.scope_id != left.scope_id
            or edge.relation != 'SUPPORTED_BY'
            or edge.from_id != left.document_id
            or edge.to_id != right.document_id
        ):
            raise ValueError('disconnected or cross-scope graph path')


@dataclass(frozen=True, slots=True)
class GraphProjectionPage:
    scope_id: str
    generation: int
    start_cursor: int
    cursor: int
    complete: bool
    nodes: tuple[GraphNodeDependency, ...]
    edges: tuple[GraphEdgeDependency, ...]
    scanned_count: int
    source_updated_at: str


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _node(evidence, scope_id):
    return GraphNodeDependency(
        scope_id,
        evidence.serving_document_id,
        evidence.serving_version_fingerprint,
        evidence.effective_permission,
        evidence.public_source_id,
        _json(asdict(evidence.version_envelope)),
    )


def _source_within_bounds(db, source_id):
    return (
        len(
            tuple(
                db.scalars(
                    select(DocumentChunk.id)
                    .join(
                        Document,
                        Document.current_document_version_id
                        == DocumentChunk.version_id,
                    )
                    .where(Document.source_id == source_id)
                    .limit(101)
                )
            )
        )
        <= 100
    )


def _trusted_within_bounds(db, kind, identifier):
    aliases = ('decision', 'decision_record') if kind == 'decision_record' else (kind,)
    links = tuple(
        db.scalars(
            select(TrustedKnowledgeApprovalLink.id)
            .where(
                TrustedKnowledgeApprovalLink.knowledge_type.in_(aliases),
                TrustedKnowledgeApprovalLink.knowledge_id == identifier,
                TrustedKnowledgeApprovalLink.active.is_(True),
            )
            .limit(51)
        )
    )
    if len(links) > 50:
        return False
    children = tuple(
        db.scalars(
            select(TrustedKnowledgeEvidenceLink.canonical_source_id)
            .where(TrustedKnowledgeEvidenceLink.approval_link_id.in_(links))
            .limit(51)
        )
    )
    if len(children) > 50:
        return False
    for source in children:
        if (
            not source.isascii()
            or not source.isdecimal()
            or not _source_within_bounds(db, int(source))
        ):
            return False
    return True


def _set_projection_transaction_limits(db):
    """Bound every PG entry path without taking ownership of its transaction."""
    if db.get_bind().dialect.name == 'postgresql':
        db.execute(text("SET LOCAL statement_timeout = '5s'"))
        db.execute(text("SET LOCAL lock_timeout = '2s'"))


def read_projection_page(
    db, *, settings, scope, cursor=0, batch_size=50, document_ids=None
):
    if (
        type(batch_size) is not int
        or not 1 <= batch_size <= 100
        or type(cursor) is not int
        or cursor < 0
    ):
        raise ValueError('projection batch/cursor outside bounds')
    _set_projection_transaction_limits(db)
    generation = db.scalar(
        select(RagServingCorpusGeneration)
        .where(RagServingCorpusGeneration.id == 1)
        .with_for_update(read=True)
        .execution_options(populate_existing=True)
    )
    if generation is None:
        raise RuntimeError('canonical corpus generation unavailable')
    scope_id = security_scope_fingerprint(scope, settings=settings)
    if document_ids is not None and (
        type(document_ids) is not tuple or not 1 <= len(document_ids) <= 50
    ):
        raise ValueError('canonical graph identity window outside bounds')
    rows = tuple(
        db.scalars(
            select(RagLexicalServingProjection)
            .where(
                RagLexicalServingProjection.corpus_generation
                == generation.corpus_generation,
                RagLexicalServingProjection.id > cursor,
                (
                    RagLexicalServingProjection.serving_document_id.in_(document_ids)
                    if document_ids is not None
                    else True
                ),
            )
            .order_by(RagLexicalServingProjection.id)
            .limit(batch_size + 1)
        )
    )
    complete = len(rows) <= batch_size
    rows = rows[:batch_size]
    nodes, edges = {}, []
    raw = CanonicalSourceObservationResolver(db=db, settings=settings)
    trusted = TrustedServingEnvelopeResolver(db=db, settings=settings)
    started = monotonic()
    for row in rows:
        if monotonic() - started > 5:
            raise RuntimeError('canonical projection page timeout')
        kind, identifier = row.serving_document_id.split(':')
        if kind == 'chunk':
            chunk_row = db.get(DocumentChunk, int(identifier))
            if chunk_row is None or not _source_within_bounds(db, chunk_row.source_id):
                continue
            projection = raw.resolve_projection_for_scope_strict(
                int(identifier), scope=scope
            )
            if projection is not None:
                nodes[row.serving_document_id] = _node(projection.evidence, scope_id)
            continue
        if not _trusted_within_bounds(db, kind, int(identifier)):
            continue
        envelope = trusted.resolve_for_scope_strict(kind, int(identifier), scope=scope)
        if envelope is None:
            continue
        evidence = envelope.evidence
        nodes[row.serving_document_id] = _node(evidence, scope_id)
        provenance = evidence.provenance
        if not isinstance(provenance, ExplicitApprovalProvenance):
            continue
        item = db.get(ReviewItem, provenance.review_item_id)
        pairs = set(zip(item.source_links, item.source_snippets, strict=True))
        # A bounded page must not fan out without limit. Oversized approvals are
        # omitted entirely from relation projection; no partial trusted path.
        if len(provenance.evidence_links) > 50:
            continue
        local_nodes, local_edges = {}, []
        for child in provenance.evidence_links:
            source = db.get(Source, int(child.canonical_source_id))
            authority = resolve_exact_source_authority(db, source=source)
            if authority is None or len(authority.chunks) > 100:
                local_edges = []
                break
            matches = [
                chunk
                for chunk in authority.chunks
                if (source.source_url, chunk.source_snippet) in pairs
            ]
            if not matches:
                local_edges = []
                break
            # Same source/snippet duplicated in a parse is resolved stably.
            chunk = min(matches, key=lambda value: value.id)
            projection = raw.resolve_projection_for_scope_strict(chunk.id, scope=scope)
            if projection is None and source.source_type == 'slack':
                projection = raw.resolve_approved_slack_child_for_scope_strict(
                    chunk.id, scope=scope
                )
            if projection is None:
                local_edges = []
                break
            node = _node(projection.evidence, scope_id)
            local_nodes[node.document_id] = node
            ref = _json(
                {
                    'approval': asdict(provenance),
                    'child': asdict(child),
                    'from_version': evidence.serving_version_fingerprint,
                    'to_version': node.version,
                }
            )
            edge_id = f'supported_by:{provenance.approval_link_id}:{child.trusted_knowledge_evidence_link_id}:{chunk.id}'
            local_edges.append(
                GraphEdgeDependency(
                    scope_id,
                    edge_id,
                    'SUPPORTED_BY',
                    evidence.serving_document_id,
                    node.document_id,
                    sha256(ref.encode()).hexdigest(),
                    strictest_permission(
                        (evidence.effective_permission, node.permission)
                    ),
                    provenance.approval_link_id,
                    child.trusted_knowledge_evidence_link_id,
                    provenance.review_item_id,
                    ref,
                )
            )
        if local_edges:
            nodes.update(local_nodes)
            edges.extend(local_edges)
    updated = generation.updated_at
    return GraphProjectionPage(
        scope_id,
        generation.corpus_generation,
        cursor,
        rows[-1].id if rows else cursor,
        complete,
        tuple(nodes[key] for key in sorted(nodes)),
        tuple(sorted(edges, key=lambda edge: edge.edge_id)),
        len(rows),
        updated.isoformat(),
    )


def reconcile_graph_step(db, *, settings, scope, store, batch_size=50):
    """One bounded, restartable admin-job step; no runtime routing side effect."""
    scope_id = security_scope_fingerprint(scope, settings=settings)
    status = store.status(scope_id)
    _set_projection_transaction_limits(db)
    generation = db.scalar(
        select(RagServingCorpusGeneration)
        .where(RagServingCorpusGeneration.id == 1)
        .with_for_update(read=True)
        .execution_options(populate_existing=True)
    )
    if generation is None:
        raise RuntimeError('canonical corpus generation unavailable')
    if status.generation > generation.corpus_generation:
        from backend.app.rag.graph_store import StaleGraphGeneration

        raise StaleGraphGeneration('canonical generation is older than projection')
    if status.generation != generation.corpus_generation or not status.scanned:
        cursor = (
            status.cursor if status.generation == generation.corpus_generation else 0
        )
        page = read_projection_page(
            db, settings=settings, scope=scope, cursor=cursor, batch_size=batch_size
        )
        store.apply_page(page)
    else:
        store.sweep(scope_id, generation.corpus_generation, batch_size=batch_size)
    return store.status(scope_id, canonical_generation=generation.corpus_generation)
