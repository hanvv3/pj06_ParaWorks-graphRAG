from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import RagLexicalServingProjection, RagServingCorpusGeneration
from backend.app.rag.embeddings import OpenAIEmbeddingConfig, OpenAIEmbeddingModel
from backend.app.rag.lexical_projection import (
    score_rag_lexical_candidate,
    tokenize_rag_lexical_query,
)
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.rag.retrieval import ClassifiedRetrievalCandidate, RetrievalRequest
from backend.app.rag.serving_contracts import (
    EvidenceAccessClassification,
    TrustedServingEnvelope,
    known_permission,
)
from backend.app.rag.serving_generation import RAG_LEXICAL_COMPAT_VERSION
from backend.app.rag.source_observations import (
    CanonicalSourceObservationEligibilityService,
    CanonicalSourceObservationResolver,
)
from backend.app.rag.trusted_evidence import (
    ServingEvidenceResolver,
    TrustedEvidenceAuthorizer,
    TrustedServingEnvelopeResolver,
)


def build_pgvector_search_store(*, db: Session, settings: Settings):
    if not settings.rag_use_pgvector_search:
        return None
    if db.bind is None or db.bind.dialect.name != 'postgresql':
        return None
    if not settings.openai_api_key:
        return None
    store = PgVectorStore(
        session=db,
        config=PgVectorConfig(embedding_dimensions=settings.openai_embedding_dimensions),
    )
    embedding_model = OpenAIEmbeddingModel(
        config=OpenAIEmbeddingConfig(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            timeout_seconds=settings.openai_embedding_timeout_seconds,
        )
    )
    return PgVectorSearchAdapter(store=store, embedding_model=embedding_model)


class PgVectorSearchAdapter:
    def __init__(self, *, store: PgVectorStore, embedding_model: OpenAIEmbeddingModel) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self._query_embeddings: dict[str, list[float]] = {}

    def search(self, *, query: str, user: DemoUser, limit: int = 5):
        query_embedding = self._query_embeddings.get(query)
        if query_embedding is None:
            query_embedding = self.embedding_model.embed(query)
            self._query_embeddings[query] = query_embedding
        return self.store.search_with_embedding(
            query_embedding=query_embedding,
            user=user,
            limit=limit,
        )


@dataclass(frozen=True, slots=True)
class PostgresKeywordApprovalLink:
    workspace_scope_id: str
    active: bool
    resolution_source: str
    child_source_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PostgresKeywordResourceCandidate:
    serving_document_id: str
    serving_kind: str
    project_key: str | None
    raw_source_id: int | None
    score: float
    links: tuple[PostgresKeywordApprovalLink, ...]


@dataclass(frozen=True, slots=True)
class PostgresKeywordResourceParameters:
    workspace_scope_id: str
    project_keys: tuple[str, ...] | None
    source_ids: tuple[int, ...] | None

    @classmethod
    def from_security_scope(
        cls,
        scope: SecurityScope,
    ) -> PostgresKeywordResourceParameters:
        return cls(
            workspace_scope_id=scope.workspace_scope_id,
            project_keys=tuple(
                value.removeprefix('project_key:')
                for value in scope.project_constraints
            ),
            source_ids=tuple(
                int(value.removeprefix('source_pk:'))
                for value in scope.source_constraints
            ),
        )


class _KeywordResourcePredicateNode(Protocol):
    def render(self, *, alias: str) -> str: ...

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class _AllResourcePredicates:
    children: tuple[_KeywordResourcePredicateNode, ...]

    def render(self, *, alias: str) -> str:
        return '(' + ' AND '.join(
            child.render(alias=alias) for child in self.children
        ) + ')'

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool:
        return all(child.evaluate(candidate, parameters) for child in self.children)


@dataclass(frozen=True, slots=True)
class _AnyResourcePredicate:
    children: tuple[_KeywordResourcePredicateNode, ...]

    def render(self, *, alias: str) -> str:
        return '(' + ' OR '.join(
            child.render(alias=alias) for child in self.children
        ) + ')'

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool:
        return any(child.evaluate(candidate, parameters) for child in self.children)


@dataclass(frozen=True, slots=True)
class _ServingKindPredicate:
    serving_kind: str

    def render(self, *, alias: str) -> str:
        return f"{alias}.serving_kind = '{self.serving_kind}'"

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool:
        del parameters
        return candidate.serving_kind == self.serving_kind


@dataclass(frozen=True, slots=True)
class _RawScopePredicate:
    def render(self, *, alias: str) -> str:
        return f"""
CAST(:project_keys AS text[]) IS NOT NULL
AND CAST(:source_ids AS bigint[]) IS NOT NULL
AND cardinality(CAST(:project_keys AS text[])) = 0
AND (
  cardinality(CAST(:source_ids AS bigint[])) = 0
  OR EXISTS (
    SELECT 1 FROM document_chunks AS chunk
    WHERE chunk.id = split_part({alias}.serving_document_id, ':', 2)::bigint
      AND chunk.source_id = ANY(CAST(:source_ids AS bigint[]))
  )
)
""".strip()

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool:
        if parameters.project_keys is None or parameters.source_ids is None:
            return False
        return bool(
            not parameters.project_keys
            and (
                not parameters.source_ids
                or candidate.raw_source_id in parameters.source_ids
            )
        )


@dataclass(frozen=True, slots=True)
class _TrustedProjectPredicate:
    def render(self, *, alias: str) -> str:
        return f"""
CAST(:project_keys AS text[]) IS NOT NULL
AND (
  cardinality(CAST(:project_keys AS text[])) = 0
  OR (
    (split_part({alias}.serving_document_id, ':', 1) = 'decision_record'
     AND EXISTS (
       SELECT 1 FROM decision_records AS knowledge
       WHERE knowledge.id = split_part({alias}.serving_document_id, ':', 2)::bigint
         AND knowledge.project_key = ANY(CAST(:project_keys AS text[]))
     ))
    OR (split_part({alias}.serving_document_id, ':', 1) = 'history_event'
     AND EXISTS (
       SELECT 1 FROM history_events AS knowledge
       WHERE knowledge.id = split_part({alias}.serving_document_id, ':', 2)::bigint
         AND knowledge.project_key = ANY(CAST(:project_keys AS text[]))
     ))
    OR (split_part({alias}.serving_document_id, ':', 1) = 'timeline_event'
     AND EXISTS (
       SELECT 1 FROM timeline_events AS knowledge
       WHERE knowledge.id = split_part({alias}.serving_document_id, ':', 2)::bigint
         AND knowledge.project_key = ANY(CAST(:project_keys AS text[]))
     ))
    OR (split_part({alias}.serving_document_id, ':', 1) = 'todo'
     AND EXISTS (
       SELECT 1 FROM todos AS knowledge
       WHERE knowledge.id = split_part({alias}.serving_document_id, ':', 2)::bigint
         AND knowledge.project_key = ANY(CAST(:project_keys AS text[]))
     ))
  )
)
""".strip()

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool:
        if parameters.project_keys is None:
            return False
        return bool(
            not parameters.project_keys
            or candidate.project_key in parameters.project_keys
        )


def _approval_target_sql(*, alias: str) -> str:
    return f"""
(
  (split_part({alias}.serving_document_id, ':', 1) = 'decision_record'
   AND approval.knowledge_type IN ('decision', 'decision_record'))
  OR (
    split_part({alias}.serving_document_id, ':', 1) <> 'decision_record'
    AND approval.knowledge_type = split_part({alias}.serving_document_id, ':', 1)
  )
)
AND approval.knowledge_id = split_part({alias}.serving_document_id, ':', 2)::bigint
AND approval.active IS TRUE
""".strip()


@dataclass(frozen=True, slots=True)
class _TrustedExplicitPredicate:
    def render(self, *, alias: str) -> str:
        target = _approval_target_sql(alias=alias)
        return f"""
CAST(:source_id_texts AS text[]) IS NOT NULL
AND EXISTS (
  SELECT 1
  FROM trusted_knowledge_approval_links AS approval
  WHERE {target}
    AND approval.security_scope_id = :workspace_scope_id
    AND approval.resolution_source IN ('human', 'auto_policy')
    AND EXISTS (
      SELECT 1 FROM trusted_knowledge_evidence_links AS child
      WHERE child.approval_link_id = approval.id
    )
    AND (
      cardinality(CAST(:source_id_texts AS text[])) = 0
      OR NOT EXISTS (
        SELECT 1 FROM trusted_knowledge_evidence_links AS child
        WHERE child.approval_link_id = approval.id
          AND child.canonical_source_id
              <> ALL(CAST(:source_id_texts AS text[]))
      )
    )
)
""".strip()

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool:
        if parameters.source_ids is None:
            return False
        return any(
            link.active
            and link.workspace_scope_id == parameters.workspace_scope_id
            and link.resolution_source in {'human', 'auto_policy'}
            and bool(link.child_source_ids)
            and (
                not parameters.source_ids
                or all(
                    source_id in parameters.source_ids
                    for source_id in link.child_source_ids
                )
            )
            for link in candidate.links
        )


@dataclass(frozen=True, slots=True)
class _TrustedLegacyPredicate:
    def render(self, *, alias: str) -> str:
        target = _approval_target_sql(alias=alias)
        return f"""
CAST(:source_id_texts AS text[]) IS NOT NULL
AND cardinality(CAST(:source_id_texts AS text[])) = 0
AND NOT EXISTS (
  SELECT 1
  FROM trusted_knowledge_approval_links AS approval
  WHERE {target}
)
""".strip()

    def evaluate(
        self,
        candidate: PostgresKeywordResourceCandidate,
        parameters: PostgresKeywordResourceParameters,
    ) -> bool:
        if parameters.source_ids is None:
            return False
        return not parameters.source_ids and not any(
            link.active for link in candidate.links
        )


POSTGRES_KEYWORD_RESOURCE_PREDICATE = _AnyResourcePredicate(
    children=(
        _AllResourcePredicates(
            children=(
                _ServingKindPredicate('raw_chunk'),
                _RawScopePredicate(),
            )
        ),
        _AllResourcePredicates(
            children=(
                _ServingKindPredicate('trusted_knowledge'),
                _TrustedProjectPredicate(),
                _AnyResourcePredicate(
                    children=(
                        _TrustedExplicitPredicate(),
                        _TrustedLegacyPredicate(),
                    )
                ),
            )
        ),
    )
)


_POSTGRES_KEYWORD_SEARCH_SQL_TEMPLATE = """
WITH coarse_candidates AS (
  SELECT
    projection.serving_document_id,
    projection.serving_kind,
    projection.support_mode,
    projection.effective_permission,
    projection.serving_identity_hmac,
    projection.serving_version_fingerprint,
    projection.model_content_hmac,
    projection.canonical_citation_projection_hmac,
    projection.title_lower,
    projection.searchable_lower
  FROM rag_lexical_serving_projections AS projection
  JOIN rag_serving_corpus_generations AS generation
    ON generation.id = projection.corpus_generation_id
   AND generation.corpus_generation = projection.corpus_generation
  WHERE projection.corpus_generation_id = 1
    AND projection.lexical_contract_version = :lexical_contract_version
    AND projection.fingerprint_key_version = :fingerprint_key_version
    AND projection.fingerprint_key_material_verifier = :key_material_verifier
    AND projection.effective_permission IN ('public', 'internal', 'restricted')
    AND EXISTS (
      SELECT 1
      FROM unnest(CAST(:query_terms AS text[])) AS coarse_term(term)
      WHERE strpos(projection.searchable_lower, coarse_term.term) > 0
    )
    AND (__RESOURCE_PREDICATE__)
), scored_candidates AS (
  SELECT
    projection.*,
    scored.score,
    scored.matched_terms
  FROM coarse_candidates AS projection
  CROSS JOIN LATERAL rag_python_lexical_score_v1(
    projection.title_lower,
    projection.searchable_lower,
    CAST(:query_terms AS text[]),
    CAST(:phrase_lower AS text)
  ) AS scored
), relevant_candidates AS (
  SELECT *
  FROM scored_candidates AS scored
  WHERE scored.score > 0
  ORDER BY
    CASE WHEN scored.serving_kind = 'trusted_knowledge' THEN 0 ELSE 1 END,
    scored.score DESC,
    scored.serving_document_id
  LIMIT 50
)
SELECT * FROM relevant_candidates
ORDER BY
  CASE WHEN serving_kind = 'trusted_knowledge' THEN 0 ELSE 1 END,
  score DESC,
  serving_document_id
"""


def build_postgres_keyword_search_sql(
    predicate: _KeywordResourcePredicateNode,
) -> str:
    marker = '__RESOURCE_PREDICATE__'
    if _POSTGRES_KEYWORD_SEARCH_SQL_TEMPLATE.count(marker) != 1:
        raise RuntimeError('PostgreSQL keyword resource predicate marker is invalid')
    return _POSTGRES_KEYWORD_SEARCH_SQL_TEMPLATE.replace(
        marker,
        predicate.render(alias='projection'),
    )


POSTGRES_KEYWORD_SEARCH_SQL = build_postgres_keyword_search_sql(
    POSTGRES_KEYWORD_RESOURCE_PREDICATE
)

_SQLITE_ORACLE_MAX_PROJECTIONS = 10_000
_POSTGRES_STATEMENT_TIMEOUT_MS = 5_000


class KeywordSearchUnavailableError(RuntimeError):
    pass


class KeywordSearchTimeoutError(KeywordSearchUnavailableError):
    pass


class SqlAlchemyKeywordSearchStore:
    """Canonical keyword candidate store for PostgreSQL and bounded SQLite smoke."""

    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def search(
        self,
        request: RetrievalRequest,
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        terms = tokenize_rag_lexical_query(request.retrieval_query_text)
        if not terms:
            return ()
        try:
            if self._db.get_bind().dialect.name == 'postgresql':
                return self._postgresql_search(request, terms=terms)
            return self._sqlite_search(request, terms=terms)
        except DBAPIError as exc:
            if _is_statement_timeout(exc):
                raise KeywordSearchTimeoutError('keyword retrieval timed out') from None
            raise KeywordSearchUnavailableError('keyword retrieval is unavailable') from None
        except SQLAlchemyError:
            raise KeywordSearchUnavailableError('keyword retrieval is unavailable') from None

    def _postgresql_search(
        self,
        request: RetrievalRequest,
        *,
        terms: tuple[str, ...],
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        self._db.execute(
            text("SELECT set_config('statement_timeout', :timeout_value, true)"),
            {'timeout_value': f'{_POSTGRES_STATEMENT_TIMEOUT_MS}ms'},
        )
        scope = request.security_scope
        rows = tuple(
            self._db.execute(
                text(POSTGRES_KEYWORD_SEARCH_SQL),
                {
                    'query_terms': list(terms),
                    'phrase_lower': request.retrieval_query_text.strip().lower(),
                    'lexical_contract_version': RAG_LEXICAL_COMPAT_VERSION,
                    'fingerprint_key_version': (
                        self._settings.agent_runtime_fingerprint_key_version
                    ),
                    'key_material_verifier': fingerprint_key_material_verifier(
                        self._settings.agent_runtime_fingerprint_secret
                    ),
                    'project_keys': [
                        value.removeprefix('project_key:')
                        for value in scope.project_constraints
                    ],
                    'source_ids': [
                        int(value.removeprefix('source_pk:'))
                        for value in scope.source_constraints
                    ],
                    'source_id_texts': [
                        value.removeprefix('source_pk:')
                        for value in scope.source_constraints
                    ],
                    'workspace_scope_id': scope.workspace_scope_id,
                },
            )
            .mappings()
            .all()
        )
        candidates: list[ClassifiedRetrievalCandidate] = []
        for row in rows:
            candidate = self._canonical_candidate(
                row,
                request=request,
                supplied_score=float(row['score']),
                supplied_terms=tuple(row['matched_terms']),
            )
            if candidate is None:
                raise KeywordSearchUnavailableError('keyword retrieval is unavailable')
            if not _is_stage_one_candidate(candidate.access):
                raise KeywordSearchUnavailableError('keyword retrieval is unavailable')
            candidates.append(candidate)
        return tuple(candidates)

    def _sqlite_search(
        self,
        request: RetrievalRequest,
        *,
        terms: tuple[str, ...],
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        generation = self._db.get(RagServingCorpusGeneration, 1)
        if generation is None:
            return ()
        rows = tuple(
            self._db.scalars(
                select(RagLexicalServingProjection)
                .where(
                    RagLexicalServingProjection.corpus_generation
                    == generation.corpus_generation,
                    RagLexicalServingProjection.lexical_contract_version
                    == RAG_LEXICAL_COMPAT_VERSION,
                    RagLexicalServingProjection.fingerprint_key_version
                    == self._settings.agent_runtime_fingerprint_key_version,
                    RagLexicalServingProjection.fingerprint_key_material_verifier
                    == fingerprint_key_material_verifier(
                        self._settings.agent_runtime_fingerprint_secret
                    ),
                    RagLexicalServingProjection.effective_permission.in_(
                        ('public', 'internal', 'restricted')
                    ),
                )
                .order_by(RagLexicalServingProjection.serving_document_id)
                .limit(_SQLITE_ORACLE_MAX_PROJECTIONS + 1)
            ).all()
        )
        if len(rows) > _SQLITE_ORACLE_MAX_PROJECTIONS:
            raise KeywordSearchUnavailableError('keyword retrieval is unavailable')
        scored: list[ClassifiedRetrievalCandidate] = []
        for row in rows:
            score, matched_terms = score_rag_lexical_candidate(
                question=request.retrieval_query_text,
                title=row.title_lower,
                text=_projection_model_text(row),
            )
            if score <= 0:
                continue
            candidate = self._canonical_candidate(
                row,
                request=request,
                supplied_score=score,
                supplied_terms=matched_terms,
            )
            if candidate is not None and _is_stage_one_candidate(candidate.access):
                scored.append(candidate)
        return tuple(sorted(scored, key=_candidate_order)[:50])

    def _canonical_candidate(
        self,
        row: Mapping[str, object] | RagLexicalServingProjection,
        *,
        request: RetrievalRequest,
        supplied_score: float,
        supplied_terms: tuple[str, ...],
    ) -> ClassifiedRetrievalCandidate | None:
        document_id = str(_row_value(row, 'serving_document_id'))
        prefix, separator, raw_id = document_id.partition(':')
        if separator != ':' or not raw_id.isascii() or not raw_id.isdecimal():
            return None
        identifier = int(raw_id)
        if identifier <= 0 or str(identifier) != raw_id:
            return None
        if prefix == 'chunk':
            observation = CanonicalSourceObservationResolver(
                db=self._db,
                settings=self._settings,
            ).resolve_for_index(identifier)
            if observation is None:
                return None
            identity = observation.identity
            evidence = observation.evidence
            access = CanonicalSourceObservationEligibilityService().classify_access(
                request.security_scope,
                observation,
            )
        elif prefix in {'decision_record', 'history_event', 'timeline_event', 'todo'}:
            envelope = TrustedServingEnvelopeResolver(
                db=self._db,
                settings=self._settings,
            ).resolve_for_index(prefix, identifier)
            if envelope is None:
                return None
            identity = envelope.identity
            evidence = envelope.evidence
            access = _classify_trusted_stage_one_access(
                TrustedEvidenceAuthorizer(
                    db=self._db,
                    settings=self._settings,
                ),
                scope=request.security_scope,
                envelope=envelope,
            )
        else:
            return None
        if not _projection_matches(row, evidence=evidence):
            return None
        oracle_score, oracle_terms = score_rag_lexical_candidate(
            question=request.retrieval_query_text,
            title=evidence.title,
            text=evidence.model_content,
        )
        if supplied_score != oracle_score or supplied_terms != oracle_terms:
            return None
        if access.permission_visibility == 'visible':
            visible = ServingEvidenceResolver(settings=self._settings).resolve_candidate(
                db=self._db,
                identity=identity,
                scope=request.security_scope,
            )
            if visible is None:
                return None
            evidence = visible
        return ClassifiedRetrievalCandidate(
            evidence=evidence,
            relevance_score=oracle_score,
            matched_terms=oracle_terms,
            access=access,
        )


def _projection_model_text(row: RagLexicalServingProjection) -> str:
    prefix = f'{row.title_lower}\n'
    if not row.searchable_lower.startswith(prefix):
        return ''
    return row.searchable_lower[len(prefix) :]


def _projection_matches(
    row: Mapping[str, object] | RagLexicalServingProjection,
    *,
    evidence: object,
) -> bool:
    return bool(
        _row_value(row, 'serving_document_id') == evidence.serving_document_id
        and _row_value(row, 'serving_kind') == evidence.serving_kind
        and _row_value(row, 'support_mode') == evidence.support_mode
        and _row_value(row, 'effective_permission') == evidence.effective_permission
        and _row_value(row, 'serving_identity_hmac')
        == evidence.serving_identity_hmac
        and _row_value(row, 'serving_version_fingerprint')
        == evidence.serving_version_fingerprint
        and _row_value(row, 'model_content_hmac') == evidence.model_content_hmac
        and _row_value(row, 'canonical_citation_projection_hmac')
        == evidence.canonical_citation_projection_hmac
        and _row_value(row, 'title_lower') == evidence.title.lower()
        and _row_value(row, 'searchable_lower')
        == f'{evidence.title}\n{evidence.model_content}'.lower()
    )


def _row_value(
    row: Mapping[str, object] | RagLexicalServingProjection,
    key: str,
) -> object:
    if isinstance(row, Mapping):
        return row[key]
    return getattr(row, key)


def _candidate_order(candidate: ClassifiedRetrievalCandidate) -> tuple[object, ...]:
    return (
        0 if candidate.evidence.serving_kind == 'trusted_knowledge' else 1,
        -candidate.relevance_score,
        candidate.evidence.serving_document_id,
    )


def _classify_trusted_stage_one_access(
    authorizer: TrustedEvidenceAuthorizer,
    *,
    scope: SecurityScope,
    envelope: TrustedServingEnvelope,
) -> EvidenceAccessClassification:
    resource_classification = authorizer.classify_resource_access(scope, envelope)
    permission = known_permission(envelope.identity.effective_permission)
    if permission is None:
        visibility = 'unknown_permission'
    elif permission in scope.allowed_permission_levels:
        visibility = 'visible'
    else:
        visibility = 'denied_known'
    return EvidenceAccessClassification(
        global_eligibility=resource_classification.global_eligibility,
        resource_scope=resource_classification.resource_scope,
        permission_visibility=visibility,
    )


def _is_stage_one_candidate(access: EvidenceAccessClassification) -> bool:
    return bool(
        access.global_eligibility == 'eligible'
        and access.resource_scope == 'in_scope'
        and access.permission_visibility in {'visible', 'denied_known'}
    )


def _is_statement_timeout(exc: DBAPIError) -> bool:
    original = exc.orig
    sqlstate = getattr(original, 'sqlstate', None) or getattr(original, 'pgcode', None)
    return sqlstate == '57014'
