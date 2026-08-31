import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_v2_identity import (
    verify_serialized_security_scope_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_authority import (
    postgres_exact_source_authority_sql,
)
from backend.app.knowledge.trusted_serving_eligibility import (
    knowledge_type_storage_aliases,
)
from backend.app.rag.embeddings import (
    ValidatedQueryEmbeddingVector,
    validate_query_embedding_vector_carrier,
)
from backend.app.rag.retrieval import RetrievalRequest
from backend.app.rag.serving_locks import (
    VectorServingLockedContext,
    VectorServingLockManager,
)
from backend.app.rag.vector_store import VectorDocument, VectorMatch, VectorSearchResult
from backend.app.rag.vector_validation import CosineIndexableVectorValidator


@dataclass(frozen=True)
class PgVectorConfig:
    table_name: str = 'rag_vector_documents'
    embedding_dimensions: int = 1536

    def __post_init__(self) -> None:
        if not re.fullmatch(r'[a-z_][a-z0-9_]*', self.table_name):
            raise ValueError('table_name must be a safe SQL identifier')
        if self.embedding_dimensions <= 0:
            raise ValueError('embedding_dimensions must be positive')


@dataclass(frozen=True, slots=True)
class PgVectorServingCandidateRow:
    serving_document_id: str
    serving_kind: str
    support_mode: str
    effective_permission: str
    serving_identity_hmac: str
    serving_version_fingerprint: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    title_lower: str
    searchable_lower: str
    score: float


class PgVectorStore:
    def __init__(
        self,
        *,
        session: Session,
        config: PgVectorConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.config = config or PgVectorConfig()
        self._settings = settings
        self._lock_manager = (
            VectorServingLockManager(db=session, settings=settings)
            if settings is not None
            else None
        )

    def schema_sql(self) -> list[str]:
        table = self.config.table_name
        dimensions = self.config.embedding_dimensions
        return [
            'CREATE EXTENSION IF NOT EXISTS vector;',
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                document_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_snippet TEXT NOT NULL,
                permission_level TEXT NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dimensions}) NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {table}_embedding_idx
            ON {table}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {table}_permission_idx
            ON {table} (permission_level);
            """,
        ]

    def ensure_schema(self) -> None:
        for statement in self.schema_sql():
            self.session.execute(text(statement))

    def upsert_with_embedding(
        self,
        document: VectorDocument,
        embedding: list[float],
        *,
        locked_context: VectorServingLockedContext | None = None,
    ) -> None:
        canonical_embedding = CosineIndexableVectorValidator().validate(
            embedding,
            expected_dimensions=self.config.embedding_dimensions,
        )
        self._validate_mutation_context(locked_context, [document.document_id])
        self.session.execute(
            text(self._upsert_sql()),
            {
                'document_id': document.document_id,
                'text': document.text,
                'source_url': document.source_url,
                'source_snippet': document.source_snippet,
                'permission_level': document.permission_level,
                'metadata_json': json.dumps(document.metadata),
                'embedding': _embedding_literal(canonical_embedding),
            },
        )

    def delete_many(
        self,
        document_ids: Sequence[str],
        *,
        locked_context: VectorServingLockedContext | None = None,
    ) -> int:
        normalized = sorted(set(document_ids))
        if not normalized:
            return 0
        self._validate_mutation_context(locked_context, normalized)
        result = self.session.execute(
            text(
                f'DELETE FROM {self.config.table_name} '
                'WHERE document_id = ANY(:document_ids)'
            ),
            {'document_ids': normalized},
        )
        return max(int(getattr(result, 'rowcount', 0) or 0), 0)

    def load_embedding(
        self,
        document_id: str,
        *,
        locked_context: VectorServingLockedContext | None = None,
    ) -> list[float] | None:
        self._validate_mutation_context(locked_context, [document_id])
        rows = (
            self.session.execute(
                text(
                    f'SELECT embedding::text AS embedding_text '
                    f'FROM {self.config.table_name} '
                    'WHERE document_id = :document_id FOR UPDATE'
                ),
                {'document_id': document_id},
            )
            .mappings()
            .all()
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError('pgvector document identity is not unique')
        parsed = _parse_embedding_literal(str(rows[0]['embedding_text']))
        canonical = CosineIndexableVectorValidator().validate(
            parsed,
            expected_dimensions=self.config.embedding_dimensions,
        )
        return list(canonical)

    def narrow_permissions(
        self,
        document_ids: Sequence[str],
        permission_level: str,
        *,
        locked_context: VectorServingLockedContext | None = None,
    ) -> int:
        permission_rank = _permission_rank(permission_level)
        normalized = sorted(set(document_ids))
        if not normalized:
            return 0
        self._validate_mutation_context(locked_context, normalized)
        current_rows = (
            self.session.execute(
                text(
                    f'SELECT document_id, permission_level '
                    f'FROM {self.config.table_name} '
                    'WHERE document_id = ANY(:document_ids)'
                ),
                {'document_ids': normalized},
            )
            .mappings()
            .all()
        )
        if any(
            _permission_rank(str(row['permission_level'])) > permission_rank
            for row in current_rows
        ):
            raise ValueError('permission broadening is not allowed')
        result = self.session.execute(
            text(
                f'UPDATE {self.config.table_name} '
                'SET permission_level = :permission_level, updated_at = now() '
                'WHERE document_id = ANY(:document_ids) AND '
                "CASE permission_level WHEN 'public' THEN 0 "
                "WHEN 'internal' THEN 1 WHEN 'restricted' THEN 2 ELSE 3 END "
                '<= :permission_rank'
            ),
            {
                'document_ids': normalized,
                'permission_level': permission_level,
                'permission_rank': permission_rank,
            },
        )
        return max(int(getattr(result, 'rowcount', 0) or 0), 0)

    def _validate_mutation_context(
        self,
        context: VectorServingLockedContext | None,
        document_ids: Sequence[str],
    ) -> None:
        bind = getattr(self.session, 'bind', None)
        if bind is None or bind.dialect.name != 'postgresql':
            return
        if self._lock_manager is None or context is None:
            raise TypeError(
                'PostgreSQL serving mutations require a locked context'
            )
        self._lock_manager.validate_locked_context(context, document_ids)

    def search_with_embedding(
        self,
        *,
        query_embedding: list[float],
        user: DemoUser,
        limit: int = 5,
    ) -> VectorSearchResult:
        rows = (
            self.session.execute(
                text(self._search_sql()),
                {
                    'query_embedding': _embedding_literal(query_embedding),
                    'allowed_permissions': _allowed_permissions_for_user(user),
                    'limit': limit,
                },
            )
            .mappings()
            .all()
        )
        matches = [
            VectorMatch(
                document=VectorDocument(
                    document_id=row['document_id'],
                    text=row['text'],
                    source_url=row['source_url'],
                    source_snippet=row['source_snippet'],
                    permission_level=row['permission_level'],
                    metadata=_metadata_from_row(row['metadata_json']),
                ),
                score=float(row['score']),
            )
            for row in rows
            if row['document_id'] is not None
        ]
        hidden_match_count = int(rows[0]['hidden_match_count']) if rows else 0
        return VectorSearchResult(matches=matches, hidden_match_count=hidden_match_count)

    def search_rag_v2(
        self,
        *,
        request: RetrievalRequest,
        query_embedding: ValidatedQueryEmbeddingVector,
    ) -> tuple[PgVectorServingCandidateRow, ...]:
        """Return the permission-unfiltered top 50 from the D serving corpus."""
        settings = self._required_settings()
        verify_serialized_security_scope_fingerprint(
            request.security_scope,
            serialized_fingerprint=request.security_scope_fingerprint,
            settings=settings,
        )
        validate_query_embedding_vector_carrier(
            query_embedding,
            expected_dimensions=self.config.embedding_dimensions,
        )
        # The declarative resource predicate is the single Task 6 authority.
        # This local import avoids making the legacy writer depend on search assembly.
        from backend.app.rag.search_store import (
            POSTGRES_KEYWORD_RESOURCE_PREDICATE,
            build_postgres_keyword_bind_values,
            render_postgres_keyword_predicate,
        )

        bind_values = build_postgres_keyword_bind_values(
            request=request,
            terms=(),
            settings=settings,
        )
        parameters = {
            'query_embedding': _embedding_literal(query_embedding.coordinates),
            'fingerprint_key_version': bind_values['fingerprint_key_version'],
            'key_material_verifier': bind_values['key_material_verifier'],
            'project_keys': bind_values['project_keys'],
            'source_ids': bind_values['source_ids'],
            'source_id_texts': bind_values['source_id_texts'],
            'workspace_scope_id': bind_values['workspace_scope_id'],
        }
        rows = tuple(
            self.session.execute(
                text(
                    self._rag_v2_search_sql(
                        resource_predicate=render_postgres_keyword_predicate(
                            POSTGRES_KEYWORD_RESOURCE_PREDICATE
                        )
                    )
                ),
                parameters,
            )
            .mappings()
            .all()
        )
        return tuple(
            PgVectorServingCandidateRow(
                serving_document_id=str(row['serving_document_id']),
                serving_kind=str(row['serving_kind']),
                support_mode=str(row['support_mode']),
                effective_permission=str(row['effective_permission']),
                serving_identity_hmac=str(row['serving_identity_hmac']),
                serving_version_fingerprint=str(
                    row['serving_version_fingerprint']
                ),
                model_content_hmac=str(row['model_content_hmac']),
                canonical_citation_projection_hmac=str(
                    row['canonical_citation_projection_hmac']
                ),
                title_lower=str(row['title_lower']),
                searchable_lower=str(row['searchable_lower']),
                score=float(row['score']),
            )
            for row in rows
        )

    def _required_settings(self) -> Settings:
        if self._settings is None:
            raise TypeError('RAG V2 search requires explicit settings')
        return self._settings

    def _rag_v2_search_sql(self, *, resource_predicate: str) -> str:
        table = self.config.table_name
        live_eligibility = self._live_eligibility_sql(
            'vector_document',
            allow_rag_v2_raw_write=True,
        )
        return f"""
        WITH ranked AS (
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
                projection.searchable_lower,
                1 - (
                    vector_document.embedding <=> CAST(:query_embedding AS vector)
                ) AS score
            FROM {table} AS vector_document
            JOIN rag_lexical_serving_projections AS projection
              ON projection.serving_document_id = vector_document.document_id
            JOIN rag_serving_corpus_generations AS generation
              ON generation.id = projection.corpus_generation_id
             AND generation.corpus_generation = projection.corpus_generation
            WHERE projection.corpus_generation_id = 1
              AND projection.fingerprint_key_version = :fingerprint_key_version
              AND projection.fingerprint_key_material_verifier = :key_material_verifier
              AND projection.effective_permission IN (
                  'public', 'internal', 'restricted'
              )
              AND vector_document.metadata_json->>'index_policy_version'
                  = 'rag-v2-serving-index:v1'
              AND vector_document.metadata_json->>'serving_identity_hmac'
                  = projection.serving_identity_hmac
              AND vector_document.metadata_json->>'serving_version_fingerprint'
                  = projection.serving_version_fingerprint
              AND vector_document.metadata_json->>'model_content_hmac'
                  = projection.model_content_hmac
              AND vector_document.metadata_json->>'canonical_citation_projection_hmac'
                  = projection.canonical_citation_projection_hmac
              AND NOT EXISTS (
                  SELECT 1
                  FROM vector_serving_tombstones AS tombstone
                  WHERE tombstone.document_id = vector_document.document_id
              )
              AND ({live_eligibility})
              AND ({resource_predicate})
        ), relevant AS (
            SELECT *
            FROM ranked
            WHERE score >= 0.25
            ORDER BY
              CASE WHEN serving_kind = 'trusted_knowledge' THEN 0 ELSE 1 END,
              score DESC,
              serving_document_id
            LIMIT 50
        )
        SELECT * FROM relevant
        ORDER BY
          CASE WHEN serving_kind = 'trusted_knowledge' THEN 0 ELSE 1 END,
          score DESC,
          serving_document_id
        """

    def _upsert_sql(self) -> str:
        table = self.config.table_name
        return f"""
        WITH candidate AS (
            SELECT
                CAST(:document_id AS text) AS document_id,
                CAST(:text AS text) AS text,
                CAST(:source_url AS text) AS source_url,
                CAST(:source_snippet AS text) AS source_snippet,
                CAST(:permission_level AS text) AS permission_level,
                CAST(:metadata_json AS jsonb) AS metadata_json,
                CAST(:embedding AS vector) AS embedding
        )
        INSERT INTO {table} (
            document_id,
            text,
            source_url,
            source_snippet,
            permission_level,
            metadata_json,
            embedding,
            updated_at
        )
        SELECT
            candidate.document_id,
            candidate.text,
            candidate.source_url,
            candidate.source_snippet,
            candidate.permission_level,
            candidate.metadata_json,
            candidate.embedding,
            now()
        FROM candidate
        WHERE NOT EXISTS (
            SELECT 1
            FROM vector_serving_tombstones
            WHERE vector_serving_tombstones.document_id = candidate.document_id
        )
        AND ({self._live_eligibility_sql('candidate', allow_rag_v2_raw_write=True)})
        ON CONFLICT (document_id) DO UPDATE SET
            text = EXCLUDED.text,
            source_url = EXCLUDED.source_url,
            source_snippet = EXCLUDED.source_snippet,
            permission_level = EXCLUDED.permission_level,
            metadata_json = EXCLUDED.metadata_json,
            embedding = EXCLUDED.embedding,
            updated_at = now();
        """

    def _search_sql(self) -> str:
        table = self.config.table_name
        return f"""
        WITH ranked AS (
            SELECT
                document_id,
                text,
                source_url,
                source_snippet,
                permission_level,
                metadata_json,
                embedding <=> CAST(:query_embedding AS vector) AS distance,
                1 - (embedding <=> CAST(:query_embedding AS vector)) AS score,
                permission_level = ANY(:allowed_permissions) AS is_visible
            FROM {table}
            WHERE NOT EXISTS (
                SELECT 1
                FROM vector_serving_tombstones
                WHERE vector_serving_tombstones.document_id = {table}.document_id
            )
            AND ({self._live_eligibility_sql()})
        ),
        hidden AS (
            SELECT count(*) AS hidden_match_count
            FROM ranked
            WHERE NOT is_visible
        ),
        visible AS (
            SELECT *
            FROM ranked
            WHERE is_visible
            ORDER BY distance
            LIMIT :limit
        )
        SELECT
            visible.document_id,
            visible.text,
            visible.source_url,
            visible.source_snippet,
            visible.permission_level,
            visible.metadata_json,
            visible.score,
            hidden.hidden_match_count
        FROM hidden
        LEFT JOIN visible ON true
        ORDER BY visible.distance;
        """

    def _live_eligibility_sql(
        self,
        table: str | None = None,
        *,
        allow_rag_v2_raw_write: bool = False,
    ) -> str:
        table = table or self.config.table_name
        permission_rank = (
            "CASE {value} WHEN 'public' THEN 0 WHEN 'internal' THEN 1 "
            "WHEN 'restricted' THEN 2 ELSE 99 END"
        )
        vector_rank = permission_rank.format(value=f'{table}.permission_level')
        source_rank = permission_rank.format(value='sources.permission_level')
        chunk_rank = permission_rank.format(value='document_chunks.permission_level')
        raw_source_authority = postgres_exact_source_authority_sql(
            source_alias='sources',
            prefix='raw_authority',
            required_chunk_id_sql='document_chunks.id',
        )
        legacy_raw_authorization = """
            EXISTS (
                SELECT 1
                FROM review_items raw_reviews
                WHERE raw_reviews.status = 'approved'
                  AND (
                      raw_reviews.resolution_source IS NULL
                      OR raw_reviews.resolution_source = 'human'
                  )
                  AND COALESCE(
                      raw_reviews.payload::jsonb->'source_ids',
                      '[]'::jsonb
                  ) ? sources.source_id
            )
        """
        raw_authorization = legacy_raw_authorization
        if allow_rag_v2_raw_write:
            raw_authorization = f"""
                (
                    (
                        {table}.metadata_json->>'index_policy_version'
                        = 'rag-v2-serving-index:v1'
                        AND {table}.metadata_json->>'serving_kind' = 'raw_chunk'
                        AND {table}.metadata_json->>'support_mode'
                            = 'source_observation'
                        AND sources.source_type IN (
                            'gmail', 'gmail_attachment', 'drive', 'calendar'
                        )
                    )
                    OR (
                        COALESCE(
                            {table}.metadata_json->>'index_policy_version', ''
                        ) <> 'rag-v2-serving-index:v1'
                        AND ({legacy_raw_authorization})
                    )
                )
            """
        knowledge_branches = ' OR '.join(
            self._knowledge_target_branch(
                table=table,
                knowledge_type=knowledge_type,
                target_table=target_table,
                vector_rank=vector_rank,
                permission_rank=permission_rank,
            )
            for knowledge_type, target_table in (
                ('decision_record', 'decision_records'),
                ('history_event', 'history_events'),
                ('timeline_event', 'timeline_events'),
                ('todo', 'todos'),
            )
        )
        return f"""
        (
            {table}.document_id LIKE 'chunk:%'
            AND EXISTS (
                SELECT 1
                FROM document_chunks
                JOIN sources ON sources.id = document_chunks.source_id
                WHERE {table}.document_id = 'chunk:' || document_chunks.id::text
                  AND ({raw_source_authority})
                  AND sources.source_type <> 'slack'
                  AND sources.permission_level IN ('public', 'internal', 'restricted')
                  AND document_chunks.permission_level IN ('public', 'internal', 'restricted')
                  AND ({raw_authorization})
                  AND {vector_rank} >= GREATEST({source_rank}, {chunk_rank})
            )
        ) OR ({knowledge_branches})
        """

    def _knowledge_target_branch(
        self,
        *,
        table: str,
        knowledge_type: str,
        target_table: str,
        vector_rank: str,
        permission_rank: str,
    ) -> str:
        stored_knowledge_types = ', '.join(
            f"'{alias}'" for alias in knowledge_type_storage_aliases(knowledge_type)
        )
        target_rank = permission_rank.format(
            value='knowledge_targets.permission_level'
        )
        item_rank = permission_rank.format(
            value='effect_reviews.permission_level'
        )
        link_rank = permission_rank.format(
            value='approval_links.permission_level'
        )
        source_rank = permission_rank.format(value='effect_sources.permission_level')
        workflow_snapshot_rank = permission_rank.format(
            value='workflow_evidence.permission_level_snapshot'
        )
        legacy_item_rank = permission_rank.format(
            value='legacy_reviews.permission_level'
        )
        current_source = self._current_evidence_source_sql(
            source_alias='effect_sources',
            evidence_alias='evidence_links',
            parser_prefix='effect',
        )
        expected_source = self._current_evidence_source_sql(
            source_alias='expected_sources',
            evidence_alias='expected_evidence',
            parser_prefix='expected',
        )
        return f"""
        EXISTS (
            SELECT 1
            FROM {target_table} knowledge_targets
            WHERE {table}.document_id = '{knowledge_type}:' || knowledge_targets.id::text
              AND knowledge_targets.review_status = 'approved'
              AND knowledge_targets.permission_level IN ('public', 'internal', 'restricted')
              AND {vector_rank} >= {target_rank}
              AND (
                  EXISTS (
                      SELECT 1
                      FROM trusted_knowledge_approval_links approval_links
                      JOIN review_items effect_reviews
                        ON effect_reviews.id = approval_links.review_item_id
                      WHERE approval_links.active = true
                        AND approval_links.knowledge_type IN ({stored_knowledge_types})
                        AND approval_links.knowledge_id = knowledge_targets.id
                        AND effect_reviews.status = 'approved'
                        AND effect_reviews.resolution_source = approval_links.resolution_source
                        AND effect_reviews.permission_level IN ('public', 'internal', 'restricted')
                        AND approval_links.permission_level IN ('public', 'internal', 'restricted')
                        AND {vector_rank} >= GREATEST({item_rank}, {link_rank})
                        AND NOT EXISTS (
                            SELECT 1
                            FROM review_item_evidence_refs workflow_links
                            JOIN agent_workflow_evidence_refs workflow_evidence
                              ON workflow_evidence.id = workflow_links.workflow_evidence_ref_id
                             AND workflow_evidence.workflow_thread_id = workflow_links.workflow_thread_id
                            WHERE workflow_links.review_item_id = effect_reviews.id
                              AND workflow_links.workflow_thread_id = effect_reviews.workflow_thread_id
                              AND (
                                  workflow_evidence.permission_level_snapshot NOT IN ('public', 'internal', 'restricted')
                                  OR {vector_rank} < {workflow_snapshot_rank}
                              )
                        )
                        AND (
                            (
                                approval_links.resolution_source = 'human'
                                AND effect_reviews.candidate_contract_version = 'c5-v1'
                            )
                            OR (
                                approval_links.resolution_source = 'auto_policy'
                                AND EXISTS (
                                    SELECT 1
                                    FROM auto_review_validations validations
                                    WHERE validations.id = effect_reviews.auto_validation_id
                                      AND validations.review_item_id = effect_reviews.id
                                      AND validations.status = 'completed'
                                      AND validations.policy_decision IN ('auto_approve', 'reuse_trusted')
                                )
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM auto_review_post_audits audits
                                    WHERE audits.review_item_id = effect_reviews.id
                                      AND (
                                          audits.status = 'remediation_required'
                                          OR audits.outcome IN (
                                              'incorrect', 'permission_violation',
                                              'source_version_violation', 'policy_violation'
                                          )
                                      )
                                )
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM auto_review_audit_corrections corrections
                                    WHERE corrections.review_item_id = effect_reviews.id
                                      AND corrections.effective_outcome IN (
                                          'incorrect', 'permission_violation',
                                          'source_version_violation', 'policy_violation'
                                      )
                                )
                            )
                        )
                        AND EXISTS (
                            SELECT 1
                            FROM trusted_knowledge_evidence_links evidence_links
                            JOIN sources effect_sources
                              ON effect_sources.id::text = evidence_links.canonical_source_id
                             AND effect_sources.source_type = evidence_links.canonical_source_kind
                             AND {current_source}
                            WHERE evidence_links.approval_link_id = approval_links.id
                              AND effect_sources.permission_level IN ('public', 'internal', 'restricted')
                              AND {vector_rank} >= {source_rank}
                              AND (
                                  approval_links.resolution_source = 'human'
                                  OR effect_sources.permission_level IN ('public', 'internal')
                              )
                        )
                        AND NOT EXISTS (
                            SELECT 1
                            FROM trusted_knowledge_evidence_links expected_evidence
                            LEFT JOIN sources expected_sources
                              ON expected_sources.id::text = expected_evidence.canonical_source_id
                             AND expected_sources.source_type = expected_evidence.canonical_source_kind
                             AND {expected_source}
                            WHERE expected_evidence.approval_link_id = approval_links.id
                              AND (
                                  expected_sources.id IS NULL
                                  OR expected_sources.permission_level NOT IN ('public', 'internal', 'restricted')
                                  OR {vector_rank} < {permission_rank.format(value='expected_sources.permission_level')}
                                  OR (
                                      approval_links.resolution_source = 'auto_policy'
                                      AND expected_sources.permission_level = 'restricted'
                                  )
                              )
                        )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM review_items legacy_reviews
                      WHERE legacy_reviews.id = knowledge_targets.source_review_item_id
                        AND legacy_reviews.status = 'approved'
                        AND (
                            legacy_reviews.resolution_source IS NULL
                            OR legacy_reviews.resolution_source = 'human'
                        )
                        AND legacy_reviews.candidate_contract_version IS DISTINCT FROM 'c5-v1'
                        AND legacy_reviews.permission_level IN ('public', 'internal', 'restricted')
                        AND {vector_rank} >= {legacy_item_rank}
                  )
              )
        )
        """

    def _current_evidence_source_sql(
        self,
        *,
        source_alias: str,
        evidence_alias: str,
        parser_prefix: str,
    ) -> str:
        return postgres_exact_source_authority_sql(
            source_alias=source_alias,
            prefix=f'{parser_prefix}_authority',
            evidence_ref_sql=(
                f'{evidence_alias}.canonical_version_or_signature'
            ),
        )


def _embedding_literal(embedding: Sequence[float]) -> str:
    return '[' + ','.join(repr(float(value)) for value in embedding) + ']'


def _parse_embedding_literal(value: str) -> tuple[float, ...]:
    if not value.startswith('[') or not value.endswith(']'):
        raise ValueError('pgvector embedding representation is invalid')
    body = value[1:-1]
    if not body:
        return ()
    return tuple(float(part) for part in body.split(','))


def _allowed_permissions_for_user(user: DemoUser) -> list[str]:
    if user.role == 'admin':
        return ['public', 'internal', 'restricted']
    return ['public', 'internal']


def _metadata_from_row(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _permission_rank(permission_level: str) -> int:
    try:
        return {'public': 0, 'internal': 1, 'restricted': 2}[permission_level]
    except KeyError:
        raise ValueError('permission level is not writable or servable') from None
