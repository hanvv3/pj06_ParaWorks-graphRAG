import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.rag.serving_locks import (
    VectorServingLockedContext,
    VectorServingLockManager,
)
from backend.app.rag.vector_store import VectorDocument, VectorMatch, VectorSearchResult


@dataclass(frozen=True)
class PgVectorConfig:
    table_name: str = 'rag_vector_documents'
    embedding_dimensions: int = 1536

    def __post_init__(self) -> None:
        if not re.fullmatch(r'[a-z_][a-z0-9_]*', self.table_name):
            raise ValueError('table_name must be a safe SQL identifier')
        if self.embedding_dimensions <= 0:
            raise ValueError('embedding_dimensions must be positive')


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
                'embedding': _embedding_literal(embedding),
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
        AND ({self._live_eligibility_sql('candidate')})
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

    def _live_eligibility_sql(self, table: str | None = None) -> str:
        table = table or self.config.table_name
        permission_rank = (
            "CASE {value} WHEN 'public' THEN 0 WHEN 'internal' THEN 1 "
            "WHEN 'restricted' THEN 2 ELSE 99 END"
        )
        vector_rank = permission_rank.format(value=f'{table}.permission_level')
        source_rank = permission_rank.format(value='sources.permission_level')
        chunk_rank = permission_rank.format(value='document_chunks.permission_level')
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
                JOIN document_versions
                  ON document_versions.id = document_chunks.version_id
                JOIN documents
                  ON documents.id = document_versions.document_id
                 AND documents.current_document_version_id = document_versions.id
                JOIN document_parser_runs
                  ON document_parser_runs.id = document_chunks.parser_run_id
                 AND document_parser_runs.document_version_id = document_versions.id
                 AND document_parser_runs.source_id = document_chunks.source_id
                JOIN sources ON sources.id = document_chunks.source_id
                WHERE {table}.document_id = 'chunk:' || document_chunks.id::text
                  AND sources.server_content_signature_schema = 'server-source-content:v1'
                  AND sources.server_content_signature = document_parser_runs.server_content_signature
                  AND document_parser_runs.parser_policy_version IS NOT NULL
                  AND document_parser_runs.parser_version IS NOT NULL
                  AND document_parser_runs.chunk_policy_version IS NOT NULL
                  AND sources.source_type <> 'slack'
                  AND sources.permission_level IN ('public', 'internal', 'restricted')
                  AND document_chunks.permission_level IN ('public', 'internal', 'restricted')
                  AND EXISTS (
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
                        AND approval_links.knowledge_type = '{knowledge_type}'
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
        return f"""
        {source_alias}.server_content_signature_schema = 'server-source-content:v1'
        AND (
            {source_alias}.server_content_signature = {evidence_alias}.canonical_version_or_signature
            OR EXISTS (
                SELECT 1
                FROM document_parser_runs {parser_prefix}_parser_runs
                JOIN documents {parser_prefix}_documents
                  ON {parser_prefix}_documents.id = {parser_prefix}_parser_runs.document_id
                 AND {parser_prefix}_documents.source_id = {source_alias}.id
                 AND {parser_prefix}_documents.current_document_version_id = {parser_prefix}_parser_runs.document_version_id
                WHERE {parser_prefix}_parser_runs.source_id = {source_alias}.id
                  AND {parser_prefix}_parser_runs.server_content_signature_schema = 'server-source-content:v1'
                  AND {parser_prefix}_parser_runs.server_content_signature = {source_alias}.server_content_signature
                  AND {parser_prefix}_parser_runs.parser_policy_version IS NOT NULL
                  AND {parser_prefix}_parser_runs.parser_version IS NOT NULL
                  AND {parser_prefix}_parser_runs.chunk_policy_version IS NOT NULL
                  AND {parser_prefix}_parser_runs.revision_id = {evidence_alias}.canonical_version_or_signature
            )
        )
        """


def _embedding_literal(embedding: list[float]) -> str:
    return '[' + ','.join(str(float(value)).rstrip('0').rstrip('.') for value in embedding) + ']'


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
