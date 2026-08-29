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
        ]
        hidden_match_count = int(rows[0]['hidden_match_count']) if rows else 0
        return VectorSearchResult(matches=matches, hidden_match_count=hidden_match_count)

    def _upsert_sql(self) -> str:
        table = self.config.table_name
        return f"""
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
            :document_id,
            :text,
            :source_url,
            :source_snippet,
            :permission_level,
            CAST(:metadata_json AS jsonb),
            CAST(:embedding AS vector),
            now()
        WHERE NOT EXISTS (
            SELECT 1
            FROM vector_serving_tombstones
            WHERE vector_serving_tombstones.document_id = :document_id
        )
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
        )
        SELECT
            ranked.document_id,
            ranked.text,
            ranked.source_url,
            ranked.source_snippet,
            ranked.permission_level,
            ranked.metadata_json,
            ranked.score,
            hidden.hidden_match_count
        FROM ranked
        CROSS JOIN hidden
        WHERE permission_level = ANY(:allowed_permissions)
        ORDER BY ranked.distance
        LIMIT :limit;
        """

    def _live_eligibility_sql(self) -> str:
        table = self.config.table_name
        permission_rank = (
            "CASE {value} WHEN 'public' THEN 0 WHEN 'internal' THEN 1 "
            "WHEN 'restricted' THEN 2 ELSE 99 END"
        )
        vector_rank = permission_rank.format(value=f'{table}.permission_level')
        source_rank = permission_rank.format(value='sources.permission_level')
        chunk_rank = permission_rank.format(value='document_chunks.permission_level')
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
        ) OR (
            {table}.document_id NOT LIKE 'chunk:%'
            AND EXISTS (
                SELECT 1
                FROM trusted_knowledge_approval_links approval_links
                JOIN review_items ON review_items.id = approval_links.review_item_id
                WHERE approval_links.active = true
                  AND approval_links.knowledge_type = split_part({table}.document_id, ':', 1)
                  AND approval_links.knowledge_id::text = split_part({table}.document_id, ':', 2)
                  AND review_items.status = 'approved'
                  AND review_items.resolution_source = approval_links.resolution_source
                  AND NOT EXISTS (
                      SELECT 1 FROM auto_review_post_audits audits
                      WHERE audits.review_item_id = review_items.id
                        AND (audits.status = 'remediation_required'
                             OR audits.outcome IN ('incorrect', 'permission_violation',
                                'source_version_violation', 'policy_violation'))
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM auto_review_audit_corrections corrections
                      WHERE corrections.review_item_id = review_items.id
                        AND corrections.effective_outcome IN (
                            'incorrect', 'permission_violation',
                            'source_version_violation', 'policy_violation'
                        )
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM trusted_knowledge_evidence_links evidence_links
                      JOIN sources
                        ON sources.id::text = evidence_links.canonical_source_id
                       AND sources.source_type = evidence_links.canonical_source_kind
                       AND sources.server_content_signature_schema = 'server-source-content:v1'
                      WHERE evidence_links.approval_link_id = approval_links.id
                        AND (
                            sources.server_content_signature = evidence_links.canonical_version_or_signature
                            OR EXISTS (
                                SELECT 1
                                FROM document_parser_runs evidence_parser_runs
                                JOIN documents evidence_documents
                                  ON evidence_documents.id = evidence_parser_runs.document_id
                                 AND evidence_documents.source_id = sources.id
                                 AND evidence_documents.current_document_version_id = evidence_parser_runs.document_version_id
                                WHERE evidence_parser_runs.source_id = sources.id
                                  AND evidence_parser_runs.server_content_signature_schema = 'server-source-content:v1'
                                  AND evidence_parser_runs.server_content_signature = sources.server_content_signature
                                  AND evidence_parser_runs.parser_policy_version IS NOT NULL
                                  AND evidence_parser_runs.parser_version IS NOT NULL
                                  AND evidence_parser_runs.chunk_policy_version IS NOT NULL
                                  AND evidence_parser_runs.revision_id = evidence_links.canonical_version_or_signature
                            )
                        )
                        AND sources.permission_level IN ('public', 'internal', 'restricted')
                        AND (
                            approval_links.resolution_source = 'human'
                            OR (
                                approval_links.resolution_source = 'auto_policy'
                                AND sources.permission_level IN ('public', 'internal')
                                AND EXISTS (
                                    SELECT 1 FROM auto_review_validations validations
                                    WHERE validations.id = review_items.auto_validation_id
                                      AND validations.review_item_id = review_items.id
                                      AND validations.status = 'completed'
                                      AND validations.policy_decision IN ('auto_approve', 'reuse_trusted')
                                )
                            )
                        )
                        AND {vector_rank} >= {source_rank}
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM trusted_knowledge_evidence_links expected_evidence
                      LEFT JOIN sources
                        ON sources.id::text = expected_evidence.canonical_source_id
                       AND sources.source_type = expected_evidence.canonical_source_kind
                       AND sources.server_content_signature_schema = 'server-source-content:v1'
                       AND (
                           sources.server_content_signature = expected_evidence.canonical_version_or_signature
                           OR EXISTS (
                               SELECT 1
                               FROM document_parser_runs expected_parser_runs
                               JOIN documents expected_documents
                                 ON expected_documents.id = expected_parser_runs.document_id
                                AND expected_documents.source_id = sources.id
                                AND expected_documents.current_document_version_id = expected_parser_runs.document_version_id
                               WHERE expected_parser_runs.source_id = sources.id
                                 AND expected_parser_runs.server_content_signature_schema = 'server-source-content:v1'
                                 AND expected_parser_runs.server_content_signature = sources.server_content_signature
                                 AND expected_parser_runs.parser_policy_version IS NOT NULL
                                 AND expected_parser_runs.parser_version IS NOT NULL
                                 AND expected_parser_runs.chunk_policy_version IS NOT NULL
                                 AND expected_parser_runs.revision_id = expected_evidence.canonical_version_or_signature
                           )
                       )
                       AND sources.permission_level IN ('public', 'internal', 'restricted')
                      WHERE expected_evidence.approval_link_id = approval_links.id
                        AND (
                            sources.id IS NULL
                            OR (
                                approval_links.resolution_source = 'auto_policy'
                                AND sources.permission_level = 'restricted'
                            )
                        )
                  )
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
