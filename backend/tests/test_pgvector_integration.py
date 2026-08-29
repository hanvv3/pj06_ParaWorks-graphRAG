import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.admin.auto_review_keys import (
    fingerprint_key_material_verifier,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.db.base import Base
from backend.app.models import (
    AutoReviewRuntimeKeyState,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    ReviewItem,
    Source,
)
from backend.app.rag.embeddings import DeterministicHashEmbeddingModel
from backend.app.rag.indexing import index_changed_vector_documents
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.rag.vector_store import VectorDocument


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_reindex_path_with_fake_embedding() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    test_id = uuid4().hex[:8]
    table_name = f'rag_vector_documents_test_{test_id}'
    settings = Settings(database_url=database_url)

    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            signature = 'a' * 64
            verifier = fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            )
            db.add(
                AutoReviewRuntimeKeyState(
                    component='auto_review_trust_promotion',
                    fingerprint_key_version=(
                        settings.agent_runtime_fingerprint_key_version
                    ),
                    fingerprint_key_material_verifier=verifier,
                    generation=1,
                    ready=True,
                )
            )
            source = Source(
                source_type='gmail',
                source_id=f'gmail:pgvector:{test_id}',
                source_url='https://pgvector.mock/company-memory',
                title='Current pgvector evidence',
                permission_level='internal',
                raw_metadata={},
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=signature,
            )
            db.add(source)
            db.flush([source])
            db.add(
                ReviewItem(
                    item_type='document_summary',
                    payload={'source_ids': [source.source_id]},
                    source_links=[source.source_url],
                    source_snippets=['pgvector stores durable company memory'],
                    confidence_score=1.0,
                    permission_level='internal',
                    status='approved',
                    resolution_source='human',
                )
            )
            document = Document(
                source_id=source.id,
                title=source.title,
                current_version='v1',
            )
            db.add(document)
            db.flush([document])
            version = DocumentVersion(
                document_id=document.id,
                version='v1',
                body=(
                    'PostgreSQL pgvector stores durable company memory '
                    'embeddings.'
                ),
            )
            db.add(version)
            db.flush([version])
            parser_run = DocumentParserRun(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='plain_text',
                parser_status='parsed',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=signature,
                parser_policy_version='parser-policy:v1',
                parser_version='plain-text:v1',
                chunk_policy_version='chunk-policy:v1',
            )
            db.add(parser_run)
            db.flush([parser_run])
            chunk = DocumentChunk(
                version_id=version.id,
                source_id=source.id,
                parser_run_id=parser_run.id,
                chunk_index=0,
                text=version.body,
                source_snippet='pgvector stores durable company memory',
                permission_level='internal',
                metadata_={},
            )
            db.add(chunk)
            db.flush([chunk])
            document.current_document_version_id = version.id
            parser_run.chunk_count = 1
            db.commit()

            store = PgVectorStore(
                session=db,
                config=PgVectorConfig(
                    table_name=table_name, embedding_dimensions=8
                ),
                settings=settings,
            )
            store.ensure_schema()
            db.commit()
            document_id = f'chunk:{chunk.id}'
            result = index_changed_vector_documents(
                db=db,
                documents=[
                    VectorDocument(
                        document_id=document_id,
                        text=chunk.text,
                        source_url=source.source_url,
                        source_snippet=chunk.source_snippet,
                        permission_level='internal',
                        metadata={
                            'chunk_id': chunk.id,
                            'source_pk': source.id,
                            'source_type': source.source_type,
                        },
                    )
                ],
                writer=store,
                embedding_model=DeterministicHashEmbeddingModel(
                    dimensions=8
                ),
                embedding_model_name='deterministic-hash:integration',
                settings=settings,
            )

            search_result = store.search_with_embedding(
                query_embedding=DeterministicHashEmbeddingModel(
                    dimensions=8
                ).embed('durable company memory'),
                user=USERS['viewer'],
                limit=5,
            )

            assert result.indexed_count == 1
            assert result.embedding_request_count == 1
            assert search_result.matches[0].document.document_id == document_id

            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
