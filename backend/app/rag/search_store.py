from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.rag.embeddings import OpenAIEmbeddingConfig, OpenAIEmbeddingModel
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore


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
