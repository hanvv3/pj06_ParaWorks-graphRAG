import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from sqlalchemy import event
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser
from backend.app.permissions.service import can_access_permission


@dataclass(frozen=True)
class VectorDocument:
    document_id: str
    text: str
    source_url: str
    source_snippet: str
    permission_level: str
    metadata: dict


@dataclass(frozen=True)
class VectorMatch:
    document: VectorDocument
    score: float


@dataclass(frozen=True)
class VectorSearchResult:
    matches: list[VectorMatch]
    hidden_match_count: int


class VectorStore(Protocol):
    def upsert(self, document: VectorDocument) -> None:
        raise NotImplementedError

    def upsert_many(self, documents: list[VectorDocument]) -> None:
        raise NotImplementedError

    def search(self, *, query: str, user: DemoUser, limit: int = 5) -> VectorSearchResult:
        raise NotImplementedError

    def delete_many(self, document_ids: Sequence[str]) -> int:
        raise NotImplementedError

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        raise NotImplementedError


class InMemoryVectorStore:
    def __init__(self, *, session: Session | None = None) -> None:
        self._documents: dict[str, VectorDocument] = {}
        self._session = session

    def upsert(self, document: VectorDocument) -> None:
        self._documents[document.document_id] = document

    def upsert_many(self, documents: list[VectorDocument]) -> None:
        for document in documents:
            self.upsert(document)

    def delete_many(self, document_ids: Sequence[str]) -> int:
        normalized = sorted(set(document_ids))
        existing = [
            document_id for document_id in normalized if document_id in self._documents
        ]

        def apply() -> None:
            for document_id in existing:
                self._documents.pop(document_id, None)

        self._apply_or_queue(apply)
        deleted = len(existing)
        return deleted

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        target_rank = _permission_rank(permission_level)
        normalized = sorted(set(document_ids))
        documents = [
            self._documents[document_id]
            for document_id in normalized
            if document_id in self._documents
        ]
        if any(_permission_rank(document.permission_level) > target_rank for document in documents):
            raise ValueError('permission broadening is not allowed')
        def apply() -> None:
            for document in documents:
                self._documents[document.document_id] = VectorDocument(
                    document_id=document.document_id,
                    text=document.text,
                    source_url=document.source_url,
                    source_snippet=document.source_snippet,
                    permission_level=permission_level,
                    metadata=document.metadata,
                )

        self._apply_or_queue(apply)
        return len(documents)

    def _apply_or_queue(self, mutation: Callable[[], None]) -> None:
        if self._session is None:
            mutation()
            return
        _queue_after_commit(self._session, mutation)

    def search(self, *, query: str, user: DemoUser, limit: int = 5) -> VectorSearchResult:
        query_vector = _term_frequency_vector(query)
        ranked_matches = [
            VectorMatch(document=document, score=_cosine_similarity(query_vector, _term_frequency_vector(document.text)))
            for document in self._documents.values()
        ]
        positive_matches = [match for match in ranked_matches if match.score > 0]
        positive_matches.sort(key=lambda match: (-match.score, match.document.document_id))

        visible_matches = [
            match
            for match in positive_matches
            if can_access_permission(user, match.document.permission_level)
        ]
        hidden_match_count = len(positive_matches) - len(visible_matches)

        return VectorSearchResult(
            matches=visible_matches[:limit],
            hidden_match_count=hidden_match_count,
        )

    def export_documents(self) -> list[dict]:
        return [asdict(document) for document in self._documents.values()]


def _term_frequency_vector(text: str) -> dict[str, float]:
    terms = [term for term in re.findall(r'[a-zA-Z0-9가-힣]+', text.lower()) if len(term) >= 3]
    vector: dict[str, float] = {}
    for term in terms:
        vector[term] = vector.get(term, 0.0) + 1.0
    return vector


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    shared_terms = set(left).intersection(right)
    dot_product = sum(left[term] * right[term] for term in shared_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _permission_rank(permission_level: str) -> int:
    try:
        return {'public': 0, 'internal': 1, 'restricted': 2}[permission_level]
    except KeyError:
        raise ValueError('permission level is not writable or servable') from None


_MEMORY_MUTATIONS_INFO_KEY = 'paraworks_in_memory_vector_mutations'
_MEMORY_LISTENERS_INFO_KEY = 'paraworks_in_memory_vector_listeners'


def _queue_after_commit(session: Session, mutation: Callable[[], None]) -> None:
    session.info.setdefault(_MEMORY_MUTATIONS_INFO_KEY, []).append(mutation)
    if session.info.get(_MEMORY_LISTENERS_INFO_KEY):
        return

    def apply_after_commit(committed_session: Session) -> None:
        mutations = committed_session.info.pop(_MEMORY_MUTATIONS_INFO_KEY, [])
        for pending in mutations:
            pending()

    def discard_after_rollback(rolled_back_session: Session) -> None:
        rolled_back_session.info.pop(_MEMORY_MUTATIONS_INFO_KEY, None)

    event.listen(session, 'after_commit', apply_after_commit)
    event.listen(session, 'after_rollback', discard_after_rollback)
    session.info[_MEMORY_LISTENERS_INFO_KEY] = True
