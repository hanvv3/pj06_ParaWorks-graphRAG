import math
import re
from dataclasses import asdict, dataclass
from typing import Protocol

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


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._documents: dict[str, VectorDocument] = {}

    def upsert(self, document: VectorDocument) -> None:
        self._documents[document.document_id] = document

    def upsert_many(self, documents: list[VectorDocument]) -> None:
        for document in documents:
            self.upsert(document)

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
