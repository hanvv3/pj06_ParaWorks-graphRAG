from __future__ import annotations

import math
import re
from collections.abc import Sequence
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
        current = self._transaction_documents()
        existing = [
            document_id for document_id in normalized if document_id in current
        ]
        self._apply_or_queue(_MemoryMutation('delete', tuple(existing)))
        deleted = len(existing)
        return deleted

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        target_rank = _permission_rank(permission_level)
        normalized = sorted(set(document_ids))
        current = self._transaction_documents()
        documents = [
            current[document_id]
            for document_id in normalized
            if document_id in current
        ]
        if any(_permission_rank(document.permission_level) > target_rank for document in documents):
            raise ValueError('permission broadening is not allowed')
        self._apply_or_queue(
            _MemoryMutation(
                'narrow',
                tuple(document.document_id for document in documents),
                permission_level,
            )
        )
        return len(documents)

    def _transaction_documents(self) -> dict[str, VectorDocument]:
        projected = dict(self._documents)
        if self._session is None:
            return projected
        for queued in self._session.info.get(_MEMORY_MUTATIONS_INFO_KEY, []):
            if queued.store is self:
                queued.mutation.apply(projected)
        return projected

    def _apply_or_queue(self, mutation: _MemoryMutation) -> None:
        if self._session is None:
            mutation.apply(self._documents)
            return
        _queue_after_commit(self._session, self, mutation)

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
_MEMORY_COMMITTING_INFO_KEY = 'paraworks_in_memory_vector_committing'


@dataclass(frozen=True)
class _MemoryMutation:
    kind: str
    document_ids: tuple[str, ...]
    permission_level: str | None = None

    def apply(self, documents: dict[str, VectorDocument]) -> None:
        if self.kind == 'delete':
            for document_id in self.document_ids:
                documents.pop(document_id, None)
            return
        if self.kind != 'narrow' or self.permission_level is None:
            raise RuntimeError('invalid in-memory vector mutation')
        for document_id in self.document_ids:
            document = documents.get(document_id)
            if document is None:
                continue
            documents[document_id] = VectorDocument(
                document_id=document.document_id,
                text=document.text,
                source_url=document.source_url,
                source_snippet=document.source_snippet,
                permission_level=self.permission_level,
                metadata=document.metadata,
            )


@dataclass(frozen=True)
class _QueuedMemoryMutation:
    store: InMemoryVectorStore
    transaction_identity: int
    mutation: _MemoryMutation


def _queue_after_commit(
    session: Session,
    store: InMemoryVectorStore,
    mutation: _MemoryMutation,
) -> None:
    transaction = session.get_nested_transaction() or session.get_transaction()
    if transaction is None:
        session.begin()
        transaction = session.get_transaction()
    assert transaction is not None
    session.info.setdefault(_MEMORY_MUTATIONS_INFO_KEY, []).append(
        _QueuedMemoryMutation(store, id(transaction), mutation)
    )
    if session.info.get(_MEMORY_LISTENERS_INFO_KEY):
        return

    def mark_committing(committing_session: Session) -> None:
        transaction = (
            committing_session.get_nested_transaction()
            or committing_session.get_transaction()
        )
        if transaction is not None:
            committing_session.info.setdefault(
                _MEMORY_COMMITTING_INFO_KEY, set()
            ).add(id(transaction))

    def finish_transaction(ended_session: Session, transaction) -> None:
        committing = ended_session.info.get(_MEMORY_COMMITTING_INFO_KEY, set())
        if id(transaction) not in committing:
            return
        committing.discard(id(transaction))
        if transaction.parent is not None:
            return
        mutations = ended_session.info.pop(_MEMORY_MUTATIONS_INFO_KEY, [])
        for pending in mutations:
            pending.mutation.apply(pending.store._documents)

    def discard_soft_rollback(rolled_back_session: Session, transaction) -> None:
        pending = rolled_back_session.info.get(_MEMORY_MUTATIONS_INFO_KEY, [])
        if transaction.parent is None:
            rolled_back_session.info.pop(_MEMORY_MUTATIONS_INFO_KEY, None)
            rolled_back_session.info.pop(_MEMORY_COMMITTING_INFO_KEY, None)
            return
        rolled_back_session.info[_MEMORY_MUTATIONS_INFO_KEY] = [
            queued
            for queued in pending
            if queued.transaction_identity != id(transaction)
        ]

    event.listen(session, 'before_commit', mark_committing)
    event.listen(session, 'after_transaction_end', finish_transaction)
    event.listen(session, 'after_soft_rollback', discard_soft_rollback)
    session.info[_MEMORY_LISTENERS_INFO_KEY] = True
