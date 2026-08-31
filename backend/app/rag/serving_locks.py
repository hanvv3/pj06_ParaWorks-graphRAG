from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyGenerationLockedContext,
)
from backend.app.agent_runtime.review_v2_preflight import advisory_key_from_hmac
from backend.app.core.config import Settings
from backend.app.knowledge.trusted_serving_eligibility import (
    canonical_knowledge_type,
    knowledge_model_for_type,
    knowledge_type_storage_aliases,
)
from backend.app.models import (
    AgentWorkflowThread,
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    DocumentChunk,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.serving_generation import (
    arm_corpus_generation_refresh,
    lock_rag_serving_generation,
)

_LATEST_KEY_CONTEXT_INFO_KEY = 'paraworks_c5_latest_keyed_context'
_BOUND_SHARED_KEY_CONTEXTS_INFO_KEY = (
    'paraworks_c5_bound_shared_key_contexts'
)
_BOUND_KEY_CONTEXTS_INFO_KEY = 'paraworks_c5_bound_serving_key_contexts'
_CONSUMED_KEY_CONTEXTS_INFO_KEY = 'paraworks_c5_consumed_serving_key_contexts'
_SERVING_CONTEXTS_INFO_KEY = 'paraworks_c5_vector_serving_contexts'
_TRANSACTION_LISTENER_INFO_KEY = 'paraworks_c5_serving_transaction_listener'
_DOCUMENT_LOCK_SQL = text('SELECT pg_advisory_xact_lock(:key)')


@dataclass(frozen=True, slots=True, init=False)
class VectorServingLockedContext:
    session_identity: int
    transaction_identity: int
    generation: int
    key_version: str
    material_verifier: str
    document_ids: tuple[str, ...]
    advisory_keys: tuple[int, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError('Vector-serving contexts are minted only by the lock manager')

    def __copy__(self) -> NoReturn:
        raise TypeError('Vector-serving contexts cannot be copied')

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError('Vector-serving contexts cannot be copied')

    def __reduce__(self) -> NoReturn:
        raise TypeError('Vector-serving contexts cannot be serialized')


@dataclass(frozen=True, slots=True, init=False)
class TransactionBoundServingKeyContext:
    session_identity: int
    transaction_identity: int
    generation: int
    key_version: str
    material_verifier: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            'Transaction-bound serving key contexts are minted only by '
            'the lock manager'
        )

    def __copy__(self) -> NoReturn:
        raise TypeError('Transaction-bound serving key contexts cannot be copied')

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError('Transaction-bound serving key contexts cannot be copied')

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            'Transaction-bound serving key contexts cannot be serialized'
        )


@dataclass(frozen=True, slots=True)
class ServingMutationLockPlan:
    document_ids: tuple[str, ...]
    source_ids: tuple[int, ...]
    workflow_thread_ids: tuple[str, ...]
    review_item_ids: tuple[int, ...]
    approval_link_ids: tuple[int, ...]
    targets: tuple[tuple[str, int], ...]
    security_scope_ids: tuple[str, ...]


def build_serving_lock_plan(
    db: Session,
    document_ids: Sequence[str],
    *,
    extra_review_item_ids: Sequence[int] = (),
) -> ServingMutationLockPlan:
    normalized = _normalize_document_ids(document_ids)
    source_ids: set[int] = set()
    review_item_ids = {int(value) for value in extra_review_item_ids}
    approval_link_ids: set[int] = set()
    targets: set[tuple[str, int]] = set()
    security_scope_ids: set[str] = set()
    for document_id in normalized:
        knowledge_type, raw_id = document_id.split(':', maxsplit=1)
        try:
            row_id = int(raw_id)
        except ValueError:
            continue
        if knowledge_type == 'chunk':
            chunk = db.get(DocumentChunk, row_id)
            if chunk is not None:
                source_ids.add(chunk.source_id)
            continue
        try:
            model = _knowledge_model(knowledge_type)
        except ValueError:
            continue
        target = db.get(model, row_id)
        if target is None:
            continue
        targets.add((knowledge_type, row_id))
        if target.source_review_item_id is not None:
            review_item_ids.add(target.source_review_item_id)
        links = tuple(
            db.scalars(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.knowledge_type.in_(
                        knowledge_type_storage_aliases(knowledge_type)
                    ),
                    TrustedKnowledgeApprovalLink.knowledge_id == row_id,
                    TrustedKnowledgeApprovalLink.active.is_(True),
                )
            ).all()
        )
        for link in links:
            approval_link_ids.add(link.id)
            review_item_ids.add(link.review_item_id)
            if link.security_scope_id:
                security_scope_ids.add(link.security_scope_id)
    if approval_link_ids:
        evidence_rows = tuple(
            db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id.in_(
                        approval_link_ids
                    )
                )
            ).all()
        )
        for evidence in evidence_rows:
            try:
                source_ids.add(int(evidence.canonical_source_id))
            except ValueError:
                continue
    if source_ids:
        external_source_ids = set(
            db.scalars(
                select(Source.source_id).where(Source.id.in_(source_ids))
            ).all()
        )
        raw_items = tuple(
            db.scalars(
                select(ReviewItem).where(
                    ReviewItem.status == 'approved',
                )
            ).all()
        )
        for item in raw_items:
            if item.resolution_source not in {None, 'human'}:
                continue
            payload_ids = {
                value
                for value in (item.payload or {}).get('source_ids', ())
                if isinstance(value, str)
            }
            if payload_ids & external_source_ids:
                review_item_ids.add(item.id)
    workflow_thread_ids: set[str] = set()
    if review_item_ids:
        items = tuple(
            db.scalars(
                select(ReviewItem).where(ReviewItem.id.in_(review_item_ids))
            ).all()
        )
        for item in items:
            if item.workflow_thread_id:
                workflow_thread_ids.add(item.workflow_thread_id)
    return ServingMutationLockPlan(
        document_ids=normalized,
        source_ids=tuple(sorted(source_ids)),
        workflow_thread_ids=tuple(sorted(workflow_thread_ids)),
        review_item_ids=tuple(sorted(review_item_ids)),
        approval_link_ids=tuple(sorted(approval_link_ids)),
        targets=tuple(sorted(targets)),
        security_scope_ids=tuple(sorted(security_scope_ids)),
    )


class ServingMutationLockCoordinator:
    """Acquire the frozen C.5 serving-mutation order for one exact plan."""

    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._settings = settings
        self._manager = VectorServingLockManager(db=db, settings=settings)

    def acquire(
        self,
        *,
        key_context: KeyGenerationLockedContext | None,
        plan: ServingMutationLockPlan,
    ) -> VectorServingLockedContext:
        from backend.app.agent_runtime.keyed_mutation_guard import (
            acquire_projection,
        )

        generation_context = lock_rag_serving_generation(
            self._db,
            settings=self._settings,
            key_context=key_context,
        )
        acquire_projection(self._db, key_context)
        self._lock_rows(
            AutoReviewRolloutState,
            AutoReviewRolloutState.security_scope_id,
            plan.security_scope_ids,
        )
        self._lock_rows(Source, Source.id, plan.source_ids, read=True)
        self._lock_rows(
            AgentWorkflowThread,
            AgentWorkflowThread.thread_id,
            plan.workflow_thread_ids,
        )
        self._lock_rows(
            ReviewItem, ReviewItem.id, plan.review_item_ids
        )
        self._lock_rows(
            AutoReviewPostAudit,
            AutoReviewPostAudit.review_item_id,
            plan.review_item_ids,
        )
        self._lock_rows(
            AutoReviewAuditCorrection,
            AutoReviewAuditCorrection.review_item_id,
            plan.review_item_ids,
        )
        self._lock_rows(
            TrustedKnowledgeApprovalLink,
            TrustedKnowledgeApprovalLink.id,
            plan.approval_link_ids,
        )
        self._lock_rows(
            TrustedKnowledgeEvidenceLink,
            TrustedKnowledgeEvidenceLink.approval_link_id,
            plan.approval_link_ids,
        )
        for knowledge_type, knowledge_id in plan.targets:
            model = _knowledge_model(knowledge_type)
            self._lock_rows(model, model.id, (knowledge_id,))
        current = build_serving_lock_plan(
            self._db,
            plan.document_ids,
            extra_review_item_ids=plan.review_item_ids,
        )
        if current != plan:
            raise RuntimeError('Serving mutation dependency plan changed')
        bound = self._manager.bind_transaction(key_context)
        locked = self._manager.acquire_documents(bound, plan.document_ids)
        self._lock_rows(
            VectorServingTombstone,
            VectorServingTombstone.document_id,
            plan.document_ids,
        )
        self._lock_rows(
            VectorIndexState,
            VectorIndexState.document_id,
            plan.document_ids,
        )
        arm_corpus_generation_refresh(
            self._db,
            settings=self._settings,
            context=generation_context,
        )
        return locked

    def _lock_rows(
        self,
        model: type,
        column: object,
        identities: Sequence[object],
        *,
        read: bool = False,
    ) -> None:
        if not identities:
            return
        statement = select(model).where(column.in_(tuple(identities)))
        primary_key = tuple(model.__table__.primary_key.columns)[0]
        statement = statement.order_by(primary_key).execution_options(
            populate_existing=True
        )
        if self._db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update(read=read)
        tuple(self._db.scalars(statement).all())


class VectorServingLockManager:
    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def bind_transaction(
        self,
        key_context: KeyGenerationLockedContext | None,
    ) -> TransactionBoundServingKeyContext:
        self._validate_shared_key_context(key_context)
        transaction = self._db.get_transaction()
        if transaction is None:
            raise TypeError(
                'Key-generation context transaction is no longer active'
            )
        transaction_identity = id(transaction)
        shared_bindings = self._db.info.setdefault(
            _BOUND_SHARED_KEY_CONTEXTS_INFO_KEY, {}
        )
        latest = self._db.info.get(_LATEST_KEY_CONTEXT_INFO_KEY)
        for context_id, (issued_context, _) in tuple(
            shared_bindings.items()
        ):
            if issued_context is not latest:
                shared_bindings.pop(context_id, None)
        existing = shared_bindings.get(id(key_context))
        if existing is not None and existing[0] is key_context:
            raise TypeError(
                'Key-generation context was already bound for document serving'
            )
        shared_bindings[id(key_context)] = (
            key_context,
            transaction_identity,
        )
        bound = object.__new__(TransactionBoundServingKeyContext)
        object.__setattr__(bound, 'session_identity', id(self._db))
        object.__setattr__(bound, 'transaction_identity', transaction_identity)
        object.__setattr__(bound, 'generation', key_context.generation)
        object.__setattr__(bound, 'key_version', key_context.key_version)
        object.__setattr__(
            bound, 'material_verifier', key_context.material_verifier
        )
        self._db.info.setdefault(_BOUND_KEY_CONTEXTS_INFO_KEY, {})[
            id(bound)
        ] = bound
        _ensure_transaction_cleanup_listener(self._db)
        return bound

    def acquire_documents(
        self,
        key_context: TransactionBoundServingKeyContext | None,
        document_ids: Sequence[str],
    ) -> VectorServingLockedContext:
        self._validate_bound_key_context(key_context)
        normalized = _normalize_document_ids(document_ids)
        if not normalized:
            raise ValueError('At least one serving document id is required')
        secret, _ = fingerprint_secret_bytes(self._settings)
        advisory_keys = tuple(
            advisory_key_from_hmac(
                keyed_fingerprint(
                    {'document_id': document_id},
                    secret=secret,
                    schema_version='vector-serving-document-lock:v1',
                    policy_version='vector-serving-document-lock:v1',
                )
            )
            for document_id in normalized
        )
        if self._db.get_bind().dialect.name == 'postgresql':
            for advisory_key in advisory_keys:
                self._db.execute(_DOCUMENT_LOCK_SQL, {'key': advisory_key})
        transaction = self._db.get_transaction()
        if transaction is None:
            raise TypeError(
                'Key-generation context transaction is no longer active'
            )
        self._db.info.setdefault(
            _CONSUMED_KEY_CONTEXTS_INFO_KEY, set()
        ).add(id(key_context))
        locked = object.__new__(VectorServingLockedContext)
        object.__setattr__(locked, 'session_identity', id(self._db))
        object.__setattr__(locked, 'transaction_identity', id(transaction))
        object.__setattr__(locked, 'generation', key_context.generation)
        object.__setattr__(locked, 'key_version', key_context.key_version)
        object.__setattr__(locked, 'material_verifier', key_context.material_verifier)
        object.__setattr__(locked, 'document_ids', normalized)
        object.__setattr__(locked, 'advisory_keys', advisory_keys)
        self._db.info.setdefault(_SERVING_CONTEXTS_INFO_KEY, {})[id(locked)] = locked
        return locked

    def validate_locked_context(
        self,
        context: VectorServingLockedContext,
        document_ids: Sequence[str],
    ) -> None:
        if not isinstance(context, VectorServingLockedContext):
            raise TypeError('A vector-serving lock context is required')
        if context.session_identity != id(self._db):
            raise TypeError('Vector-serving lock context belongs to another session')
        runtime = self._db.scalar(
            select(AutoReviewRuntimeKeyState).where(
                AutoReviewRuntimeKeyState.component
                == 'auto_review_trust_promotion'
            )
        )
        if runtime is None or runtime.generation != context.generation:
            raise TypeError('Vector-serving lock context generation is stale')
        if (
            runtime.fingerprint_key_version != context.key_version
            or runtime.fingerprint_key_material_verifier
            != context.material_verifier
        ):
            raise TypeError('Vector-serving lock context key identity is stale')
        issued = self._db.info.get(_SERVING_CONTEXTS_INFO_KEY, {}).get(id(context))
        if issued is not context:
            raise TypeError('Vector-serving lock context was not issued for this session')
        transaction = self._db.get_transaction()
        if transaction is None or id(transaction) != context.transaction_identity:
            raise TypeError('Vector-serving lock context transaction is no longer active')
        requested = set(_normalize_document_ids(document_ids))
        if not requested.issubset(context.document_ids):
            raise TypeError('Vector-serving lock context does not cover every document')
        self._validate_configured_key(
            context.key_version,
            context.material_verifier,
        )

    def _validate_shared_key_context(
        self, context: KeyGenerationLockedContext | None
    ) -> None:
        if not isinstance(context, KeyGenerationLockedContext):
            raise TypeError('A locked key-generation context is required')
        if context.session_identity != id(self._db):
            raise TypeError('Key-generation context belongs to another session')
        if self._db.info.get(_LATEST_KEY_CONTEXT_INFO_KEY) is not context:
            raise TypeError('Key-generation context is not active for this session')
        self._validate_configured_key(
            context.key_version,
            context.material_verifier,
        )

    def _validate_bound_key_context(
        self, context: TransactionBoundServingKeyContext | None
    ) -> None:
        if not isinstance(context, TransactionBoundServingKeyContext):
            raise TypeError(
                'A transaction-bound serving key context is required'
            )
        if context.session_identity != id(self._db):
            raise TypeError(
                'Transaction-bound serving key context belongs to another session'
            )
        transaction = self._db.get_transaction()
        if (
            transaction is None
            or id(transaction) != context.transaction_identity
        ):
            raise TypeError(
                'Key-generation context transaction is no longer active'
            )
        issued = self._db.info.get(_BOUND_KEY_CONTEXTS_INFO_KEY, {}).get(
            id(context)
        )
        if issued is not context:
            raise TypeError(
                'Transaction-bound serving key context is not active'
            )
        if id(context) in self._db.info.get(
            _CONSUMED_KEY_CONTEXTS_INFO_KEY, set()
        ):
            raise TypeError(
                'Transaction-bound serving key context was already consumed'
            )
        self._validate_configured_key(
            context.key_version,
            context.material_verifier,
        )

    def _validate_configured_key(
        self, key_version: str, material_verifier: str
    ) -> None:
        secret, configured_version = fingerprint_secret_bytes(self._settings)
        if configured_version != key_version:
            raise TypeError('Configured vector-serving key version is stale')
        configured_verifier = fingerprint_key_material_verifier(
            secret.decode('utf-8')
        )
        if configured_verifier != material_verifier:
            raise TypeError('Configured vector-serving key material is stale')


def _normalize_document_ids(document_ids: Sequence[str]) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value.strip() for value in document_ids):
        raise ValueError('Serving document ids must be non-empty strings')
    normalized: set[str] = set()
    for document_id in document_ids:
        knowledge_type, separator, raw_id = document_id.partition(':')
        normalized.add(
            f'{canonical_knowledge_type(knowledge_type)}:{raw_id}'
            if separator
            else document_id
        )
    return tuple(sorted(normalized))


def _knowledge_model(knowledge_type: str) -> type:
    return knowledge_model_for_type(knowledge_type)


def _ensure_transaction_cleanup_listener(db: Session) -> None:
    if db.info.get(_TRANSACTION_LISTENER_INFO_KEY):
        return

    def clear_transaction_contexts(
        ended_session: Session, transaction: object
    ) -> None:
        transaction_identity = id(transaction)
        bound_contexts = ended_session.info.get(
            _BOUND_KEY_CONTEXTS_INFO_KEY, {}
        )
        expired_bound_ids = {
            context_id
            for context_id, context in tuple(bound_contexts.items())
            if context.transaction_identity == transaction_identity
        }
        for context_id in expired_bound_ids:
            bound_contexts.pop(context_id, None)
        consumed = ended_session.info.get(
            _CONSUMED_KEY_CONTEXTS_INFO_KEY, set()
        )
        consumed.difference_update(expired_bound_ids)
        serving_contexts = ended_session.info.get(
            _SERVING_CONTEXTS_INFO_KEY, {}
        )
        for context_id, context in tuple(serving_contexts.items()):
            if context.transaction_identity == transaction_identity:
                serving_contexts.pop(context_id, None)

    event.listen(db, 'after_transaction_end', clear_transaction_contexts)
    db.info[_TRANSACTION_LISTENER_INFO_KEY] = True
