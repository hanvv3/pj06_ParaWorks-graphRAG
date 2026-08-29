from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

from sqlalchemy import select, text
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
from backend.app.models import AutoReviewRuntimeKeyState

_LATEST_KEY_CONTEXT_INFO_KEY = 'paraworks_c5_latest_keyed_context'
_SERVING_CONTEXTS_INFO_KEY = 'paraworks_c5_vector_serving_contexts'
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


class VectorServingLockManager:
    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def acquire_documents(
        self,
        key_context: KeyGenerationLockedContext | None,
        document_ids: Sequence[str],
    ) -> VectorServingLockedContext:
        self._validate_key_context(key_context)
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
            raise RuntimeError('Vector-serving locks require an active transaction')
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

    def _validate_key_context(
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
    return tuple(sorted(set(document_ids)))
