import hashlib
import hmac
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.app.admin.auto_review_retained_state import (
    C5SchemaCapabilityError,
    c5_schema_available,
    has_retained_c5_state,
)
from backend.app.agent_runtime.keyed_mutation_guard import KeyedMutationGuard
from backend.app.core.config import Settings
from backend.app.models.auto_review import (
    AutoReviewRuntimeKeyState,
    TrustedKnowledgeFingerprintProjectionState,
)

AUTO_REVIEW_KEY_COMPONENT = 'auto_review_trust_promotion'
TRUSTED_FINGERPRINT_PROJECTION_COMPONENT = 'trusted_knowledge_fingerprints'
TRUSTED_FINGERPRINT_PROJECTION_SCHEMA = 'trusted-fingerprint-projection:v1'
KEY_MATERIAL_VERIFIER_MARKER = b'paraworks:auto-review-key-material-verifier:v1'


class AutoReviewKeyBootstrapError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class AutoReviewKeyBootstrapResult:
    schema_available: bool
    initialized: bool
    ready: bool


def fingerprint_key_material_verifier(secret: str) -> str:
    return hmac.new(
        secret.encode('utf-8'),
        KEY_MATERIAL_VERIFIER_MARKER,
        hashlib.sha256,
    ).hexdigest()


class AutoReviewKeyBootstrapService:
    def __init__(
        self,
        *,
        session_factory: sessionmaker,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings

    def ensure_initialized(self) -> AutoReviewKeyBootstrapResult:
        with self._session_factory() as db:
            try:
                schema_available = c5_schema_available(db)
            except C5SchemaCapabilityError as exc:
                raise AutoReviewKeyBootstrapError(
                    'incomplete_c5_schema', str(exc)
                ) from exc
            if not schema_available:
                db.rollback()
                return AutoReviewKeyBootstrapResult(
                    schema_available=False,
                    initialized=False,
                    ready=False,
                )

            dialect = db.get_bind().dialect.name
            sqlite_disabled = (
                dialect == 'sqlite' and self._settings.auto_review_mode == 'disabled'
            )
            if not sqlite_disabled:
                try:
                    self._settings.require_c5_durable_key_ready()
                except ValueError as exc:
                    raise AutoReviewKeyBootstrapError(
                        'durable_key_not_ready', str(exc)
                    ) from exc

            key_version = self._settings.agent_runtime_fingerprint_key_version.strip()
            verifier = fingerprint_key_material_verifier(
                self._settings.agent_runtime_fingerprint_secret
            )
            with KeyedMutationGuard.generation_barrier(db, exclusive=True):
                runtime = KeyedMutationGuard.lock_runtime_key_state(
                    db, for_update=True
                )
                retained = has_retained_c5_state(db, ignore_runtime=True)
                if runtime is None and retained:
                    raise AutoReviewKeyBootstrapError(
                        'retained_keyed_state_without_runtime_identity',
                        'retained C.5 keyed state exists without runtime key identity',
                    )
                if runtime is not None:
                    _require_runtime_identity(
                        runtime,
                        key_version=key_version,
                        verifier=verifier,
                    )
                    projection = db.scalar(
                        select(TrustedKnowledgeFingerprintProjectionState).where(
                            TrustedKnowledgeFingerprintProjectionState.component
                            == TRUSTED_FINGERPRINT_PROJECTION_COMPONENT
                        )
                    )
                    if projection is None:
                        raise AutoReviewKeyBootstrapError(
                            'projection_state_missing',
                            'runtime key identity exists without projection state',
                        )
                    _require_projection_identity(projection, runtime=runtime)
                    db.commit()
                    return AutoReviewKeyBootstrapResult(
                        schema_available=True,
                        initialized=False,
                        ready=bool(runtime.ready),
                    )

                runtime = AutoReviewRuntimeKeyState(
                    component=AUTO_REVIEW_KEY_COMPONENT,
                    fingerprint_key_version=key_version,
                    fingerprint_key_material_verifier=verifier,
                    generation=1,
                    ready=dialect == 'postgresql',
                )
                projection = TrustedKnowledgeFingerprintProjectionState(
                    component=TRUSTED_FINGERPRINT_PROJECTION_COMPONENT,
                    projection_schema_version=(
                        TRUSTED_FINGERPRINT_PROJECTION_SCHEMA
                    ),
                    fingerprint_key_version=key_version,
                    fingerprint_key_material_verifier=verifier,
                    generation=1,
                    ready=False,
                    source_active_count=0,
                    projected_active_count=0,
                    rebuild_required=True,
                )
                db.add_all([runtime, projection])
                db.commit()
                return AutoReviewKeyBootstrapResult(
                    schema_available=True,
                    initialized=True,
                    ready=bool(runtime.ready),
                )


def _require_runtime_identity(
    runtime: AutoReviewRuntimeKeyState,
    *,
    key_version: str,
    verifier: str,
) -> None:
    if (
        runtime.fingerprint_key_version != key_version
        or runtime.fingerprint_key_material_verifier != verifier
    ):
        raise AutoReviewKeyBootstrapError(
            'runtime_key_identity_mismatch',
            'configured key identity does not match retained runtime state',
        )


def _require_projection_identity(
    projection: TrustedKnowledgeFingerprintProjectionState,
    *,
    runtime: AutoReviewRuntimeKeyState,
) -> None:
    if (
        projection.fingerprint_key_version != runtime.fingerprint_key_version
        or projection.fingerprint_key_material_verifier
        != runtime.fingerprint_key_material_verifier
        or projection.generation != runtime.generation
    ):
        raise AutoReviewKeyBootstrapError(
            'projection_key_identity_mismatch',
            'projection key identity does not match retained runtime state',
        )
