import argparse
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Protocol

from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_retained_state import (
    C5SchemaCapabilityError,
    c5_schema_available,
    has_retained_c5_state,
)
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    acquire_projection,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.models.auto_review import (
    AutoReviewExtractionCall,
    AutoReviewRuntimeKeyState,
    AutoReviewValidationCall,
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


class AutoReviewKeyAdminError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class AutoReviewKeyBootstrapResult:
    schema_available: bool
    initialized: bool
    ready: bool


@dataclass(frozen=True, slots=True)
class FingerprintKeyRing:
    current_version: str
    current_secret: SecretStr
    next_version: str
    next_secret: SecretStr

    def __post_init__(self) -> None:
        if not self.current_version.strip() or not self.next_version.strip():
            raise ValueError('fingerprint key-ring versions are required')
        current = self.current_secret.get_secret_value()
        next_secret = self.next_secret.get_secret_value()
        if len(current.encode('utf-8')) < 32 or len(next_secret.encode('utf-8')) < 32:
            raise ValueError('fingerprint key-ring material must be at least 32 bytes')
        if hmac.compare_digest(current.encode('utf-8'), next_secret.encode('utf-8')):
            raise ValueError('next fingerprint key material must differ')


class FingerprintKeyRingSource(Protocol):
    def load(self) -> FingerprintKeyRing: ...


@dataclass(frozen=True, slots=True)
class ProjectionRebuildResult:
    source_count: int
    projected_count: int
    source_checksum: str
    projected_checksum: str
    replayed: bool
    ready: bool


@dataclass(frozen=True, slots=True)
class AutoReviewKeyStatus:
    runtime_generation: int | None
    runtime_version: str | None
    runtime_ready: bool
    projection_ready: bool
    projection_rebuild_required: bool
    nonterminal_extraction_count: int
    nonterminal_validation_count: int


class AutoReviewKeyAdminService:
    def __init__(
        self,
        *,
        session_factory: sessionmaker,
        settings: Settings,
        key_ring_source: FingerprintKeyRingSource | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._key_ring_source = key_ring_source

    @staticmethod
    def refuse_nonterminal_counts(
        *, extraction_count: int, validation_count: int
    ) -> None:
        if extraction_count < 0 or validation_count < 0:
            raise ValueError('non-terminal provider call counts cannot be negative')
        if extraction_count or validation_count:
            raise AutoReviewKeyAdminError(
                'nonterminal_provider_calls',
                'provider call ledger recovery is required before key rotation',
            )

    def status(self) -> AutoReviewKeyStatus:
        with self._session_factory() as db:
            try:
                schema_available = c5_schema_available(db)
            except C5SchemaCapabilityError as exc:
                raise AutoReviewKeyAdminError(
                    'incomplete_c5_schema', str(exc)
                ) from exc
            if not schema_available:
                return AutoReviewKeyStatus(
                    runtime_generation=None,
                    runtime_version=None,
                    runtime_ready=False,
                    projection_ready=False,
                    projection_rebuild_required=True,
                    nonterminal_extraction_count=0,
                    nonterminal_validation_count=0,
                )
            with KeyedMutationGuard.generation_barrier(db):
                context = lock_runtime_state(db, mode='share')
                acquire_projection(db, context)
                return _read_locked_status(db, settings=self._settings)

    def bootstrap(self) -> AutoReviewKeyBootstrapResult:
        return AutoReviewKeyBootstrapService(
            session_factory=self._session_factory,
            settings=self._settings,
        ).ensure_initialized()

    def rebuild_trusted_fingerprint_projection(self) -> ProjectionRebuildResult:
        from backend.app.knowledge.trusted_fingerprint_projection import (
            rebuild_trusted_fingerprint_projection,
        )

        return rebuild_trusted_fingerprint_projection(
            session_factory=self._session_factory,
            settings=self._settings,
        )

    def rotate_fingerprint_key(
        self,
        *,
        expected_version: str,
        next_version: str,
        reason: str,
        principal: str,
    ) -> ProjectionRebuildResult:
        from backend.app.knowledge.trusted_fingerprint_projection import (
            rotate_fingerprint_key,
        )

        if self._key_ring_source is None:
            raise AutoReviewKeyAdminError(
                'key_ring_unavailable', 'fingerprint key ring is unavailable'
            )
        return rotate_fingerprint_key(
            session_factory=self._session_factory,
            settings=self._settings,
            key_ring=self._key_ring_source.load(),
            expected_version=expected_version,
            next_version=next_version,
            reason=reason,
            principal=principal,
        )


def fingerprint_key_material_verifier(secret: str) -> str:
    return hmac.new(
        secret.encode('utf-8'),
        KEY_MATERIAL_VERIFIER_MARKER,
        hashlib.sha256,
    ).hexdigest()


def _read_locked_status(db: Session, *, settings: Settings) -> AutoReviewKeyStatus:
    from backend.app.knowledge.trusted_fingerprint_projection import (
        build_missing_active_projection_exists_statement,
        projection_identity_ready,
        source_projection_snapshots,
    )

    runtime = db.scalar(
        select(AutoReviewRuntimeKeyState).where(
            AutoReviewRuntimeKeyState.component == AUTO_REVIEW_KEY_COMPONENT
        )
    )
    projection = db.scalar(
        select(TrustedKnowledgeFingerprintProjectionState).where(
            TrustedKnowledgeFingerprintProjectionState.component
            == TRUSTED_FINGERPRINT_PROJECTION_COMPONENT
        )
    )
    extraction_count = db.scalar(
        select(func.count())
        .select_from(AutoReviewExtractionCall)
        .where(AutoReviewExtractionCall.status == 'claimed')
    ) or 0
    validation_count = db.scalar(
        select(func.count())
        .select_from(AutoReviewValidationCall)
        .where(AutoReviewValidationCall.status == 'claimed')
    ) or 0
    configured_verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    runtime_identity_ready = bool(
        runtime
        and runtime.ready
        and runtime.fingerprint_key_version
        == settings.agent_runtime_fingerprint_key_version
        and runtime.fingerprint_key_material_verifier == configured_verifier
    )
    missing_active_row = True
    if (
        runtime is not None
        and projection is not None
        and db.get_bind().dialect.name == 'postgresql'
    ):
        missing_active_row = bool(
            db.scalar(
                build_missing_active_projection_exists_statement(
                    expected_rows=source_projection_snapshots(
                        db,
                        settings=settings,
                        fingerprint_key_material_verifier=(
                            runtime.fingerprint_key_material_verifier
                        ),
                    ),
                )
            )
        )
    effective_projection_ready = bool(
        runtime_identity_ready
        and projection_identity_ready(
            runtime,
            projection,
            missing_active_row=missing_active_row,
        )
    )
    return AutoReviewKeyStatus(
        runtime_generation=runtime.generation if runtime is not None else None,
        runtime_version=(runtime.fingerprint_key_version if runtime is not None else None),
        runtime_ready=runtime_identity_ready,
        projection_ready=effective_projection_ready,
        projection_rebuild_required=bool(
            projection is None
            or projection.rebuild_required
            or not effective_projection_ready
        ),
        nonterminal_extraction_count=int(extraction_count),
        nonterminal_validation_count=int(validation_count),
    )


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
                KeyedMutationGuard.acquire_projection_lock(db)
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


class _BoundedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if message.startswith('unrecognized arguments:'):
            message = 'unrecognized arguments were provided'
        super().error(message)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = _BoundedArgumentParser(
        prog='python -m backend.app.admin.auto_review_keys'
    )
    subcommands = parser.add_subparsers(dest='command', required=True)
    subcommands.add_parser('status')
    subcommands.add_parser('bootstrap')
    subcommands.add_parser('rebuild')
    rotate = subcommands.add_parser('rotate')
    rotate.add_argument('--expected-version', required=True)
    rotate.add_argument('--next-version', required=True)
    rotate.add_argument('--reason', required=True, type=_bounded_reason)
    return parser


def _bounded_reason(value: str) -> str:
    normalized = value.strip()
    if not 1 <= len(normalized) <= 500:
        raise argparse.ArgumentTypeError('reason must contain 1 to 500 characters')
    return normalized


class _EnvironmentKeyRingSource:
    """CLI-only adapter for a deployment-injected secret-manager environment."""

    def load(self) -> FingerprintKeyRing:
        names = (
            'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_VERSION',
            'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_SECRET',
            'PARAWORKS_AUTO_REVIEW_NEXT_KEY_VERSION',
            'PARAWORKS_AUTO_REVIEW_NEXT_KEY_SECRET',
        )
        values = tuple(os.environ.get(name) for name in names)
        if any(value is None for value in values):
            raise AutoReviewKeyAdminError(
                'key_ring_unavailable',
                'deployment fingerprint key ring is unavailable',
            )
        current_version, current_secret, next_version, next_secret = values
        return FingerprintKeyRing(
            current_version=current_version or '',
            current_secret=SecretStr(current_secret or ''),
            next_version=next_version or '',
            next_secret=SecretStr(next_secret or ''),
        )


def status(
    *,
    session_factory: sessionmaker | None = None,
    settings: Settings | None = None,
) -> AutoReviewKeyStatus:
    return _default_service(
        session_factory=session_factory,
        settings=settings,
    ).status()


def bootstrap(
    *,
    session_factory: sessionmaker | None = None,
    settings: Settings | None = None,
) -> AutoReviewKeyBootstrapResult:
    return _default_service(
        session_factory=session_factory,
        settings=settings,
    ).bootstrap()


def rebuild_trusted_fingerprint_projection(
    *,
    session_factory: sessionmaker | None = None,
    settings: Settings | None = None,
) -> ProjectionRebuildResult:
    return _default_service(
        session_factory=session_factory,
        settings=settings,
    ).rebuild_trusted_fingerprint_projection()


def rotate_fingerprint_key(
    *,
    expected_version: str,
    next_version: str,
    reason: str,
    principal: str,
    session_factory: sessionmaker | None = None,
    settings: Settings | None = None,
    key_ring_source: FingerprintKeyRingSource | None = None,
) -> ProjectionRebuildResult:
    return _default_service(
        session_factory=session_factory,
        settings=settings,
        key_ring_source=key_ring_source or _EnvironmentKeyRingSource(),
    ).rotate_fingerprint_key(
        expected_version=expected_version,
        next_version=next_version,
        reason=reason,
        principal=principal,
    )


def _default_service(
    *,
    session_factory: sessionmaker | None,
    settings: Settings | None,
    key_ring_source: FingerprintKeyRingSource | None = None,
) -> AutoReviewKeyAdminService:
    if session_factory is None:
        from backend.app.db.session import SessionLocal

        session_factory = SessionLocal
    return AutoReviewKeyAdminService(
        session_factory=session_factory,
        settings=settings or Settings(),
        key_ring_source=key_ring_source,
    )


def main(argv: list[str] | None = None) -> int:
    from backend.app.db.session import SessionLocal

    args = build_cli_parser().parse_args(argv)
    settings = Settings()
    service = AutoReviewKeyAdminService(
        session_factory=SessionLocal,
        settings=settings,
        key_ring_source=_EnvironmentKeyRingSource(),
    )
    try:
        if args.command == 'status':
            result = service.status()
            payload = {
                'runtime_generation': result.runtime_generation,
                'runtime_version': result.runtime_version,
                'runtime_ready': result.runtime_ready,
                'projection_ready': result.projection_ready,
                'projection_rebuild_required': result.projection_rebuild_required,
                'nonterminal_extraction_count': result.nonterminal_extraction_count,
                'nonterminal_validation_count': result.nonterminal_validation_count,
            }
            ready = result.runtime_ready and result.projection_ready
        elif args.command == 'bootstrap':
            result = service.bootstrap()
            payload = {
                'schema_available': result.schema_available,
                'initialized': result.initialized,
                'ready': result.ready,
            }
            ready = result.ready
        elif args.command == 'rebuild':
            result = service.rebuild_trusted_fingerprint_projection()
            payload = _rebuild_payload(result)
            ready = result.ready
        else:
            result = service.rotate_fingerprint_key(
                expected_version=args.expected_version,
                next_version=args.next_version,
                reason=args.reason,
                principal='system:local-auto-review-key-admin',
            )
            payload = _rebuild_payload(result)
            ready = result.ready
    except (AutoReviewKeyAdminError, AutoReviewKeyBootstrapError, ValueError) as exc:
        code = getattr(exc, 'code', 'configuration_refused')
        print(json.dumps({'ok': False, 'code': code}, separators=(',', ':')))
        return 2
    except SQLAlchemyError:
        print(
            json.dumps(
                {'ok': False, 'code': 'storage_unavailable'},
                separators=(',', ':'),
            )
        )
        return 3
    print(json.dumps(payload, separators=(',', ':'), sort_keys=True))
    return 0 if ready else 3


def _rebuild_payload(result: ProjectionRebuildResult) -> dict[str, object]:
    return {
        'source_count': result.source_count,
        'projected_count': result.projected_count,
        'source_checksum': result.source_checksum,
        'projected_checksum': result.projected_checksum,
        'replayed': result.replayed,
        'ready': result.ready,
    }


if __name__ == '__main__':
    raise SystemExit(main())
