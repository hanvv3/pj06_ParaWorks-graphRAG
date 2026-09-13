from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from sqlalchemy import Connection, create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from backend.app.agent_runtime.durable_file_authority import (
    DurableFileAuthority,
    DurableFileAuthorityError,
)
from backend.app.agent_runtime.fingerprints import (
    canonical_json_bytes,
    fingerprint_secret_bytes,
)
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
    load_registered_advisory_capability,
)
from backend.app.agent_runtime.rag_safety_identity import require_lower_hmac
from backend.app.core.config import Settings, get_settings
from backend.app.rag.release_authority import (
    RagReleaseAuthority,
    RagReleaseAuthorityError,
    RagReleaseSnapshot,
    ValidationDatabaseIdentity,
)
from backend.app.rag.release_schema import (
    RAG_RELEASE_TABLE_NAMES,
    build_rag_release_metadata,
    release_tables,
)

MAX_REVIEW_ENVELOPE_BYTES = 32_768
_REVIEW_DOMAIN = b'paraworks:rag-release-admin-review:v1\x00'
_REVIEW_SCHEMA = 'rag-release-admin-review:v1'
_REVIEW_KEY_VERIFIER_DOMAIN = (
    b'paraworks:rag-release-admin-review-key-verifier:v1\x00'
)
_NONCE_DOMAIN = b'paraworks:rag-release-admin-review-nonce:v1\x00'
_OPERATIONS = frozenset(
    {
        'release-ledger-init',
        'release-ledger-rebootstrap',
        'release-ledger-disaster-init',
    }
)
_CLI_COMMANDS = (*sorted(_OPERATIONS), 'status')
_SIGNED_KEYS = frozenset(
    {
        'actor_subject_hmac',
        'expected_context',
        'implementation_plan_reference_hmac',
        'nonce',
        'operation',
        'rebootstrap_reason_hmac',
        'review_authority_key_id',
        'schema_version',
        'target',
    }
)
_TARGET_KEYS = frozenset(
    {
        'database_target_hmac',
        'designated_environment_id_hmac',
        'designated_host_id_hmac',
        'kind',
        'marker_target_hmac',
        'provider_safety_target_hmac',
    }
)

# Enabling a key requires a reviewed code change containing only its opaque
# verifier. Key bytes remain in an owner-controlled external file. Empty is the
# deliberate production fail-closed state; rotation is outside Task 23.
COMMITTED_RAG_RELEASE_REVIEW_KEYS: Mapping[str, str] = {}

__all__ = (
    'COMMITTED_RAG_RELEASE_REVIEW_KEYS',
    'MAX_REVIEW_ENVELOPE_BYTES',
    'RagLiveGateAdminService',
    'RagReleaseAdminResult',
    'RagReleaseAdminStatus',
    'RagReleaseAdminTarget',
    'RagReleaseReviewError',
    'VerifiedRagReleaseReview',
    'release_review_key_material_verifier',
    'verify_release_review_envelope',
)


class RagReleaseReviewError(RagReleaseAuthorityError):
    pass


def release_review_key_material_verifier(review_secret: bytes) -> str:
    if type(review_secret) is not bytes or len(review_secret) < 32:
        raise RagReleaseReviewError('release review authority is unavailable')
    return hmac.new(
        review_secret, _REVIEW_KEY_VERIFIER_DOMAIN, hashlib.sha256
    ).hexdigest()


def _review_hmac(value: object, *, secret: bytes, domain: bytes) -> str:
    return hmac.new(secret, domain + canonical_json_bytes(value), hashlib.sha256).hexdigest()


def _require_review_hmac(value: object, field: str) -> str:
    try:
        return require_lower_hmac(value, field)
    except ValueError as exc:
        raise RagReleaseReviewError(f'{field} is invalid') from exc


def _database_locator_identity(value: str) -> tuple[str, str | None, int | None, str]:
    from sqlalchemy.engine import make_url

    try:
        parsed = make_url(value)
    except Exception as exc:
        raise RagReleaseReviewError('release database target is invalid') from exc
    if not parsed.database:
        raise RagReleaseReviewError('release database target is invalid')
    host = parsed.host.rstrip('.').casefold() if parsed.host else None
    if host is not None:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            try:
                host = host.encode('idna').decode('ascii')
            except UnicodeError as exc:
                raise RagReleaseReviewError(
                    'release database target is ambiguous'
                ) from exc
            if host == 'localhost':
                host = 'loopback'
        else:
            host = 'loopback' if address.is_loopback else address.compressed
    port = parsed.port
    if parsed.get_backend_name() == 'postgresql' and port is None:
        port = 5432
    return parsed.get_backend_name(), host, port, parsed.database


@dataclass(frozen=True, slots=True)
class RagReleaseAdminTarget:
    database_url: str
    marker_path: Path
    provider_safety_latch_path: Path
    designated_environment_id: str
    designated_host_id: str
    review_identity: dict[str, str]

    @classmethod
    def build(
        cls,
        *,
        database_url: str,
        marker_path: str | Path,
        provider_safety_latch_path: str | Path,
        designated_environment_id: str,
        designated_host_id: str,
        review_secret: bytes,
    ) -> RagReleaseAdminTarget:
        if type(review_secret) is not bytes or len(review_secret) < 32:
            raise RagReleaseReviewError('release review authority is unavailable')
        if type(database_url) is not str or not database_url.strip():
            raise RagReleaseReviewError('release database target is invalid')
        marker = DurableFileAuthority.validate_configured_path(marker_path)
        provider = DurableFileAuthority.validate_configured_path(
            provider_safety_latch_path
        )
        for value in (designated_environment_id, designated_host_id):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or '\x00' in value
            ):
                raise RagReleaseReviewError('release target identity is invalid')

        def target_hmac(domain: bytes, value: object) -> str:
            return _review_hmac(value, secret=review_secret, domain=domain)

        identity = {
            'database_target_hmac': target_hmac(
                b'paraworks:rag-release-admin-db-target:v1\x00',
                {'database_url': database_url},
            ),
            'designated_environment_id_hmac': target_hmac(
                b'paraworks:rag-release-admin-environment:v1\x00',
                {'designated_environment_id': designated_environment_id},
            ),
            'designated_host_id_hmac': target_hmac(
                b'paraworks:rag-release-admin-host:v1\x00',
                {'designated_host_id': designated_host_id},
            ),
            'kind': 'live_validation',
            'marker_target_hmac': target_hmac(
                b'paraworks:rag-release-admin-marker-target:v1\x00',
                {'marker_path': str(marker)},
            ),
            'provider_safety_target_hmac': target_hmac(
                b'paraworks:rag-release-admin-provider-target:v1\x00',
                {'provider_safety_latch_path': str(provider)},
            ),
        }
        return cls(
            database_url=database_url,
            marker_path=marker,
            provider_safety_latch_path=provider,
            designated_environment_id=designated_environment_id,
            designated_host_id=designated_host_id,
            review_identity=identity,
        )


@dataclass(frozen=True, slots=True)
class VerifiedRagReleaseReview:
    operation: str
    actor_subject_hmac: str
    nonce: str
    nonce_hmac: str
    envelope_hmac: str
    rebootstrap_reason_hmac: str | None


def verify_release_review_envelope(
    raw: bytes,
    *,
    expected_operation: str,
    expected_target: RagReleaseAdminTarget,
    expected_context: Mapping[str, object] | None,
    review_secret: bytes,
    review_key_id: str,
    implementation_plan_reference_hmac: str,
    review_key_registry: Mapping[str, str],
) -> VerifiedRagReleaseReview:
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > MAX_REVIEW_ENVELOPE_BYTES
    ):
        raise RagReleaseReviewError('release review envelope is invalid')
    if expected_operation not in _OPERATIONS:
        raise RagReleaseReviewError('release review operation is invalid')
    _require_review_hmac(
        implementation_plan_reference_hmac, 'implementation plan reference'
    )
    verifier = release_review_key_material_verifier(review_secret)
    if review_key_registry.get(review_key_id) != verifier:
        raise RagReleaseReviewError('release review key registry refused the key')
    try:
        envelope = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RagReleaseReviewError('release review envelope is invalid') from exc
    if (
        type(envelope) is not dict
        or set(envelope) != {'hmac_sha256', 'signed_payload'}
        or canonical_json_bytes(envelope) != raw
    ):
        raise RagReleaseReviewError('release review envelope is noncanonical')
    signed = envelope['signed_payload']
    if type(signed) is not dict or set(signed) != _SIGNED_KEYS:
        raise RagReleaseReviewError('release review envelope keys are invalid')
    target = signed['target']
    if (
        type(target) is not dict
        or set(target) != _TARGET_KEYS
        or target != expected_target.review_identity
    ):
        raise RagReleaseReviewError('release review target differs')
    if (
        signed['schema_version'] != _REVIEW_SCHEMA
        or signed['operation'] != expected_operation
        or signed['review_authority_key_id'] != review_key_id
        or signed['implementation_plan_reference_hmac']
        != implementation_plan_reference_hmac
    ):
        raise RagReleaseReviewError('release review binding differs')
    if signed['expected_context'] != (
        None if expected_context is None else dict(expected_context)
    ):
        raise RagReleaseReviewError('release review context differs')
    signature = envelope['hmac_sha256']
    expected_signature = _review_hmac(
        signed, secret=review_secret, domain=_REVIEW_DOMAIN
    )
    if (
        type(signature) is not str
        or not hmac.compare_digest(signature, expected_signature)
    ):
        raise RagReleaseReviewError('release review signature is invalid')
    actor = signed['actor_subject_hmac']
    _require_review_hmac(actor, 'actor subject')
    reason = signed['rebootstrap_reason_hmac']
    if expected_operation == 'release-ledger-init':
        if reason is not None:
            raise RagReleaseReviewError('release review reason is inapplicable')
        if expected_context is not None:
            raise RagReleaseReviewError('release init context is invalid')
    else:
        if reason is None:
            raise RagReleaseReviewError('release review reason is required')
        _require_review_hmac(reason, 'release review reason')
        if expected_context is None:
            raise RagReleaseReviewError('release recovery context is required')
    nonce = signed['nonce']
    try:
        parsed_nonce = UUID(str(nonce))
    except (TypeError, ValueError) as exc:
        raise RagReleaseReviewError('release review nonce is invalid') from exc
    if parsed_nonce.int == 0 or str(parsed_nonce) != nonce:
        raise RagReleaseReviewError('release review nonce is invalid')
    return VerifiedRagReleaseReview(
        operation=expected_operation,
        actor_subject_hmac=actor,
        nonce=nonce,
        nonce_hmac=_review_hmac(nonce, secret=review_secret, domain=_NONCE_DOMAIN),
        envelope_hmac=signature,
        rebootstrap_reason_hmac=reason,
    )


@dataclass(frozen=True, slots=True)
class RagReleaseAdminStatus:
    authority_present: bool
    ledger_epoch: int | None
    generation: int | None
    state: str


@dataclass(frozen=True, slots=True)
class RagReleaseAdminResult:
    operation: str
    ledger_epoch: int
    generation: int


class RagLiveGateAdminService:
    def __init__(
        self,
        *,
        target: RagReleaseAdminTarget,
        connection_factory: Callable[[], Connection],
        authority: RagReleaseAuthority,
        review_secret: bytes,
        review_key_id: str,
        implementation_plan_reference_hmac: str,
        review_key_registry: Mapping[str, str],
        database_identity_factory: Callable[
            [Connection], ValidationDatabaseIdentity | None
        ] = lambda _connection: None,
    ) -> None:
        _require_review_hmac(
            implementation_plan_reference_hmac, 'implementation plan reference'
        )
        self.target = target
        self.connection_factory = connection_factory
        self.authority = authority
        self._review_secret = review_secret
        self._review_key_id = review_key_id
        self._plan_hmac = implementation_plan_reference_hmac
        self._registry = dict(review_key_registry)
        self._database_identity_factory = database_identity_factory

    def _verify(
        self,
        raw: bytes,
        operation: str,
        context: Mapping[str, object] | None,
    ) -> VerifiedRagReleaseReview:
        return verify_release_review_envelope(
            raw,
            expected_operation=operation,
            expected_target=self.target,
            expected_context=context,
            review_secret=self._review_secret,
            review_key_id=self._review_key_id,
            implementation_plan_reference_hmac=self._plan_hmac,
            review_key_registry=self._registry,
        )

    @staticmethod
    def _release_table_state(connection: Connection) -> set[str]:
        from sqlalchemy import inspect

        return set(inspect(connection).get_table_names()) & set(
            RAG_RELEASE_TABLE_NAMES
        )

    def _nonce_exists(self, connection: Connection, nonce_hmac: str) -> bool:
        if self._release_table_state(connection) != set(RAG_RELEASE_TABLE_NAMES):
            return False
        table = release_tables(build_rag_release_metadata()).ledgers
        return bool(
            connection.scalar(
                select(func.count()).select_from(table).where(
                    table.c.bootstrap_review_nonce_hmac == nonce_hmac
                )
            )
        )

    def initialize(self, raw: bytes) -> RagReleaseAdminResult:
        reviewed = self._verify(raw, 'release-ledger-init', None)
        with self.connection_factory() as connection:
            if self._nonce_exists(connection, reviewed.nonce_hmac):
                raise RagReleaseReviewError('release review nonce was already consumed')
            snapshot = self.authority.initialize(
                connection,
                database_identity=self._database_identity_factory(connection),
                review_envelope_hmac=reviewed.envelope_hmac,
                review_nonce_hmac=reviewed.nonce_hmac,
            )
        return self._result('release-ledger-init', snapshot)

    def _recovery_context_locked(
        self,
        connection: Connection,
        raw_marker: bytes | None,
    ) -> dict[str, object]:
        """Build signed context only while authority owns provider/release locks."""
        try:
            if raw_marker is None:
                raise RagReleaseAuthorityError('release marker is unavailable')
            _body, snapshot = self.authority._parse(raw_marker)
            state = 'marker_valid'
            marker_digest = snapshot.marker_file_digest
            marker_bytes_hmac = _review_hmac(
                hashlib.sha256(raw_marker).hexdigest(),
                secret=self._review_secret,
                domain=b'paraworks:rag-release-admin-marker-context:v1\x00',
            )
            ledger_epoch: int | None = snapshot.ledger_epoch
            generation: int | None = snapshot.generation
        except RagReleaseAuthorityError:
            state = 'marker_missing_or_corrupt'
            marker_digest = None
            if raw_marker is not None:
                marker_bytes_hmac = _review_hmac(
                    hashlib.sha256(raw_marker).hexdigest(),
                    secret=self._review_secret,
                    domain=b'paraworks:rag-release-admin-marker-context:v1\x00',
                )
            else:
                marker_bytes_hmac = None
            ledger_epoch = None
            generation = None
        table_state = self._release_table_state(connection)
        row_count = 0
        ledger_set_hmac: str | None = None
        if table_state == set(RAG_RELEASE_TABLE_NAMES):
            table = release_tables(build_rag_release_metadata()).ledgers
            row_count = int(connection.scalar(select(func.count()).select_from(table)))
            rows = connection.execute(
                select(
                    table.c.ledger_uuid,
                    table.c.ledger_epoch,
                    table.c.generation,
                    table.c.last_transition_digest,
                    table.c.marker_file_digest,
                    table.c.validation_database_identity_hmac,
                    table.c.validation_database_locator_hmac,
                ).order_by(table.c.ledger_uuid, table.c.ledger_epoch)
            ).all()
            ledger_set_hmac = _review_hmac(
                [list(row) for row in rows],
                secret=self._review_secret,
                domain=b'paraworks:rag-release-admin-db-context:v1\x00',
            )
        context = {
            'generation': generation,
            'ledger_set_hmac': ledger_set_hmac,
            'ledger_epoch': ledger_epoch,
            'marker_bytes_hmac': marker_bytes_hmac,
            'marker_file_digest': marker_digest,
            'marker_state': state,
            'release_ledger_row_count': row_count,
        }
        return context

    def rebootstrap(self, raw: bytes) -> RagReleaseAdminResult:
        with self.connection_factory() as connection:
            def verify_locked(
                locked_connection: Connection,
                raw_marker: bytes | None,
            ) -> tuple[str, str, str]:
                context = self._recovery_context_locked(
                    locked_connection, raw_marker
                )
                reviewed = self._verify(
                    raw, 'release-ledger-rebootstrap', context
                )
                return (
                    reviewed.envelope_hmac,
                    reviewed.nonce_hmac,
                    reviewed.rebootstrap_reason_hmac or '',
                )

            snapshot = self.authority.rebootstrap(
                connection,
                database_identity=self._database_identity_factory(connection),
                review_verifier=verify_locked,
            )
        return self._result('release-ledger-rebootstrap', snapshot)

    def disaster_initialize(self, raw: bytes) -> RagReleaseAdminResult:
        with self.connection_factory() as connection:
            def verify_locked(
                locked_connection: Connection,
                raw_marker: bytes | None,
            ) -> tuple[str, str, str]:
                context = self._recovery_context_locked(
                    locked_connection, raw_marker
                )
                reviewed = self._verify(
                    raw, 'release-ledger-disaster-init', context
                )
                return (
                    reviewed.envelope_hmac,
                    reviewed.nonce_hmac,
                    reviewed.rebootstrap_reason_hmac or '',
                )

            snapshot = self.authority.disaster_initialize(
                connection,
                database_identity=self._database_identity_factory(connection),
                review_verifier=verify_locked,
            )
        return self._result('release-ledger-disaster-init', snapshot)

    def status(self) -> RagReleaseAdminStatus:
        with self.connection_factory() as connection:
            table_state = self._release_table_state(connection)
            if not self.target.marker_path.exists() and not table_state:
                return RagReleaseAdminStatus(False, None, None, 'absent')
            try:
                snapshot = self.authority.inspect(
                    connection,
                    database_identity=self._database_identity_factory(connection),
                )
            except RagReleaseAuthorityError:
                return RagReleaseAdminStatus(True, None, None, 'inconsistent')
            return RagReleaseAdminStatus(
                True, snapshot.ledger_epoch, snapshot.generation, 'ready'
            )

    @staticmethod
    def _result(operation: str, snapshot: RagReleaseSnapshot) -> RagReleaseAdminResult:
        return RagReleaseAdminResult(
            operation=operation,
            ledger_epoch=snapshot.ledger_epoch,
            generation=snapshot.generation,
        )


@dataclass(frozen=True, slots=True)
class CliOutcome:
    exit_code: int
    payload: dict[str, object]


def _read_stdin(stdin: BinaryIO) -> bytes:
    raw = stdin.read(MAX_REVIEW_ENVELOPE_BYTES + 1)
    if len(raw) > MAX_REVIEW_ENVELOPE_BYTES:
        raise RagReleaseReviewError('release review envelope is too large')
    return raw


def _run_cli(
    argv: list[str],
    *,
    stdin: BinaryIO,
    service_factory: Callable[[], RagLiveGateAdminService],
) -> CliOutcome:
    if len(argv) != 1 or argv[0] not in _CLI_COMMANDS:
        return CliOutcome(2, {'code': 'command_refused', 'ok': False})
    try:
        service = service_factory()
        command = argv[0]
        if command == 'status':
            status = service.status()
            return CliOutcome(
                0,
                {
                    'authority_present': status.authority_present,
                    'generation': status.generation,
                    'ledger_epoch': status.ledger_epoch,
                    'ok': True,
                    'state': status.state,
                },
            )
        raw = _read_stdin(stdin)
        if command == 'release-ledger-init':
            result = service.initialize(raw)
        elif command == 'release-ledger-rebootstrap':
            result = service.rebootstrap(raw)
        else:
            result = service.disaster_initialize(raw)
        return CliOutcome(
            0,
            {
                'generation': result.generation,
                'ledger_epoch': result.ledger_epoch,
                'ok': True,
                'operation': result.operation,
            },
        )
    except RagReleaseReviewError:
        return CliOutcome(2, {'code': 'review_refused', 'ok': False})
    except RagReleaseAuthorityError:
        return CliOutcome(3, {'code': 'authority_refused', 'ok': False})


def _load_review_secret(path_value: str | None) -> bytes:
    if not path_value:
        raise RagReleaseReviewError('release review key file is unavailable')
    path = Path(path_value)
    if not path.is_absolute():
        raise RagReleaseReviewError('release review key path is invalid')
    try:
        DurableFileAuthority.open_runtime(path)._validate_existing_regular(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_nlink != 1:
            raise RagReleaseReviewError('release review key file is untrusted')
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RagReleaseReviewError('release review key file changed')
            value = os.read(descriptor, 4097)
        finally:
            os.close(descriptor)
    except RagReleaseReviewError:
        raise
    except OSError as exc:
        raise RagReleaseReviewError('release review key file is unavailable') from exc
    if not 32 <= len(value) <= 4096:
        raise RagReleaseReviewError('release review key file is invalid')
    return value


@dataclass(slots=True)
class _DefaultResources:
    service: RagLiveGateAdminService
    engine: object
    provider_resources: object

    def close(self) -> None:
        self.engine.dispose()  # type: ignore[attr-defined]
        self.provider_resources.close()  # type: ignore[attr-defined]


def _build_default_resources(settings: Settings) -> _DefaultResources:
    from sqlalchemy.engine import make_url

    from backend.app.admin.rag_provider_safety import (
        _build_default_admin_resources,
    )

    database_url = settings.paraworks_rag_live_validation_database_url
    marker_path = settings.paraworks_release_ledger_authority_path
    provider_path = settings.paraworks_rag_live_validation_provider_safety_latch_path
    host_id = settings.paraworks_rag_live_validation_host_id
    plan_hmac = settings.paraworks_release_implementation_plan_reference_hmac
    backup_path = settings.paraworks_rag_live_validation_database_backup_path
    if not all(
        (
            database_url,
            marker_path,
            provider_path,
            host_id,
            plan_hmac,
            backup_path,
        )
    ):
        raise RagReleaseReviewError('release authority configuration is incomplete')
    _require_review_hmac(plan_hmac, 'implementation plan reference')
    if make_url(database_url).get_backend_name() != 'postgresql':
        raise RagReleaseReviewError('release authority requires PostgreSQL')
    if _database_locator_identity(
        settings.resolved_database_url()
    ) == _database_locator_identity(database_url):
        raise RagReleaseReviewError('production and validation databases must differ')
    review_secret = _load_review_secret(settings.paraworks_release_review_key_path)
    target = RagReleaseAdminTarget.build(
        database_url=database_url,
        marker_path=marker_path,
        provider_safety_latch_path=provider_path,
        designated_environment_id=(
            settings.paraworks_rag_live_validation_environment_id
        ),
        designated_host_id=host_id,
        review_secret=review_secret,
    )
    engine = create_engine(database_url, pool_pre_ping=True)
    provider_resources = None
    try:
        provider_resources = _build_default_admin_resources(settings)
        provider_admin = provider_resources.service
        if (
            provider_admin.target.kind != 'live_validation'
            or _database_locator_identity(provider_admin.target.database_url)
            != _database_locator_identity(database_url)
            or provider_admin.target.latch_path != target.provider_safety_latch_path
            or provider_admin.target.designated_environment_id
            != settings.paraworks_rag_live_validation_environment_id
        ):
            raise RagReleaseReviewError(
                'release provider safety peer target differs'
            )
        with engine.connect() as connection:
            advisory = load_registered_advisory_capability(
                connection,
                RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
                identity_namespace='static',
            )
        runtime_secret, key_version = fingerprint_secret_bytes(settings)
        authority = RagReleaseAuthority(
            marker_path=marker_path,
            provider_safety_latch_path=provider_path,
            identity_secret=runtime_secret,
            fingerprint_key_version=key_version,
            designated_environment_id=(
                settings.paraworks_rag_live_validation_environment_id
            ),
            designated_host_id=host_id,
            repository_roots=(Path.cwd(),),
            database_backup_roots=(backup_path,),
            advisory_capability=advisory,
            provider_safety_release_peer=provider_admin.release_peer(),
        )
        return _DefaultResources(
            service=RagLiveGateAdminService(
                target=target,
                connection_factory=engine.connect,
                authority=authority,
                review_secret=review_secret,
                review_key_id=settings.paraworks_release_review_key_id,
                implementation_plan_reference_hmac=plan_hmac,
                review_key_registry=COMMITTED_RAG_RELEASE_REVIEW_KEYS,
            ),
            engine=engine,
            provider_resources=provider_resources,
        )
    except Exception:
        engine.dispose()
        if provider_resources is not None:
            provider_resources.close()
        raise


def main(argv: list[str] | None = None) -> int:
    resources: _DefaultResources | None = None
    try:
        resources = _build_default_resources(get_settings())
        outcome = _run_cli(
            list(sys.argv[1:] if argv is None else argv),
            stdin=sys.stdin.buffer,
            service_factory=lambda: resources.service,  # type: ignore[union-attr]
        )
    except (
        DurableFileAuthorityError,
        RagReleaseReviewError,
        RagReleaseAuthorityError,
        SQLAlchemyError,
        ValueError,
    ):
        outcome = CliOutcome(2, {'code': 'configuration_refused', 'ok': False})
    finally:
        if resources is not None:
            resources.close()
    sys.stdout.buffer.write(canonical_json_bytes(outcome.payload) + b'\n')
    return outcome.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
