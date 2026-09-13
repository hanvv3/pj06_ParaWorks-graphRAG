from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import Connection, func, insert, inspect, select

from backend.app.agent_runtime.durable_file_authority import (
    DurableFileAuthority,
    DurableFileAuthorityError,
)
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
    RegisteredAdvisoryLock,
    acquire_advisory_lock,
    begin_rag_lock_order,
    release_advisory_lock,
)
from backend.app.agent_runtime.rag_safety_identity import (
    rag_identity_hmac,
    require_lower_hmac,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.rag.release_schema import (
    RAG_RELEASE_TABLE_NAMES,
    assert_rag_release_physical_contract,
    build_rag_release_metadata,
    release_tables,
)

_MARKER_SCHEMA = 'rag-release-ledger-marker-body:v1'
_MARKER_ENVELOPE_SCHEMA = 'rag-release-ledger-marker-envelope:v1'
_POLICY_VERSION = 'rag-live-gate:v1'
_BODY_KEYS = frozenset(
    {
        'designated_environment_id_hmac',
        'designated_host_id_hmac',
        'generation',
        'last_transition_digest',
        'ledger_epoch',
        'ledger_uuid',
        'marker_schema_version',
        'predecessor_marker_digest',
        'rebootstrap_reason_hmac',
        'bootstrap_review_envelope_hmac',
        'bootstrap_review_nonce_hmac',
        'bootstrap_operation',
        'validation_database_identity_hmac',
        'validation_database_locator_hmac',
    }
)
_SIGNED_KEYS = frozenset(
    {
        'body',
        'fingerprint_key_material_verifier',
        'fingerprint_key_version',
    }
)


class RagReleaseAuthorityError(RuntimeError):
    pass


def _release_barrier_boundary():
    from threading import get_ident
    from weakref import WeakKeyDictionary

    active = WeakKeyDictionary()
    provider_checkpoints = WeakKeyDictionary()
    held = set()
    held_threads = set()

    def state(guard):
        if type(guard) is not Guard or guard not in active:
            raise RagReleaseAuthorityError('release barrier guard is invalid')
        owner, connection, provider, thread, transaction = active[guard]
        if (
            connection.closed
            or thread != get_ident()
            or (
                transaction is not None
                and connection.get_transaction() is not transaction
            )
        ):
            raise RagReleaseAuthorityError('release barrier guard is invalid')
        if transaction is None and connection.get_transaction() is not None:
            active[guard] = (
                owner,
                connection,
                provider,
                thread,
                connection.get_transaction(),
            )
        from backend.app.admin.rag_provider_safety import _require_provider_guard

        _require_provider_guard(
            owner._provider_safety_release_peer, provider, connection
        )
        return owner, connection, provider, thread

    class Guard:
        __slots__ = ('__weakref__',)

        def __init__(self, *args, **kwargs):
            raise TypeError('release guards require an active authority context')

        @property
        def connection(self):
            return state(self)[1]

        def apply_provider_incident(self, plan):
            from backend.app.admin.rag_provider_safety import (
                RagProviderSafetyReleasePeerGuard,
                _freeze_release_peer,
            )

            _owner, connection, provider, _thread = state(self)
            evidence = RagProviderSafetyReleasePeerGuard.apply_incident(
                provider, connection, plan
            )
            provider_checkpoints[self] = _freeze_release_peer(
                _owner._provider_safety_release_peer, provider, connection
            )
            return evidence

        def freeze_provider(self):
            from backend.app.admin.rag_provider_safety import _freeze_release_peer

            owner, connection, provider, _thread = state(self)
            if self in provider_checkpoints:
                raise RagReleaseAuthorityError('provider checkpoint is already frozen')
            provider_checkpoints[self] = _freeze_release_peer(
                owner._provider_safety_release_peer, provider, connection
            )

        def revalidate_provider(self):
            _owner, connection, provider, _thread = state(self)
            if self in provider_checkpoints:
                provider_checkpoints[self]()
            else:
                provider.revalidate_database_peer(connection)

        def __copy__(self):
            raise TypeError('release guards cannot be copied')

        def __deepcopy__(self, memo):
            raise TypeError('release guards cannot be copied')

        def __reduce_ex__(self, protocol):
            raise TypeError('release guards cannot be serialized')

    @contextmanager
    def scope(self, connection, *, marker):
        # Refuse recursion before acquiring Windows non-reentrant sidecars.
        thread = get_ident()
        if connection in held or thread in held_threads:
            raise RagReleaseAuthorityError('release barrier is already active')
        held.add(connection)
        held_threads.add(thread)
        try:
            with self._authority_transport(connection, marker=marker) as provider:
                guard = object.__new__(Guard)
                active[guard] = (
                    self,
                    connection,
                    provider,
                    get_ident(),
                    connection.get_transaction(),
                )
                try:
                    self._assert_barrier_guard(guard, connection)
                    yield guard
                finally:
                    # Expiration precedes transport teardown, even on exceptions.
                    del active[guard]
                    provider_checkpoints.pop(guard, None)
        finally:
            held.remove(connection)
            held_threads.remove(thread)

    def require(self, guard, connection):
        owner, actual, _provider, _thread = state(guard)
        if owner is not self or actual is not connection:
            raise RagReleaseAuthorityError('release barrier guard is invalid')
        return guard

    return Guard, scope, require


_RagReleaseBarrierGuard, _release_scope, _require_release_guard = (
    _release_barrier_boundary()
)


@dataclass(frozen=True, slots=True)
class ValidationDatabaseIdentity:
    database_name: str
    database_oid: int

    def __post_init__(self) -> None:
        if (
            type(self.database_name) is not str
            or not self.database_name
            or self.database_name != self.database_name.strip()
            or '\x00' in self.database_name
        ):
            raise ValueError('validation database identity is invalid')
        if type(self.database_oid) is not int or self.database_oid <= 0:
            raise ValueError('validation database identity is invalid')


def _native_database_identity(supplied):
    """Detach public identity inputs before any callback or publication."""
    if supplied is None:
        return None
    if type(supplied) is not ValidationDatabaseIdentity:
        raise RagReleaseAuthorityError('validation database identity is invalid')
    try:
        return ValidationDatabaseIdentity(
            database_name=object.__getattribute__(supplied, 'database_name'),
            database_oid=object.__getattribute__(supplied, 'database_oid'),
        )
    except (AttributeError, ValueError):
        raise RagReleaseAuthorityError(
            'validation database identity is invalid'
        ) from None


@dataclass(frozen=True, slots=True)
class RagReleaseSnapshot:
    marker_schema_version: Literal['rag-release-ledger-marker-body:v1']
    ledger_uuid: UUID
    ledger_epoch: int
    generation: int
    last_transition_digest: str | None
    predecessor_marker_digest: str | None
    rebootstrap_reason_hmac: str | None
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    marker_envelope_hmac: str
    marker_file_digest: str
    designated_environment_id_hmac: str
    designated_host_id_hmac: str
    validation_database_identity_hmac: str
    validation_database_locator_hmac: str
    bootstrap_review_envelope_hmac: str
    bootstrap_review_nonce_hmac: str
    bootstrap_operation: str


RecoveryReviewVerifier = Callable[[Connection, bytes | None], tuple[str, str, str]]


class ExternalAuthorityPathSetValidator:
    """Validate the provider/release data+stable-lock leaves as one set."""

    @staticmethod
    def freeze(path):
        try:
            text = os.fspath(path)
        except TypeError:
            raise RagReleaseAuthorityError('authority path is invalid') from None
        if type(text) is not str:
            raise RagReleaseAuthorityError('authority path is invalid')
        return str(DurableFileAuthority.validate_configured_path(text))

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        candidate_text = os.path.normcase(str(candidate))
        root_text = os.path.normcase(str(root))
        try:
            return os.path.commonpath((candidate_text, root_text)) == root_text
        except ValueError:
            return False

    @classmethod
    def validate(
        cls,
        *,
        provider_path: str | Path,
        release_path: str | Path,
        repository_roots: Iterable[str | Path],
        database_backup_roots: Iterable[str | Path],
    ) -> tuple[Path, Path, Path, Path]:
        provider = DurableFileAuthority.validate_configured_path(provider_path)
        release = DurableFileAuthority.validate_configured_path(release_path)
        leaves = (
            provider,
            Path(str(provider) + '.lock'),
            release,
            Path(str(release) + '.lock'),
        )
        canonical = tuple(
            DurableFileAuthority.validate_configured_path(path) for path in leaves
        )
        normalized = tuple(os.path.normcase(str(path)) for path in canonical)
        for index, left in enumerate(normalized):
            for right in normalized[index + 1 :]:
                try:
                    common = os.path.commonpath((left, right))
                except ValueError:
                    continue
                if left == right or common in {left, right}:
                    raise RagReleaseAuthorityError(
                        'provider and release authority leaves must be distinct'
                    )
        existing = [path for path in canonical if path.exists()]
        for index, left in enumerate(existing):
            for right in existing[index + 1 :]:
                try:
                    same = os.path.samefile(left, right)
                except OSError as exc:
                    raise RagReleaseAuthorityError(
                        'authority leaf identity is unavailable'
                    ) from exc
                if same:
                    raise RagReleaseAuthorityError(
                        'provider and release authority leaves must be distinct'
                    )
        excluded = tuple(
            DurableFileAuthority.validate_configured_path(root)
            for root in (*tuple(repository_roots), *tuple(database_backup_roots))
        )
        if any(cls._is_within(release, root) for root in excluded):
            raise RagReleaseAuthorityError(
                'release authority must remain outside repository and database backup roots'
            )
        return canonical


class RagReleaseAuthority:
    """External-first, validation-database-only release ledger authority."""

    def __init__(
        self,
        *,
        marker_path: str | Path,
        provider_safety_latch_path: str | Path,
        identity_secret: bytes,
        fingerprint_key_version: str,
        designated_environment_id: str,
        designated_host_id: str,
        repository_roots: Iterable[str | Path] = (),
        database_backup_roots: Iterable[str | Path] = (),
        advisory_capability: RegisteredAdvisoryLock | None = None,
        provider_safety_release_peer: object,
        after_marker_replace: Callable[[], None] | None = None,
    ) -> None:
        if type(identity_secret) is not bytes or len(identity_secret) < 32:
            raise ValueError('release fingerprint key is unavailable')
        try:
            identity_secret.decode('utf-8')
        except UnicodeError:
            raise ValueError('release fingerprint key is invalid') from None
        for label, value in (
            ('fingerprint key version', fingerprint_key_version),
            ('designated environment', designated_environment_id),
            ('designated host', designated_host_id),
        ):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or '\x00' in value
            ):
                raise ValueError(f'release {label} is invalid')
        # Consume each iterable once and every caller path protocol once. Later
        # filesystem checks retain canonical native strings, never caller objects.
        roots = tuple(repository_roots), tuple(database_backup_roots)
        freeze = ExternalAuthorityPathSetValidator.freeze
        self._repository_roots = tuple(freeze(root) for root in roots[0])
        self._database_backup_roots = tuple(freeze(root) for root in roots[1])
        paths = ExternalAuthorityPathSetValidator.validate(
            provider_path=freeze(provider_safety_latch_path),
            release_path=freeze(marker_path),
            repository_roots=self._repository_roots,
            database_backup_roots=self._database_backup_roots,
        )
        self._provider_path, _provider_lock, self._marker_path, _release_lock = paths
        self._secret = identity_secret
        self._key_version = fingerprint_key_version
        self._environment_hmac = self._identity_hmac(
            'rag-live-designated-environment-id:v1',
            {
                'designated_environment_id_bytes': exact_utf8_bytes(
                    designated_environment_id
                )
            },
        )
        self._host_hmac = self._identity_hmac(
            'rag-live-designated-host-id:v1',
            {'designated_host_id_bytes': exact_utf8_bytes(designated_host_id)},
        )
        self._advisory_capability = advisory_capability
        from backend.app.admin.rag_provider_safety import RagProviderSafetyReleasePeer

        if type(provider_safety_release_peer) is not RagProviderSafetyReleasePeer:
            raise ValueError('pinned provider safety release peer is required')
        self._provider_safety_release_peer = provider_safety_release_peer
        self._after_marker_replace = after_marker_replace
        if advisory_capability is not None and not advisory_capability.matches(
            RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        ):
            raise ValueError('release advisory capability identity is invalid')

    def _identity_hmac(self, schema_version: str, value: object) -> str:
        return rag_identity_hmac(
            value,
            secret=self._secret,
            schema_version=schema_version,
            policy_version=_POLICY_VERSION,
        )

    @property
    def marker_path(self) -> Path:
        return self._marker_path

    def _key_verifier(self) -> str:
        from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier

        return fingerprint_key_material_verifier(self._secret.decode('utf-8'))

    def _signature(self, signed_payload: object) -> str:
        return self._identity_hmac(_MARKER_ENVELOPE_SCHEMA, signed_payload)

    @staticmethod
    def _file_digest_bytes(raw: bytes) -> str:
        return hashlib.sha256(
            b'paraworks:release-ledger-marker-file:v1\x00' + raw
        ).hexdigest()

    def _database_hmac(
        self,
        identity: ValidationDatabaseIdentity,
        identity_uuid: UUID,
    ) -> str:
        return self._identity_hmac(
            'rag-live-validation-database-identity:v1',
            {
                'database_name_bytes': identity.database_name,
                'database_oid': identity.database_oid,
                'database_system': 'postgresql',
                'validation_database_identity_uuid': str(identity_uuid),
            },
        )

    def _database_locator_hmac(self, identity: ValidationDatabaseIdentity) -> str:
        return self._identity_hmac(
            'rag-live-validation-database-locator:v1',
            {
                'database_name_bytes': identity.database_name,
                'database_oid': identity.database_oid,
                'database_system': 'postgresql',
            },
        )

    @staticmethod
    def _current_database_identity(
        connection: Connection,
        supplied: ValidationDatabaseIdentity | None,
    ) -> ValidationDatabaseIdentity:
        supplied = _native_database_identity(supplied)
        if connection.dialect.name != 'postgresql':
            raise RagReleaseAuthorityError('release authority requires PostgreSQL')
        row = connection.exec_driver_sql(
            'SELECT current_database(), oid FROM pg_database '
            'WHERE datname = current_database()'
        ).one()
        current = ValidationDatabaseIdentity(database_name=row[0], database_oid=row[1])
        if supplied is not None and supplied != current:
            raise RagReleaseAuthorityError('supplied database identity differs')
        return current

    def _body(
        self,
        *,
        ledger_uuid: UUID,
        ledger_epoch: int,
        generation: int,
        last_transition_digest: str | None,
        predecessor_marker_digest: str | None,
        rebootstrap_reason_hmac: str | None,
        database_identity_hmac: str,
        database_locator_hmac: str,
        review_envelope_hmac: str,
        review_nonce_hmac: str,
        review_operation: str,
    ) -> dict[str, object]:
        return {
            'designated_environment_id_hmac': self._environment_hmac,
            'designated_host_id_hmac': self._host_hmac,
            'generation': generation,
            'last_transition_digest': last_transition_digest,
            'ledger_epoch': ledger_epoch,
            'ledger_uuid': str(ledger_uuid),
            'marker_schema_version': _MARKER_SCHEMA,
            'predecessor_marker_digest': predecessor_marker_digest,
            'rebootstrap_reason_hmac': rebootstrap_reason_hmac,
            'bootstrap_review_envelope_hmac': review_envelope_hmac,
            'bootstrap_review_nonce_hmac': review_nonce_hmac,
            'bootstrap_operation': review_operation,
            'validation_database_identity_hmac': database_identity_hmac,
            'validation_database_locator_hmac': database_locator_hmac,
        }

    def _wrap(self, body: dict[str, object]) -> dict[str, object]:
        signed_payload = {
            'body': body,
            'fingerprint_key_material_verifier': self._key_verifier(),
            'fingerprint_key_version': self._key_version,
        }
        return {
            'hmac_sha256': self._signature(signed_payload),
            'signed_payload': signed_payload,
        }

    def _parse(self, raw: bytes) -> tuple[dict[str, object], RagReleaseSnapshot]:
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RagReleaseAuthorityError('release marker is invalid') from exc
        if type(value) is not dict or set(value) != {'hmac_sha256', 'signed_payload'}:
            raise RagReleaseAuthorityError('release marker is invalid')
        if canonical_json_bytes(value) != raw:
            raise RagReleaseAuthorityError('release marker is noncanonical')
        signed = value['signed_payload']
        if type(signed) is not dict or set(signed) != _SIGNED_KEYS:
            raise RagReleaseAuthorityError('release marker is invalid')
        body = signed['body']
        if type(body) is not dict or set(body) != _BODY_KEYS:
            raise RagReleaseAuthorityError('release marker is invalid')
        signature = value['hmac_sha256']
        if (
            type(signature) is not str
            or not hmac.compare_digest(signature, self._signature(signed))
            or signed['fingerprint_key_version'] != self._key_version
            or signed['fingerprint_key_material_verifier'] != self._key_verifier()
        ):
            raise RagReleaseAuthorityError('release marker verification failed')
        try:
            ledger_uuid = UUID(str(body['ledger_uuid']))
            epoch = body['ledger_epoch']
            generation = body['generation']
            if str(ledger_uuid) != body['ledger_uuid']:
                raise ValueError
            if type(epoch) is not int or epoch <= 0:
                raise ValueError
            if type(generation) is not int or generation < 0:
                raise ValueError
            if body['marker_schema_version'] != _MARKER_SCHEMA:
                raise ValueError
            for key in (
                'designated_environment_id_hmac',
                'designated_host_id_hmac',
                'validation_database_identity_hmac',
                'validation_database_locator_hmac',
                'bootstrap_review_envelope_hmac',
                'bootstrap_review_nonce_hmac',
            ):
                require_lower_hmac(body[key])
            if body['bootstrap_operation'] not in {
                'release-ledger-init',
                'release-ledger-rebootstrap',
                'release-ledger-disaster-init',
            }:
                raise ValueError
            if body['bootstrap_operation'] == 'release-ledger-init':
                if (
                    epoch != 1
                    or body['predecessor_marker_digest'] is not None
                    or body['rebootstrap_reason_hmac'] is not None
                ):
                    raise ValueError
            elif body['bootstrap_operation'] == 'release-ledger-disaster-init':
                if (
                    epoch != 1
                    or body['predecessor_marker_digest'] is not None
                    or body['rebootstrap_reason_hmac'] is None
                ):
                    raise ValueError
            elif (
                epoch <= 1
                or body['predecessor_marker_digest'] is None
                or body['rebootstrap_reason_hmac'] is None
            ):
                raise ValueError
            for key in (
                'last_transition_digest',
                'predecessor_marker_digest',
                'rebootstrap_reason_hmac',
            ):
                if body[key] is not None:
                    require_lower_hmac(body[key])
            if (generation == 0) != (body['last_transition_digest'] is None):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise RagReleaseAuthorityError('release marker is invalid') from exc
        raw_digest = self._file_digest_bytes(raw)
        return body, RagReleaseSnapshot(
            marker_schema_version=_MARKER_SCHEMA,
            ledger_uuid=ledger_uuid,
            ledger_epoch=epoch,
            generation=generation,
            last_transition_digest=body['last_transition_digest'],  # type: ignore[arg-type]
            predecessor_marker_digest=body['predecessor_marker_digest'],  # type: ignore[arg-type]
            rebootstrap_reason_hmac=body['rebootstrap_reason_hmac'],  # type: ignore[arg-type]
            fingerprint_key_version=self._key_version,
            fingerprint_key_material_verifier=self._key_verifier(),
            marker_envelope_hmac=signature,
            marker_file_digest=raw_digest,
            designated_environment_id_hmac=body['designated_environment_id_hmac'],  # type: ignore[arg-type]
            designated_host_id_hmac=body['designated_host_id_hmac'],  # type: ignore[arg-type]
            validation_database_identity_hmac=body['validation_database_identity_hmac'],  # type: ignore[arg-type]
            validation_database_locator_hmac=body['validation_database_locator_hmac'],  # type: ignore[arg-type]
            bootstrap_review_envelope_hmac=body['bootstrap_review_envelope_hmac'],  # type: ignore[arg-type]
            bootstrap_review_nonce_hmac=body['bootstrap_review_nonce_hmac'],  # type: ignore[arg-type]
            bootstrap_operation=body['bootstrap_operation'],  # type: ignore[arg-type]
        )

    def _validate_path_set(self) -> None:
        ExternalAuthorityPathSetValidator.validate(
            provider_path=self._provider_path,
            release_path=self._marker_path,
            repository_roots=self._repository_roots,
            database_backup_roots=self._database_backup_roots,
        )

    @contextmanager
    def _registered_advisory(self, connection: Connection) -> Iterator[None]:
        if connection.dialect.name != 'postgresql':
            raise RagReleaseAuthorityError('release authority requires PostgreSQL')
        if self._advisory_capability is None:
            raise RagReleaseAuthorityError(
                'registered release advisory capability is required'
            )
        acquire_advisory_lock(connection, self._advisory_capability, shared=False)
        try:
            yield
        finally:
            release_advisory_lock(connection, self._advisory_capability, shared=False)

    @contextmanager
    def _authority_transport(
        self,
        connection: Connection,
        *,
        marker: DurableFileAuthority,
    ) -> Iterator[_RagReleaseBarrierGuard]:
        """Hold Task-22 provider then Task-23 release authorities in global order."""
        if connection.dialect.name != 'postgresql':
            raise RagReleaseAuthorityError('release authority requires PostgreSQL')
        order = begin_rag_lock_order('live_release')
        provider_sidecar = order.acquire('provider_stable_sidecar')
        try:
            with self._provider_safety_release_peer.locked(
                connection,
                order=order,
                sidecar_capability=provider_sidecar,
            ) as provider_guard:
                order.acquire('release_stable_sidecar_advisory')
                with marker.locked(), self._registered_advisory(connection):
                    provider_rows = order.acquire('provider_safety_rows')
                    provider_guard.validate_database_peer(
                        connection,
                        order=order,
                        safety_capability=provider_rows,
                    )
                    order.acquire('release_rows')
                    yield provider_guard
        except RagReleaseAuthorityError:
            raise
        except Exception as exc:
            from backend.app.agent_runtime.rag_provider_safety import (
                RagProviderSafetyError,
            )

            if isinstance(exc, RagProviderSafetyError):
                raise RagReleaseAuthorityError(
                    'provider safety authority is unavailable'
                ) from exc
            raise

    _authority_barrier = _release_scope
    _assert_barrier_guard = _require_release_guard

    @staticmethod
    def _table_state(connection: Connection) -> set[str]:
        return {
            name
            for name in inspect(connection).get_table_names()
            if name.startswith('rag_live_gate_')
        }

    def _require_fresh_database(
        self, connection: Connection, *, disaster: bool = False
    ) -> None:
        state = self._table_state(connection)
        if state and state != set(RAG_RELEASE_TABLE_NAMES):
            raise RagReleaseAuthorityError('release schema is partial')
        if state:
            try:
                assert_rag_release_physical_contract(connection)
            except ValueError as exc:
                raise RagReleaseAuthorityError(
                    'release physical schema is inconsistent'
                ) from exc
            tables = release_tables(build_rag_release_metadata())
            if any(
                connection.scalar(select(func.count()).select_from(table)) != 0
                for table in (
                    tables.ledgers,
                    tables.authorizations,
                    tables.cases,
                    tables.dispatches,
                    tables.transitions,
                    tables.quality_reports,
                )
            ):
                if disaster:
                    raise RagReleaseAuthorityError(
                        'existing validation database identity refuses disaster init'
                    )
                raise RagReleaseAuthorityError('release authority already exists')

    @staticmethod
    def _is_pending_initial_marker(snapshot: RagReleaseSnapshot) -> bool:
        return (
            snapshot.bootstrap_operation == 'release-ledger-init'
            and snapshot.ledger_epoch == 1
            and snapshot.generation == 0
            and snapshot.last_transition_digest is None
            and snapshot.predecessor_marker_digest is None
            and snapshot.rebootstrap_reason_hmac is None
        )

    def _insert_ledger(
        self,
        connection: Connection,
        *,
        snapshot: RagReleaseSnapshot,
        database_identity_uuid: UUID,
        database_identity: ValidationDatabaseIdentity,
        review_envelope_hmac: str,
        review_nonce_hmac: str,
        review_operation: str,
    ) -> dict[str, object]:
        require_lower_hmac(review_envelope_hmac)
        require_lower_hmac(review_nonce_hmac)
        if review_operation not in {
            'release-ledger-init',
            'release-ledger-rebootstrap',
            'release-ledger-disaster-init',
        }:
            raise RagReleaseAuthorityError('release review operation is invalid')
        if (
            snapshot.bootstrap_review_envelope_hmac != review_envelope_hmac
            or snapshot.bootstrap_review_nonce_hmac != review_nonce_hmac
            or snapshot.bootstrap_operation != review_operation
        ):
            raise RagReleaseAuthorityError('release review marker binding differs')
        metadata = build_rag_release_metadata()
        metadata.create_all(connection)
        try:
            assert_rag_release_physical_contract(connection)
        except ValueError as exc:
            raise RagReleaseAuthorityError(
                'release physical schema is inconsistent'
            ) from exc
        table = release_tables(metadata).ledgers
        inserted_row = {
            'ledger_uuid': str(snapshot.ledger_uuid),
            'ledger_epoch': snapshot.ledger_epoch,
            'generation': snapshot.generation,
            'last_transition_digest': snapshot.last_transition_digest,
            'predecessor_marker_digest': snapshot.predecessor_marker_digest,
            'rebootstrap_reason_hmac': snapshot.rebootstrap_reason_hmac,
            'marker_file_digest': snapshot.marker_file_digest,
            'bootstrap_review_envelope_hmac': review_envelope_hmac,
            'bootstrap_review_nonce_hmac': review_nonce_hmac,
            'bootstrap_operation': review_operation,
            'fingerprint_key_version': snapshot.fingerprint_key_version,
            'fingerprint_key_material_verifier': (
                snapshot.fingerprint_key_material_verifier
            ),
            'designated_environment_id_hmac': (snapshot.designated_environment_id_hmac),
            'designated_host_id_hmac': snapshot.designated_host_id_hmac,
            'validation_database_identity_hmac': (
                snapshot.validation_database_identity_hmac
            ),
            'validation_database_locator_hmac': (
                snapshot.validation_database_locator_hmac
            ),
            'validation_database_identity_uuid': str(database_identity_uuid),
            'validation_database_oid': database_identity.database_oid,
        }
        result = connection.execute(insert(table).values(**inserted_row))
        if result.rowcount != 1:
            raise RagReleaseAuthorityError('release ledger insert failed')
        return inserted_row

    @staticmethod
    @contextmanager
    def _rollback_on_error(connection):
        try:
            yield
        except BaseException:
            connection.rollback()
            raise

    def _publication_checkpoint(self, connection, marker, guard):
        """Pin before callbacks; only owned file/SQL reads after publication starts."""
        guard.freeze_provider()
        tables = build_rag_release_metadata().tables

        def image():
            names = self._table_state(connection)
            if names and names != set(tables):
                raise RagReleaseAuthorityError('release schema is inconsistent')
            return {
                name: [
                    dict(row)
                    for row in connection.execute(
                        select(tables[name]).order_by(*tables[name].primary_key.columns)
                    ).mappings()
                ]
                for name in sorted(names)
            }

        before = image()
        self._assert_barrier_guard(guard, connection)

        def checkpoint(expected_marker, inserted_row=None):
            self._assert_barrier_guard(guard, connection).revalidate_provider()
            self._validate_path_set()
            actual_marker = (
                marker._read_bytes_unlocked() if self._marker_path.exists() else None
            )
            if actual_marker != expected_marker:
                raise RagReleaseAuthorityError('release publication marker changed')
            expected = before
            if inserted_row is not None:
                expected = {name: list(before.get(name, [])) for name in tables}
                rows = expected['rag_live_gate_ledgers']
                rows.append(inserted_row)
                rows.sort(key=lambda row: (row['ledger_uuid'], row['ledger_epoch']))
            if image() != expected:
                raise RagReleaseAuthorityError('release publication database changed')
            self._assert_barrier_guard(guard, connection).revalidate_provider()

        return checkpoint

    def initialize(
        self,
        connection: Connection,
        *,
        database_identity: ValidationDatabaseIdentity | None = None,
        review_envelope_hmac: str,
        review_nonce_hmac: str,
    ) -> RagReleaseSnapshot:
        database_identity = _native_database_identity(database_identity)
        self._validate_path_set()
        if self._marker_path.exists():
            raise RagReleaseAuthorityError('release authority already exists')
        marker = DurableFileAuthority(self._marker_path)
        try:
            with (
                self._authority_barrier(connection, marker=marker) as guard,
                self._rollback_on_error(connection),
            ):
                checkpoint = self._publication_checkpoint(connection, marker, guard)
                current = self._current_database_identity(connection, database_identity)
                if self._marker_path.exists():
                    raise RagReleaseAuthorityError('release authority already exists')
                self._require_fresh_database(connection)
                self._assert_review_nonce_fresh(
                    connection,
                    raw_marker=None,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                )
                identity_uuid = uuid4()
                database_hmac = self._database_hmac(current, identity_uuid)
                envelope = self._wrap(
                    self._body(
                        ledger_uuid=uuid4(),
                        ledger_epoch=1,
                        generation=0,
                        last_transition_digest=None,
                        predecessor_marker_digest=None,
                        rebootstrap_reason_hmac=None,
                        database_identity_hmac=database_hmac,
                        database_locator_hmac=self._database_locator_hmac(current),
                        review_envelope_hmac=review_envelope_hmac,
                        review_nonce_hmac=review_nonce_hmac,
                        review_operation='release-ledger-init',
                    )
                )
                _body, snapshot = self._parse(canonical_json_bytes(envelope))
                checkpoint(None)
                marker._replace_unlocked(envelope)
                self._validate_path_set()
                if self._after_marker_replace is not None:
                    self._after_marker_replace()
                checkpoint(canonical_json_bytes(envelope))
                inserted_row = self._insert_ledger(
                    connection,
                    snapshot=snapshot,
                    database_identity_uuid=identity_uuid,
                    database_identity=current,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                    review_operation='release-ledger-init',
                )
                checkpoint(canonical_json_bytes(envelope), inserted_row)
                self._inspect_locked(connection, snapshot, database_identity=current)
                checkpoint(canonical_json_bytes(envelope), inserted_row)
                connection.commit()
                return snapshot
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError('release authority lock failed') from exc

    def _read_marker(self) -> tuple[dict[str, object], RagReleaseSnapshot]:
        self._validate_path_set()
        try:
            authority = DurableFileAuthority.open_runtime(self._marker_path)
            with authority.locked():
                encoded = authority._read_bytes_unlocked()
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError('release marker is unavailable') from exc
        return self._parse(encoded)

    def _ledger_row(
        self,
        connection: Connection,
        snapshot: RagReleaseSnapshot,
    ) -> dict[str, object]:
        row = self._ledger_row_or_none(connection, snapshot)
        if row is None:
            raise RagReleaseAuthorityError('release database peer is missing')
        return row

    def _ledger_row_or_none(
        self,
        connection: Connection,
        snapshot: RagReleaseSnapshot,
    ) -> dict[str, object] | None:
        if self._table_state(connection) != set(RAG_RELEASE_TABLE_NAMES):
            raise RagReleaseAuthorityError('release schema is inconsistent')
        try:
            assert_rag_release_physical_contract(connection)
        except ValueError as exc:
            raise RagReleaseAuthorityError(
                'release physical schema is inconsistent'
            ) from exc
        table = release_tables(build_rag_release_metadata()).ledgers
        row = (
            connection.execute(
                select(table).where(
                    table.c.ledger_uuid == str(snapshot.ledger_uuid),
                    table.c.ledger_epoch == snapshot.ledger_epoch,
                )
            )
            .mappings()
            .one_or_none()
        )
        return None if row is None else dict(row)

    def _assert_review_nonce_fresh(
        self,
        connection: Connection,
        *,
        raw_marker: bytes | None,
        review_envelope_hmac: str,
        review_nonce_hmac: str,
    ) -> None:
        require_lower_hmac(review_envelope_hmac)
        require_lower_hmac(review_nonce_hmac)
        if raw_marker is not None:
            try:
                _body, marker_snapshot = self._parse(raw_marker)
            except RagReleaseAuthorityError:
                marker_snapshot = None
            if (
                marker_snapshot is not None
                and marker_snapshot.bootstrap_review_nonce_hmac == review_nonce_hmac
            ):
                raise RagReleaseAuthorityError('release review nonce was already bound')
        if self._table_state(connection) == set(RAG_RELEASE_TABLE_NAMES):
            table = release_tables(build_rag_release_metadata()).ledgers
            if connection.scalar(
                select(func.count())
                .select_from(table)
                .where(table.c.bootstrap_review_nonce_hmac == review_nonce_hmac)
            ):
                raise RagReleaseAuthorityError(
                    'release review nonce was already consumed'
                )

    @staticmethod
    def _verified_recovery_review(
        verifier: RecoveryReviewVerifier,
        connection: Connection,
        raw_marker: bytes | None,
    ) -> tuple[str, str, str]:
        if not callable(verifier):
            raise RagReleaseAuthorityError('reviewed recovery capability is required')
        value = verifier(connection, raw_marker)
        if type(value) is not tuple or len(value) != 3:
            raise RagReleaseAuthorityError('reviewed recovery capability is invalid')
        envelope_hmac, nonce_hmac, reason_hmac = value
        for item in value:
            require_lower_hmac(item)
        return envelope_hmac, nonce_hmac, reason_hmac

    def inspect(
        self,
        connection: Connection,
        *,
        database_identity: ValidationDatabaseIdentity | None = None,
    ) -> RagReleaseSnapshot:
        database_identity = _native_database_identity(database_identity)
        self._validate_path_set()
        marker = DurableFileAuthority.open_runtime(self._marker_path)
        try:
            with (
                self._authority_barrier(connection, marker=marker) as guard,
                self._rollback_on_error(connection),
            ):
                checkpoint = self._publication_checkpoint(connection, marker, guard)
                raw_marker = marker._read_bytes_unlocked()
                _body, snapshot = self._parse(raw_marker)
                result = self._inspect_locked(
                    connection,
                    snapshot,
                    database_identity=database_identity,
                )
                checkpoint(raw_marker)
                return result
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError('release marker is unavailable') from exc

    def _inspect_locked(
        self,
        connection: Connection,
        snapshot: RagReleaseSnapshot,
        *,
        database_identity: ValidationDatabaseIdentity | None,
    ) -> RagReleaseSnapshot:
        current = self._current_database_identity(connection, database_identity)
        row = self._ledger_row(connection, snapshot)
        try:
            identity_uuid = UUID(str(row['validation_database_identity_uuid']))
        except (TypeError, ValueError) as exc:
            raise RagReleaseAuthorityError(
                'release database identity is invalid'
            ) from exc
        expected_database_hmac = self._database_hmac(current, identity_uuid)
        exact = {
            'generation': snapshot.generation,
            'last_transition_digest': snapshot.last_transition_digest,
            'marker_file_digest': snapshot.marker_file_digest,
            'fingerprint_key_version': snapshot.fingerprint_key_version,
            'fingerprint_key_material_verifier': snapshot.fingerprint_key_material_verifier,
            'designated_environment_id_hmac': snapshot.designated_environment_id_hmac,
            'designated_host_id_hmac': snapshot.designated_host_id_hmac,
            'validation_database_identity_hmac': snapshot.validation_database_identity_hmac,
            'validation_database_locator_hmac': snapshot.validation_database_locator_hmac,
            'bootstrap_review_envelope_hmac': snapshot.bootstrap_review_envelope_hmac,
            'bootstrap_review_nonce_hmac': snapshot.bootstrap_review_nonce_hmac,
            'bootstrap_operation': snapshot.bootstrap_operation,
        }
        if (
            snapshot.designated_environment_id_hmac != self._environment_hmac
            or snapshot.designated_host_id_hmac != self._host_hmac
            or snapshot.validation_database_identity_hmac != expected_database_hmac
            or snapshot.validation_database_locator_hmac
            != self._database_locator_hmac(current)
            or row['validation_database_oid'] != current.database_oid
            or any(row[key] != value for key, value in exact.items())
        ):
            raise RagReleaseAuthorityError(
                'release marker and database identity differ'
            )
        self._verify_transition_history(connection, snapshot)
        return snapshot

    def _verify_transition_history(
        self,
        connection: Connection,
        snapshot: RagReleaseSnapshot,
    ) -> None:
        table = release_tables(build_rag_release_metadata()).transitions
        rows = (
            connection.execute(
                select(table)
                .where(
                    table.c.ledger_uuid == str(snapshot.ledger_uuid),
                    table.c.ledger_epoch == snapshot.ledger_epoch,
                )
                .order_by(table.c.generation)
            )
            .mappings()
            .all()
        )
        if len(rows) != snapshot.generation:
            raise RagReleaseAuthorityError('release transition history is not gapless')
        for expected_generation, row in enumerate(rows, start=1):
            raw = bytes(row['payload_canonical_bytes'])
            try:
                payload = json.loads(raw.decode('utf-8'))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RagReleaseAuthorityError(
                    'release transition payload is invalid'
                ) from exc
            if type(payload) is not dict or canonical_json_bytes(payload) != raw:
                raise RagReleaseAuthorityError(
                    'release transition payload is noncanonical'
                )
            from backend.app.rag.release_ledger import validate_transition_payload

            validated = validate_transition_payload(
                payload, identity_secret=self._secret
            )
            if (
                row['generation'] != expected_generation
                or payload.get('from_generation') != expected_generation - 1
                or payload.get('to_generation') != expected_generation
                or payload.get('ledger_uuid') != str(snapshot.ledger_uuid)
                or payload.get('ledger_epoch') != snapshot.ledger_epoch
                or payload.get('validation_database_identity_hmac')
                != snapshot.validation_database_identity_hmac
                or payload.get('transition_kind') != row['transition_kind']
                or row['transition_digest'] != validated.transition_digest
            ):
                raise RagReleaseAuthorityError(
                    'release transition history verification failed'
                )
        if rows and rows[-1]['transition_digest'] != snapshot.last_transition_digest:
            raise RagReleaseAuthorityError(
                'release transition history tail differs from marker'
            )

    def rebootstrap(
        self,
        connection: Connection,
        *,
        database_identity: ValidationDatabaseIdentity | None = None,
        review_verifier: RecoveryReviewVerifier,
    ) -> RagReleaseSnapshot:
        database_identity = _native_database_identity(database_identity)
        self._validate_path_set()
        marker = DurableFileAuthority.open_runtime(self._marker_path)
        try:
            with (
                self._authority_barrier(connection, marker=marker) as guard,
                self._rollback_on_error(connection),
            ):
                checkpoint = self._publication_checkpoint(connection, marker, guard)
                current = self._current_database_identity(connection, database_identity)
                raw_marker = marker._read_bytes_unlocked()
                _body, prior = self._parse(raw_marker)
                (
                    review_envelope_hmac,
                    review_nonce_hmac,
                    rebootstrap_reason_hmac,
                ) = self._verified_recovery_review(
                    review_verifier, connection, raw_marker
                )
                self._assert_review_nonce_fresh(
                    connection,
                    raw_marker=raw_marker,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                )
                row = self._ledger_row_or_none(connection, prior)
                tables = release_tables(build_rag_release_metadata())
                if row is None:
                    row = (
                        connection.execute(
                            select(tables.ledgers)
                            .where(
                                tables.ledgers.c.ledger_uuid == str(prior.ledger_uuid),
                                tables.ledgers.c.ledger_epoch < prior.ledger_epoch,
                            )
                            .order_by(tables.ledgers.c.ledger_epoch.desc())
                            .limit(1)
                        )
                        .mappings()
                        .one_or_none()
                    )
                    if row is None:
                        raise RagReleaseAuthorityError(
                            'release database peer is missing'
                        )
                    row = dict(row)
                identity_uuid = UUID(str(row['validation_database_identity_uuid']))
                if (
                    row['validation_database_identity_hmac']
                    != self._database_hmac(current, identity_uuid)
                    or prior.validation_database_identity_hmac
                    != row['validation_database_identity_hmac']
                ):
                    raise RagReleaseAuthorityError('release database identity differs')
                if row['ledger_epoch'] == prior.ledger_epoch and (
                    row['generation'] == prior.generation
                    and row['last_transition_digest'] == prior.last_transition_digest
                    and row['marker_file_digest'] == prior.marker_file_digest
                ):
                    raise RagReleaseAuthorityError(
                        'healthy release authority cannot be rebootstraped'
                    )
                envelope = self._wrap(
                    self._body(
                        ledger_uuid=prior.ledger_uuid,
                        ledger_epoch=prior.ledger_epoch + 1,
                        generation=0,
                        last_transition_digest=None,
                        predecessor_marker_digest=prior.marker_file_digest,
                        rebootstrap_reason_hmac=rebootstrap_reason_hmac,
                        database_identity_hmac=(
                            prior.validation_database_identity_hmac
                        ),
                        database_locator_hmac=(prior.validation_database_locator_hmac),
                        review_envelope_hmac=review_envelope_hmac,
                        review_nonce_hmac=review_nonce_hmac,
                        review_operation='release-ledger-rebootstrap',
                    )
                )
                _new_body, snapshot = self._parse(canonical_json_bytes(envelope))
                checkpoint(raw_marker)
                marker._replace_unlocked(envelope)
                self._validate_path_set()
                if self._after_marker_replace is not None:
                    self._after_marker_replace()
                checkpoint(canonical_json_bytes(envelope))
                inserted_row = self._insert_ledger(
                    connection,
                    snapshot=snapshot,
                    database_identity_uuid=identity_uuid,
                    database_identity=current,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                    review_operation='release-ledger-rebootstrap',
                )
                checkpoint(canonical_json_bytes(envelope), inserted_row)
                self._inspect_locked(connection, snapshot, database_identity=current)
                checkpoint(canonical_json_bytes(envelope), inserted_row)
                connection.commit()
                return snapshot
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError('release marker lock failed') from exc

    def disaster_initialize(
        self,
        connection: Connection,
        *,
        database_identity: ValidationDatabaseIdentity | None = None,
        review_verifier: RecoveryReviewVerifier,
    ) -> RagReleaseSnapshot:
        database_identity = _native_database_identity(database_identity)
        self._validate_path_set()
        marker = (
            DurableFileAuthority.open_runtime(self._marker_path)
            if self._marker_path.exists()
            else DurableFileAuthority(self._marker_path)
        )
        try:
            with (
                self._authority_barrier(connection, marker=marker) as guard,
                self._rollback_on_error(connection),
            ):
                checkpoint = self._publication_checkpoint(connection, marker, guard)
                current = self._current_database_identity(connection, database_identity)
                # Attest an existing exact-six schema before any new marker bytes.
                # An absent schema is the legal marker-first init crash state.
                self._require_fresh_database(connection, disaster=True)
                raw_marker: bytes | None = None
                if self._marker_path.exists():
                    try:
                        raw_marker = marker._read_bytes_unlocked()
                        _body, valid = self._parse(raw_marker)
                    except RagReleaseAuthorityError:
                        pass
                    else:
                        if (
                            valid.validation_database_locator_hmac
                            == self._database_locator_hmac(current)
                            and not self._is_pending_initial_marker(valid)
                        ):
                            raise RagReleaseAuthorityError(
                                'valid release authority requires '
                                'same-ledger rebootstrap'
                            )
                (
                    review_envelope_hmac,
                    review_nonce_hmac,
                    rebootstrap_reason_hmac,
                ) = self._verified_recovery_review(
                    review_verifier, connection, raw_marker
                )
                self._assert_review_nonce_fresh(
                    connection,
                    raw_marker=raw_marker,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                )
                identity_uuid = uuid4()
                database_hmac = self._database_hmac(current, identity_uuid)
                envelope = self._wrap(
                    self._body(
                        ledger_uuid=uuid4(),
                        ledger_epoch=1,
                        generation=0,
                        last_transition_digest=None,
                        predecessor_marker_digest=None,
                        rebootstrap_reason_hmac=rebootstrap_reason_hmac,
                        database_identity_hmac=database_hmac,
                        database_locator_hmac=self._database_locator_hmac(current),
                        review_envelope_hmac=review_envelope_hmac,
                        review_nonce_hmac=review_nonce_hmac,
                        review_operation='release-ledger-disaster-init',
                    )
                )
                _body, snapshot = self._parse(canonical_json_bytes(envelope))
                checkpoint(raw_marker)
                marker._replace_unlocked(envelope)
                self._validate_path_set()
                if self._after_marker_replace is not None:
                    self._after_marker_replace()
                checkpoint(canonical_json_bytes(envelope))
                inserted_row = self._insert_ledger(
                    connection,
                    snapshot=snapshot,
                    database_identity_uuid=identity_uuid,
                    database_identity=current,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                    review_operation='release-ledger-disaster-init',
                )
                checkpoint(canonical_json_bytes(envelope), inserted_row)
                self._inspect_locked(connection, snapshot, database_identity=current)
                checkpoint(canonical_json_bytes(envelope), inserted_row)
                connection.commit()
                return snapshot
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError('release authority lock failed') from exc
