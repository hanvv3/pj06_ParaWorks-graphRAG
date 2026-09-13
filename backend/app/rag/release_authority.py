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
    release_advisory_lock,
)
from backend.app.agent_runtime.rag_safety_identity import (
    rag_identity_hmac,
    require_lower_hmac,
)
from backend.app.rag.release_schema import (
    RAG_RELEASE_TABLE_NAMES,
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
        'validation_database_identity_hmac',
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


class ExternalAuthorityPathSetValidator:
    """Validate the provider/release data+stable-lock leaves as one set."""

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
        after_marker_replace: Callable[[], None] | None = None,
    ) -> None:
        if type(identity_secret) is not bytes or len(identity_secret) < 32:
            raise ValueError('release fingerprint key is unavailable')
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
        paths = ExternalAuthorityPathSetValidator.validate(
            provider_path=provider_safety_latch_path,
            release_path=marker_path,
            repository_roots=repository_roots,
            database_backup_roots=database_backup_roots,
        )
        self._provider_path, _provider_lock, self._marker_path, _release_lock = paths
        self._secret = identity_secret
        self._key_version = fingerprint_key_version
        self._environment_hmac = self._identity_hmac(
            'rag-live-designated-environment-id:v1', designated_environment_id
        )
        self._host_hmac = self._identity_hmac(
            'rag-live-designated-host-id:v1', designated_host_id
        )
        self._repository_roots = tuple(repository_roots)
        self._database_backup_roots = tuple(database_backup_roots)
        self._advisory_capability = advisory_capability
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
        return hmac.new(
            self._secret,
            b'paraworks:rag-release-key-material-verifier:v1\x00',
            hashlib.sha256,
        ).hexdigest()

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

    @staticmethod
    def _current_database_identity(
        connection: Connection,
        supplied: ValidationDatabaseIdentity | None,
    ) -> ValidationDatabaseIdentity:
        if supplied is not None:
            return supplied
        if connection.dialect.name != 'postgresql':
            raise RagReleaseAuthorityError(
                'an explicit validation database identity is required outside PostgreSQL'
            )
        row = connection.exec_driver_sql(
            'SELECT current_database(), oid FROM pg_database '
            'WHERE datname = current_database()'
        ).one()
        return ValidationDatabaseIdentity(database_name=row[0], database_oid=row[1])

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
            'validation_database_identity_hmac': database_identity_hmac,
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
            ):
                require_lower_hmac(body[key])
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
        )

    def _validate_path_set(self) -> None:
        ExternalAuthorityPathSetValidator.validate(
            provider_path=self._provider_path,
            release_path=self._marker_path,
            repository_roots=self._repository_roots,
            database_backup_roots=self._database_backup_roots,
        )
        provider = DurableFileAuthority.open_runtime(self._provider_path)
        try:
            with provider.locked():
                provider._read_bytes_unlocked()
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError(
                'provider safety authority is unavailable'
            ) from exc

    @contextmanager
    def _registered_advisory(self, connection: Connection) -> Iterator[None]:
        if connection.dialect.name != 'postgresql':
            yield
            return
        if self._advisory_capability is None:
            raise RagReleaseAuthorityError(
                'registered release advisory capability is required'
            )
        acquire_advisory_lock(connection, self._advisory_capability, shared=False)
        try:
            yield
        finally:
            release_advisory_lock(connection, self._advisory_capability, shared=False)

    @staticmethod
    def _table_state(connection: Connection) -> set[str]:
        return set(inspect(connection).get_table_names()) & set(RAG_RELEASE_TABLE_NAMES)

    def _require_fresh_database(self, connection: Connection) -> None:
        state = self._table_state(connection)
        if state and state != set(RAG_RELEASE_TABLE_NAMES):
            raise RagReleaseAuthorityError('release schema is partial')
        if state:
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
                raise RagReleaseAuthorityError('release authority already exists')

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
    ) -> None:
        require_lower_hmac(review_envelope_hmac)
        require_lower_hmac(review_nonce_hmac)
        if review_operation not in {
            'release-ledger-init',
            'release-ledger-rebootstrap',
            'release-ledger-disaster-init',
        }:
            raise RagReleaseAuthorityError('release review operation is invalid')
        metadata = build_rag_release_metadata()
        metadata.create_all(connection)
        table = release_tables(metadata).ledgers
        result = connection.execute(
            insert(table).values(
                ledger_uuid=str(snapshot.ledger_uuid),
                ledger_epoch=snapshot.ledger_epoch,
                generation=snapshot.generation,
                last_transition_digest=snapshot.last_transition_digest,
                predecessor_marker_digest=snapshot.predecessor_marker_digest,
                rebootstrap_reason_hmac=snapshot.rebootstrap_reason_hmac,
                marker_file_digest=snapshot.marker_file_digest,
                bootstrap_review_envelope_hmac=review_envelope_hmac,
                bootstrap_review_nonce_hmac=review_nonce_hmac,
                bootstrap_operation=review_operation,
                fingerprint_key_version=snapshot.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    snapshot.fingerprint_key_material_verifier
                ),
                designated_environment_id_hmac=(
                    snapshot.designated_environment_id_hmac
                ),
                designated_host_id_hmac=snapshot.designated_host_id_hmac,
                validation_database_identity_hmac=(
                    snapshot.validation_database_identity_hmac
                ),
                validation_database_identity_uuid=str(database_identity_uuid),
                validation_database_oid=database_identity.database_oid,
            )
        )
        if result.rowcount != 1:
            raise RagReleaseAuthorityError('release ledger insert failed')

    def initialize(
        self,
        connection: Connection,
        *,
        database_identity: ValidationDatabaseIdentity | None = None,
        review_envelope_hmac: str,
        review_nonce_hmac: str,
    ) -> RagReleaseSnapshot:
        self._validate_path_set()
        if self._marker_path.exists():
            raise RagReleaseAuthorityError('release authority already exists')
        current = self._current_database_identity(connection, database_identity)
        marker = DurableFileAuthority(self._marker_path)
        try:
            with marker.locked(), self._registered_advisory(connection):
                if self._marker_path.exists():
                    raise RagReleaseAuthorityError(
                        'release authority already exists'
                    )
                self._require_fresh_database(connection)
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
                    )
                )
                _body, snapshot = self._parse(canonical_json_bytes(envelope))
                marker._replace_unlocked(envelope)
                self._validate_path_set()
                if self._after_marker_replace is not None:
                    self._after_marker_replace()
                self._insert_ledger(
                    connection,
                    snapshot=snapshot,
                    database_identity_uuid=identity_uuid,
                    database_identity=current,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                    review_operation='release-ledger-init',
                )
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
        if self._table_state(connection) != set(RAG_RELEASE_TABLE_NAMES):
            raise RagReleaseAuthorityError('release schema is inconsistent')
        table = release_tables(build_rag_release_metadata()).ledgers
        row = connection.execute(
            select(table).where(
                table.c.ledger_uuid == str(snapshot.ledger_uuid),
                table.c.ledger_epoch == snapshot.ledger_epoch,
            )
        ).mappings().one_or_none()
        if row is None:
            raise RagReleaseAuthorityError('release database peer is missing')
        return dict(row)

    def inspect(
        self,
        connection: Connection,
        *,
        database_identity: ValidationDatabaseIdentity | None = None,
    ) -> RagReleaseSnapshot:
        self._validate_path_set()
        marker = DurableFileAuthority.open_runtime(self._marker_path)
        try:
            with marker.locked(), self._registered_advisory(connection):
                _body, snapshot = self._parse(marker._read_bytes_unlocked())
                return self._inspect_locked(
                    connection,
                    snapshot,
                    database_identity=database_identity,
                )
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
            raise RagReleaseAuthorityError('release database identity is invalid') from exc
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
        }
        if (
            snapshot.designated_environment_id_hmac != self._environment_hmac
            or snapshot.designated_host_id_hmac != self._host_hmac
            or snapshot.validation_database_identity_hmac != expected_database_hmac
            or row['validation_database_oid'] != current.database_oid
            or any(row[key] != value for key, value in exact.items())
        ):
            raise RagReleaseAuthorityError('release marker and database identity differ')
        self._verify_transition_history(connection, snapshot)
        return snapshot

    def _verify_transition_history(
        self,
        connection: Connection,
        snapshot: RagReleaseSnapshot,
    ) -> None:
        table = release_tables(build_rag_release_metadata()).transitions
        rows = connection.execute(
            select(table)
            .where(
                table.c.ledger_uuid == str(snapshot.ledger_uuid),
                table.c.ledger_epoch == snapshot.ledger_epoch,
            )
            .order_by(table.c.generation)
        ).mappings().all()
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
            expected_digest = rag_identity_hmac(
                payload,
                secret=self._secret,
                schema_version='rag-release-ledger-transition:v1',
                policy_version=_POLICY_VERSION,
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
                or row['transition_digest'] != expected_digest
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
        rebootstrap_reason_hmac: str,
        review_envelope_hmac: str,
        review_nonce_hmac: str,
    ) -> RagReleaseSnapshot:
        require_lower_hmac(rebootstrap_reason_hmac)
        self._validate_path_set()
        current = self._current_database_identity(connection, database_identity)
        marker = DurableFileAuthority.open_runtime(self._marker_path)
        try:
            with marker.locked(), self._registered_advisory(connection):
                body, prior = self._parse(marker._read_bytes_unlocked())
                row = self._ledger_row(connection, prior)
                identity_uuid = UUID(str(row['validation_database_identity_uuid']))
                if (
                    row['validation_database_identity_hmac']
                    != self._database_hmac(current, identity_uuid)
                    or prior.validation_database_identity_hmac
                    != row['validation_database_identity_hmac']
                ):
                    raise RagReleaseAuthorityError(
                        'release database identity differs'
                    )
                if (
                    row['generation'] == prior.generation
                    and row['last_transition_digest']
                    == prior.last_transition_digest
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
                    )
                )
                _new_body, snapshot = self._parse(canonical_json_bytes(envelope))
                marker._replace_unlocked(envelope)
                self._validate_path_set()
                if self._after_marker_replace is not None:
                    self._after_marker_replace()
                self._insert_ledger(
                    connection,
                    snapshot=snapshot,
                    database_identity_uuid=identity_uuid,
                    database_identity=current,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                    review_operation='release-ledger-rebootstrap',
                )
                connection.commit()
                return snapshot
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError('release marker lock failed') from exc

    def disaster_initialize(
        self,
        connection: Connection,
        *,
        database_identity: ValidationDatabaseIdentity | None = None,
        rebootstrap_reason_hmac: str,
        review_envelope_hmac: str,
        review_nonce_hmac: str,
    ) -> RagReleaseSnapshot:
        require_lower_hmac(rebootstrap_reason_hmac)
        self._validate_path_set()
        current = self._current_database_identity(connection, database_identity)
        marker = (
            DurableFileAuthority.open_runtime(self._marker_path)
            if self._marker_path.exists()
            else DurableFileAuthority(self._marker_path)
        )
        try:
            with marker.locked(), self._registered_advisory(connection):
                if self._marker_path.exists():
                    try:
                        _body, valid = self._parse(marker._read_bytes_unlocked())
                        row = self._ledger_row(connection, valid)
                        prior_uuid = UUID(
                            str(row['validation_database_identity_uuid'])
                        )
                        if (
                            row['validation_database_identity_hmac']
                            == self._database_hmac(current, prior_uuid)
                        ):
                            raise RagReleaseAuthorityError(
                                'valid release authority requires '
                                'same-ledger rebootstrap'
                            )
                    except RagReleaseAuthorityError as exc:
                        if 'requires same-ledger' in str(exc):
                            raise
                state = self._table_state(connection)
                if state and state != set(RAG_RELEASE_TABLE_NAMES):
                    raise RagReleaseAuthorityError('release schema is partial')
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
                    )
                )
                _body, snapshot = self._parse(canonical_json_bytes(envelope))
                marker._replace_unlocked(envelope)
                self._validate_path_set()
                if self._after_marker_replace is not None:
                    self._after_marker_replace()
                self._insert_ledger(
                    connection,
                    snapshot=snapshot,
                    database_identity_uuid=identity_uuid,
                    database_identity=current,
                    review_envelope_hmac=review_envelope_hmac,
                    review_nonce_hmac=review_nonce_hmac,
                    review_operation='release-ledger-disaster-init',
                )
                connection.commit()
                return snapshot
        except DurableFileAuthorityError as exc:
            raise RagReleaseAuthorityError('release authority lock failed') from exc
