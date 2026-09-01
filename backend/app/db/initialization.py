from __future__ import annotations

import secrets
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from threading import Condition, RLock, get_ident
from time import monotonic_ns
from types import MappingProxyType
from typing import Literal, NamedTuple

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import (
    ArgumentError,
    DBAPIError,
    DisconnectionError,
    InterfaceError,
    NoSuchModuleError,
    OperationalError,
)
from sqlalchemy.exc import (
    TimeoutError as SQLAlchemyTimeoutError,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool


class DatabaseConfigurationError(RuntimeError):
    code = 'database_configuration_invalid'


class DatabaseInitializationError(RuntimeError):
    code = 'database_initialization_failed'


class PostgresRuntimeHealthUnavailableError(TypeError):
    """Sanitized refusal after the trusted runtime is fail-stopped."""


_POSTGRES_BOOTSTRAP_SEAL = object()
_POSTGRES_RUNTIME_HEALTH_SEAL = object()
_POSTGRES_RUNTIME_HEALTH_LEASE_SEAL = object()
_POSTGRES_CLEANUP_OWNER_SEAL = object()
_POSTGRES_EMERGENCY_TRANSITION_SEAL = object()
_POSTGRES_PHYSICAL_CLEANUP_SEAL = object()


@dataclass(frozen=True, slots=True)
class PostgresRuntimeHealthSnapshot:
    healthy: bool
    failure_count: int
    code: Literal['rag_postgres_transport_cleanup_failed'] | None
    first_failure_monotonic_ns: int | None


@dataclass(slots=True, repr=False)
class _PostgresRuntimeHealthLease:
    health: TrustedPostgresRuntimeHealth = field(repr=False)
    epoch: int
    purpose: str
    active: bool = field(default=True, repr=False)
    _seal: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PostgresRuntimeExclusiveTicket:
    sequence: int
    owner_thread_id: int
    purpose: Literal['cleanup', 'poison']


@dataclass(slots=True, repr=False)
class _PostgresCleanupOwnerCapability:
    health: TrustedPostgresRuntimeHealth = field(repr=False)
    owner_thread_id: int
    generation: int
    active: bool = field(default=True, repr=False)
    _seal: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _PostgresEmergencyCleanupCapability:
    """Opaque handle; all cleanup authority lives in the private registry."""


class _PostgresEmergencyCleanupRecord(NamedTuple):
    capability: _PostgresEmergencyCleanupCapability
    authority: object
    operation_lease: _PostgresRuntimeHealthLease
    operation_epoch: int
    operation_purpose: str
    owner_thread_id: int
    phase: Literal[
        'ACTIVE',
        'REVOKED_UNCERTAIN',
        'REVOKED_CLEAN',
    ] = 'ACTIVE'
    revision: int = 0
    cleanup_ticket: _PostgresRuntimeExclusiveTicket | None = None
    cleanup_generation: int | None = None
    physical_cleanup_registered: bool = False


class _PostgresExclusiveOwnerState(NamedTuple):
    owner_thread_id: int
    depth: int
    generation: int
    capability: _PostgresCleanupOwnerCapability
    operation_lease_id: int | None = None


class _PostgresPhysicalCleanupResponsibility(NamedTuple):
    operation_lease: _PostgresRuntimeHealthLease
    authority: object
    owner_thread_id: int
    steps: tuple[Callable[[], bool | None], ...]
    seal: object


class TrustedPostgresRuntimeHealth:
    """Process-local fail-stop admission shared by one DB runtime."""

    __slots__ = (
        '_active_effects',
        '_active_operation_leases',
        '_active_operation_authorities',
        '_condition',
        '_effect_depths',
        '_epoch',
        '_exclusive_generation_sequence',
        '_exclusive_owner_state',
        '_exclusive_ticket_sequence',
        '_exclusive_tickets',
        '_emergency_cleanup_operation_index',
        '_emergency_cleanup_records',
        '_failure_count',
        '_first_failure_monotonic_ns',
        '_healthy',
        '_lock',
        '_poison_requested',
        '_physical_cleanup_responsibilities',
        '_seal',
    )

    def __init__(self, *, _seal: object) -> None:
        if _seal is not _POSTGRES_RUNTIME_HEALTH_SEAL:
            raise TypeError('trusted PostgreSQL runtime health is unavailable')
        self._epoch = 0
        self._failure_count = 0
        self._first_failure_monotonic_ns: int | None = None
        self._healthy = True
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._active_effects = 0
        self._active_operation_leases: dict[
            int,
            _PostgresRuntimeHealthLease,
        ] = {}
        self._active_operation_authorities: dict[int, object] = {}
        self._effect_depths: dict[int, int] = {}
        self._exclusive_owner_state: _PostgresExclusiveOwnerState | None = None
        self._exclusive_generation_sequence = 0
        self._exclusive_ticket_sequence = 0
        self._exclusive_tickets: deque[_PostgresRuntimeExclusiveTicket] = deque()
        self._emergency_cleanup_records: dict[
            int,
            _PostgresEmergencyCleanupRecord,
        ] = {}
        self._emergency_cleanup_operation_index: dict[
            int,
            _PostgresEmergencyCleanupCapability,
        ] = {}
        self._poison_requested = False
        self._physical_cleanup_responsibilities: dict[
            int,
            _PostgresPhysicalCleanupResponsibility,
        ] = {}
        self._seal = _seal

    @property
    def snapshot(self) -> PostgresRuntimeHealthSnapshot:
        with self._lock:
            return PostgresRuntimeHealthSnapshot(
                healthy=self._healthy,
                failure_count=self._failure_count,
                code=(
                    None
                    if self._healthy
                    else 'rag_postgres_transport_cleanup_failed'
                ),
                first_failure_monotonic_ns=self._first_failure_monotonic_ns,
            )

    @property
    def _exclusive_waiters(self) -> int:
        return len(self._exclusive_tickets)

    @property
    def _exclusive_owner(self) -> int | None:
        state = self._exclusive_owner_state
        return None if state is None else state.owner_thread_id

    @property
    def _exclusive_depth(self) -> int:
        state = self._exclusive_owner_state
        return 0 if state is None else state.depth

    @property
    def _exclusive_generation(self) -> int:
        state = self._exclusive_owner_state
        return self._exclusive_generation_sequence if state is None else state.generation

    @property
    def _cleanup_owner_capability(self) -> _PostgresCleanupOwnerCapability | None:
        state = self._exclusive_owner_state
        return None if state is None else state.capability

    @property
    def _emergency_cleanup_capabilities(
        self,
    ) -> dict[int, _PostgresEmergencyCleanupRecord]:
        """Compatibility view; the one authoritative registry is records."""
        return {
            id(record.operation_lease): record
            for record in self._emergency_cleanup_records.values()
            if record.phase == 'ACTIVE'
        }

    @contextmanager
    def _operation(self, purpose: str):
        if type(purpose) is not str or not purpose:
            raise TypeError('PostgreSQL runtime health purpose is invalid')
        with self._condition:
            if not self._healthy or self._poison_requested:
                raise PostgresRuntimeHealthUnavailableError(
                    'RAG PostgreSQL runtime health is fail-stopped'
                )
            lease = _PostgresRuntimeHealthLease(
                health=self,
                epoch=self._epoch,
                purpose=purpose,
                _seal=_POSTGRES_RUNTIME_HEALTH_LEASE_SEAL,
            )
            self._active_operation_leases[id(lease)] = lease
        try:
            yield lease
        finally:
            physical: _PostgresPhysicalCleanupResponsibility | None
            physical_valid = False
            with self._condition:
                physical = self._physical_cleanup_responsibilities.get(id(lease))
                if physical is not None:
                    physical_valid = not (
                        physical.seal is not _POSTGRES_PHYSICAL_CLEANUP_SEAL
                        or physical.operation_lease is not lease
                        or self._active_operation_authorities.get(id(lease))
                        is not physical.authority
                        or physical.owner_thread_id != get_ident()
                    )
            # Trusted cleanup adapters may take lifecycle/session/bootstrap locks.
            # Never execute them while holding the runtime-health condition.
            physical_failed = physical is not None and not physical_valid
            if physical is not None and physical_valid:
                for step in physical.steps:
                    completed = False
                    for _attempt in range(2):
                        try:
                            completed = not bool(step())
                        except BaseException:
                            completed = False
                        if completed:
                            break
                    if not completed:
                        physical_failed = True
            with self._condition:
                if physical is not None:
                    self._physical_cleanup_responsibilities.pop(id(lease), None)
                if physical_failed:
                    self._force_fail_stop_locked()
                capability = self._emergency_cleanup_operation_index.get(id(lease))
                record = (
                    None
                    if capability is None
                    else self._emergency_cleanup_records.get(id(capability))
                )
                if record is None:
                    record = next(
                        (
                            candidate
                            for candidate in self._emergency_cleanup_records.values()
                            if candidate.operation_lease is lease
                        ),
                        None,
                    )
                if record is not None and record.operation_lease is lease:
                    if (
                        record.phase != 'REVOKED_CLEAN'
                        or (record.physical_cleanup_registered and physical is None)
                    ):
                        self._force_fail_stop_locked()
                    self._drain_exact_emergency_record_locked(record)
                    self._emergency_cleanup_records.pop(id(record.capability), None)
                    self._emergency_cleanup_operation_index.pop(id(lease), None)
                lease.active = False
                current = self._active_operation_leases.get(id(lease))
                if current is lease:
                    self._active_operation_leases.pop(id(lease), None)
                    self._active_operation_authorities.pop(id(lease), None)
                self._condition.notify_all()

    def _register_physical_cleanup(
        self,
        operation_lease: _PostgresRuntimeHealthLease,
        *,
        authority: object,
        steps: tuple[Callable[[], bool | None], ...],
        _seal: object,
    ) -> None:
        with self._condition:
            self._require_emergency_cleanup_capability(
                self._emergency_cleanup_operation_index[id(operation_lease)],
                authority=authority,
            )
            if (
                _seal is not _POSTGRES_PHYSICAL_CLEANUP_SEAL
                or not steps
                or any(not callable(step) for step in steps)
                or id(operation_lease) in self._physical_cleanup_responsibilities
            ):
                raise TypeError('PostgreSQL physical cleanup authority changed')
            self._physical_cleanup_responsibilities[id(operation_lease)] = (
                _PostgresPhysicalCleanupResponsibility(
                    operation_lease=operation_lease,
                    authority=authority,
                    owner_thread_id=get_ident(),
                    steps=steps,
                    seal=_POSTGRES_PHYSICAL_CLEANUP_SEAL,
                )
            )
            capability = self._emergency_cleanup_operation_index[id(operation_lease)]
            record = self._emergency_cleanup_records[id(capability)]
            self._emergency_cleanup_records[id(capability)] = record._replace(
                revision=record.revision + 1,
                physical_cleanup_registered=True,
            )

    @contextmanager
    def _guard(self, lease: _PostgresRuntimeHealthLease):
        self._enter_effect(lease)
        try:
            yield
        finally:
            self._exit_effect()

    @contextmanager
    def _effect(self, purpose: str):
        """Admit one compound effect under the shared healthy epoch."""
        with self._operation(purpose) as lease, self._guard(lease):
            yield

    def _enter_effect(self, lease: _PostgresRuntimeHealthLease) -> None:
        thread_id = get_ident()
        with self._condition:
            depth = self._effect_depths.get(thread_id, 0)
            if depth:
                self._require_effect_lease(lease)
                self._effect_depths[thread_id] = depth + 1
                return
            while self._exclusive_owner is not None or self._exclusive_tickets:
                self._condition.wait()
            self._require_effect_lease(lease)
            self._active_effects += 1
            self._effect_depths[thread_id] = 1

    def _exit_effect(self) -> None:
        thread_id = get_ident()
        with self._condition:
            depth = self._effect_depths.get(thread_id, 0)
            if depth <= 0:
                raise RuntimeError('PostgreSQL runtime effect lease underflow')
            if depth > 1:
                self._effect_depths[thread_id] = depth - 1
                return
            del self._effect_depths[thread_id]
            self._active_effects -= 1
            if self._active_effects < 0:
                raise RuntimeError('PostgreSQL runtime effect count underflow')
            if self._active_effects == 0 and self._poison_requested:
                self._apply_poison()
            self._condition.notify_all()

    def _require_effect_lease(
        self,
        lease: _PostgresRuntimeHealthLease,
    ) -> None:
        if (
            type(lease) is not _PostgresRuntimeHealthLease
            or lease._seal is not _POSTGRES_RUNTIME_HEALTH_LEASE_SEAL
            or lease.health is not self
            or not lease.active
            or self._active_operation_leases.get(id(lease)) is not lease
            or lease.epoch != self._epoch
            or not self._healthy
            or self._poison_requested
        ):
            raise PostgresRuntimeHealthUnavailableError(
                'RAG PostgreSQL runtime health lease changed'
            )

    def _poison(self) -> None:
        thread_id = get_ident()
        with self._condition:
            if not self._healthy:
                return
            if self._exclusive_owner == thread_id:
                self._apply_poison()
                return
            if self._effect_depths.get(thread_id, 0):
                self._poison_requested = True
                self._condition.notify_all()
                return
            ticket = self._enqueue_exclusive_ticket(
                thread_id=thread_id,
                purpose='poison',
            )
            try:
                while self._healthy and (
                    not self._is_head_exclusive_ticket(ticket)
                    or self._active_effects
                    or self._exclusive_owner is not None
                ):
                    self._condition.wait()
                if not self._healthy:
                    self._cancel_exclusive_ticket(ticket)
                    return
                self._claim_head_exclusive_ticket(ticket)
                self._apply_poison()
                self._condition.notify_all()
            except BaseException:
                self._cancel_exclusive_ticket(ticket)
                raise

    def _apply_poison(self) -> None:
        if not self._healthy:
            return
        self._healthy = False
        self._poison_requested = False
        self._epoch += 1
        self._failure_count = 1
        self._first_failure_monotonic_ns = monotonic_ns()

    @contextmanager
    def _cleanup_boundary(self):
        """Linearize cleanup success or poison before another admission."""
        capability = self._enter_cleanup()
        try:
            yield capability
        finally:
            self._exit_cleanup(capability)

    def _enter_cleanup(
        self,
        *,
        emergency_capability: _PostgresEmergencyCleanupCapability | None = None,
    ) -> _PostgresCleanupOwnerCapability:
        thread_id = get_ident()
        with self._condition:
            if emergency_capability is not None:
                self._require_emergency_cleanup_capability(
                    emergency_capability
                )
            if self._effect_depths.get(thread_id, 0):
                raise RuntimeError(
                    'cleanup cannot run inside a PostgreSQL runtime effect'
                )
            if self._exclusive_owner == thread_id:
                if self._exclusive_tickets:
                    raise RuntimeError(
                        'cleanup reentrancy cannot bypass queued authority'
                    )
                owner_state = self._exclusive_owner_state
                capability = self._cleanup_owner_capability
                if capability is None:
                    raise RuntimeError('PostgreSQL cleanup owner is unavailable')
                assert owner_state is not None
                self._exclusive_owner_state = owner_state._replace(
                    depth=owner_state.depth + 1
                )
                return capability
            ticket = (
                self._begin_emergency_cleanup_ticket(
                    emergency_capability,
                    _seal=_POSTGRES_EMERGENCY_TRANSITION_SEAL,
                )
                if emergency_capability is not None
                else self._enqueue_exclusive_ticket(
                    thread_id=thread_id,
                    purpose='cleanup',
                )
            )
            try:
                while (
                    not self._is_head_exclusive_ticket(ticket)
                    or self._active_effects
                    or self._exclusive_owner is not None
                ):
                    self._condition.wait()
                if emergency_capability is not None:
                    self._emergency_cleanup_transition_checkpoint('after_wait')
                    capability = self._claim_emergency_cleanup_ticket(
                        emergency_capability,
                        ticket,
                        _seal=_POSTGRES_EMERGENCY_TRANSITION_SEAL,
                    )
                else:
                    capability = self._claim_plain_cleanup_ticket(ticket)
                return capability
            except BaseException:
                if emergency_capability is not None:
                    self._rollback_emergency_cleanup_transition(
                        emergency_capability,
                        ticket,
                        _seal=_POSTGRES_EMERGENCY_TRANSITION_SEAL,
                    )
                else:
                    self._cancel_exclusive_ticket(ticket)
                raise

    def _exit_cleanup(
        self,
        capability: _PostgresCleanupOwnerCapability,
    ) -> None:
        with self._condition:
            self._require_cleanup_owner(capability)
            owner_state = self._exclusive_owner_state
            assert owner_state is not None
            if owner_state.depth == 1:
                capability.active = False
                self._exclusive_owner_state = None
                self._condition.notify_all()
            else:
                self._exclusive_owner_state = owner_state._replace(
                    depth=owner_state.depth - 1
                )

    def _require_cleanup_owner(
        self,
        capability: object,
        *,
        outermost: bool = False,
    ) -> None:
        with self._condition:
            if (
                type(capability) is not _PostgresCleanupOwnerCapability
                or capability._seal is not _POSTGRES_CLEANUP_OWNER_SEAL
                or capability.health is not self
                or not capability.active
                or self._cleanup_owner_capability is not capability
                or capability.owner_thread_id != get_ident()
                or self._exclusive_owner != get_ident()
                or self._exclusive_depth <= 0
                or (outermost and self._exclusive_depth != 1)
                or capability.generation != self._exclusive_generation
            ):
                raise TypeError('PostgreSQL cleanup owner capability changed')

    def _claim_plain_cleanup_ticket(
        self,
        ticket: _PostgresRuntimeExclusiveTicket,
    ) -> _PostgresCleanupOwnerCapability:
        if not self._is_head_exclusive_ticket(ticket):
            raise RuntimeError('PostgreSQL cleanup authority ticket changed')
        generation = self._exclusive_generation_sequence + 1
        capability = _PostgresCleanupOwnerCapability(
            health=self,
            owner_thread_id=ticket.owner_thread_id,
            generation=generation,
            _seal=_POSTGRES_CLEANUP_OWNER_SEAL,
        )
        owner_state = _PostgresExclusiveOwnerState(
            owner_thread_id=ticket.owner_thread_id,
            depth=1,
            generation=generation,
            capability=capability,
        )
        claimed = True
        owner_published = False
        try:
            self._claim_head_exclusive_ticket(ticket)
            self._emergency_cleanup_transition_checkpoint('after_head_removal')
            self._emergency_cleanup_transition_checkpoint(
                'before_owner_state_publication'
            )
            self._exclusive_generation_sequence = generation
            owner_published = True
            self._exclusive_owner_state = owner_state
            self._emergency_cleanup_transition_checkpoint(
                'after_owner_state_publication'
            )
            return capability
        except BaseException:
            if owner_published and self._exclusive_owner_state is owner_state:
                capability.active = False
                self._exclusive_owner_state = None
            if claimed and ticket not in self._exclusive_tickets:
                self._exclusive_tickets.appendleft(ticket)
            self._condition.notify_all()
            raise

    def _mint_emergency_cleanup_capability(
        self,
        operation_lease: _PostgresRuntimeHealthLease,
        *,
        authority: object,
    ) -> _PostgresEmergencyCleanupCapability:
        """Seal cleanup responsibility to one admitted runtime operation."""
        with self._condition:
            return self._mint_emergency_cleanup_capability_locked(
                operation_lease,
                authority=authority,
            )

    def _force_mint_emergency_cleanup_capability(
        self,
        operation_lease: _PostgresRuntimeHealthLease,
        *,
        authority: object,
    ) -> _PostgresEmergencyCleanupCapability:
        """Non-hook bootstrap for responsibility before any external effect."""
        with self._condition:
            existing_capability = self._emergency_cleanup_operation_index.get(
                id(operation_lease)
            )
            if existing_capability is not None:
                self._require_emergency_cleanup_capability(
                    existing_capability,
                    authority=authority,
                )
                return existing_capability
            return self._mint_emergency_cleanup_capability_locked(
                operation_lease,
                authority=authority,
            )

    def _mint_emergency_cleanup_capability_locked(
        self,
        operation_lease: _PostgresRuntimeHealthLease,
        *,
        authority: object,
    ) -> _PostgresEmergencyCleanupCapability:
        self._require_effect_lease(operation_lease)
        if (
            authority is None
            or operation_lease.purpose != 'rag_finalization_or_recovery'
            or id(operation_lease) in self._emergency_cleanup_operation_index
        ):
            raise TypeError('PostgreSQL emergency cleanup capability changed')
        capability = _PostgresEmergencyCleanupCapability()
        self._emergency_cleanup_records[id(capability)] = (
            _PostgresEmergencyCleanupRecord(
                capability=capability,
                authority=authority,
                operation_lease=operation_lease,
                operation_epoch=operation_lease.epoch,
                operation_purpose=operation_lease.purpose,
                owner_thread_id=get_ident(),
            )
        )
        self._emergency_cleanup_operation_index[id(operation_lease)] = capability
        self._active_operation_authorities[id(operation_lease)] = authority
        return capability

    def _emergency_fail_stop(
        self,
        capability: _PostgresEmergencyCleanupCapability,
    ) -> None:
        """Poison immediately without stealing a foreign FIFO cleanup owner."""
        with self._condition:
            self._require_emergency_cleanup_capability(capability)
            self._force_fail_stop_locked()
            self._condition.notify_all()

    def _force_emergency_fail_stop(
        self,
        capability: _PostgresEmergencyCleanupCapability,
    ) -> None:
        """Mandatory one-way latch used when the normal cleanup hook fails."""
        with self._condition:
            self._require_emergency_cleanup_capability(capability)
            self._force_fail_stop_locked()
            self._condition.notify_all()

    def _force_fail_stop_locked(self) -> None:
        if not self._healthy:
            return
        self._healthy = False
        self._poison_requested = False
        self._epoch += 1
        self._failure_count = 1
        self._first_failure_monotonic_ns = monotonic_ns()

    def _finish_emergency_cleanup(
        self,
        capability: _PostgresEmergencyCleanupCapability,
    ) -> None:
        """Expire the sealed fallback and drain only its own interrupted gate state."""
        with self._condition:
            self._require_emergency_cleanup_capability(capability)
            self._revoke_emergency_cleanup_locked(capability)

    def _force_finish_emergency_cleanup(
        self,
        capability: _PostgresEmergencyCleanupCapability,
    ) -> None:
        """Unconditionally revoke the exact operation registry entry."""
        with self._condition:
            self._require_emergency_cleanup_capability(capability)
            self._revoke_emergency_cleanup_locked(capability)

    def _revoke_emergency_cleanup_locked(
        self,
        capability: _PostgresEmergencyCleanupCapability,
    ) -> None:
        record = self._emergency_cleanup_record(capability)
        self._emergency_cleanup_transition_checkpoint(
            'before_revoke_uncertain_publication'
        )
        phased = record._replace(
            phase='REVOKED_UNCERTAIN',
            revision=record.revision + 1,
        )
        self._emergency_cleanup_records[id(capability)] = phased
        self._emergency_cleanup_transition_checkpoint(
            'after_revoke_uncertain_publication'
        )
        self._drain_exact_emergency_record_locked(phased)
        self._emergency_cleanup_transition_checkpoint('after_revoke_resource_drain')
        self._emergency_cleanup_transition_checkpoint(
            'before_revoke_clean_attestation'
        )
        clean = phased._replace(
            phase='REVOKED_CLEAN',
            revision=phased.revision + 1,
            cleanup_ticket=None,
            cleanup_generation=None,
        )
        self._emergency_cleanup_records[id(capability)] = clean
        self._emergency_cleanup_transition_checkpoint(
            'after_revoke_clean_attestation'
        )
        self._condition.notify_all()

    def _drain_exact_emergency_record_locked(
        self,
        record: _PostgresEmergencyCleanupRecord,
    ) -> None:
        ticket = record.cleanup_ticket
        if ticket is not None:
            self._cancel_exclusive_ticket(ticket)
        owner_state = self._exclusive_owner_state
        if (
            record.cleanup_generation is not None
            and owner_state is not None
            and owner_state.generation == record.cleanup_generation
            and owner_state.owner_thread_id == record.owner_thread_id
            and owner_state.operation_lease_id == id(record.operation_lease)
        ):
            owner_state.capability.active = False
            self._exclusive_owner_state = None

    def _force_emergency_cleanup_disposition(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        *,
        authority: object,
        poison: bool,
    ) -> None:
        """Idempotently finish exact active/revoked cleanup from registry only."""
        with self._condition:
            record = self._emergency_cleanup_record_or_none(capability)
            if record is not None:
                self._require_emergency_cleanup_capability(
                    capability,
                    authority=authority,
                )
                self._revoke_emergency_cleanup_locked(capability)
                record = self._emergency_cleanup_records.get(id(capability))
            else:
                record = self._emergency_cleanup_records.get(id(capability))
            if (
                record is None
                or record.capability is not capability
                or record.phase
                not in {'REVOKED_UNCERTAIN', 'REVOKED_CLEAN'}
                or record.authority is not authority
                or record.owner_thread_id != get_ident()
                or self._active_operation_leases.get(id(record.operation_lease))
                is not record.operation_lease
                or self._active_operation_authorities.get(
                    id(record.operation_lease)
                )
                is not authority
            ):
                raise TypeError('PostgreSQL emergency cleanup disposition changed')
            if record.phase == 'REVOKED_UNCERTAIN':
                self._drain_exact_emergency_record_locked(record)
                record = record._replace(
                    phase='REVOKED_CLEAN',
                    revision=record.revision + 1,
                    cleanup_ticket=None,
                    cleanup_generation=None,
                )
                self._emergency_cleanup_records[id(capability)] = record
            if poison:
                self._force_fail_stop_locked()
            self._condition.notify_all()

    def _emergency_cleanup_is_clean(
        self,
        capability: object,
        *,
        authority: object,
    ) -> bool:
        with self._condition:
            try:
                record = self._require_emergency_cleanup_capability_any_phase(
                    capability,
                    authority=authority,
                )
            except BaseException:
                return False
            return record.phase == 'REVOKED_CLEAN'

    def _require_emergency_cleanup_capability(
        self,
        capability: object,
        *,
        authority: object | None = None,
    ) -> None:
        record = self._emergency_cleanup_record_or_none(capability)
        if (
            type(capability) is not _PostgresEmergencyCleanupCapability
            or record is None
            or (authority is not None and record.authority is not authority)
            or record.owner_thread_id != get_ident()
            or type(record.operation_lease) is not _PostgresRuntimeHealthLease
            or record.operation_lease._seal
            is not _POSTGRES_RUNTIME_HEALTH_LEASE_SEAL
            or record.operation_lease.health is not self
            or not record.operation_lease.active
            or record.operation_lease.epoch != record.operation_epoch
            or record.operation_lease.purpose != record.operation_purpose
            or self._active_operation_leases.get(id(record.operation_lease))
            is not record.operation_lease
            or self._active_operation_authorities.get(id(record.operation_lease))
            is not record.authority
        ):
            raise TypeError('PostgreSQL emergency cleanup capability changed')

    def _require_emergency_cleanup_capability_any_phase(
        self,
        capability: object,
        *,
        authority: object | None = None,
    ) -> _PostgresEmergencyCleanupRecord:
        """Validate an exact still-owned operation during terminal cleanup."""
        record = (
            None
            if type(capability) is not _PostgresEmergencyCleanupCapability
            else self._emergency_cleanup_records.get(id(capability))
        )
        if (
            record is None
            or record.capability is not capability
            or record.phase
            not in {'ACTIVE', 'REVOKED_UNCERTAIN', 'REVOKED_CLEAN'}
            or (authority is not None and record.authority is not authority)
            or record.owner_thread_id != get_ident()
            or record.operation_lease._seal
            is not _POSTGRES_RUNTIME_HEALTH_LEASE_SEAL
            or record.operation_lease.health is not self
            or not record.operation_lease.active
            or self._active_operation_leases.get(id(record.operation_lease))
            is not record.operation_lease
            or self._active_operation_authorities.get(id(record.operation_lease))
            is not record.authority
        ):
            raise TypeError('PostgreSQL emergency cleanup capability changed')
        return record

    def _emergency_cleanup_capability_is_active(
        self,
        capability: object,
    ) -> bool:
        with self._condition:
            return self._emergency_cleanup_record_or_none(capability) is not None

    def _emergency_cleanup_record_or_none(
        self,
        capability: object,
    ) -> _PostgresEmergencyCleanupRecord | None:
        if type(capability) is not _PostgresEmergencyCleanupCapability:
            return None
        record = self._emergency_cleanup_records.get(id(capability))
        if (
            record is None
            or record.capability is not capability
            or record.phase != 'ACTIVE'
        ):
            return None
        return record

    def _emergency_cleanup_record(
        self,
        capability: object,
    ) -> _PostgresEmergencyCleanupRecord:
        record = self._emergency_cleanup_record_or_none(capability)
        if record is None:
            raise TypeError('PostgreSQL emergency cleanup capability changed')
        return record

    def _require_emergency_transition(
        self,
        capability: object,
        *,
        _seal: object,
    ) -> _PostgresEmergencyCleanupRecord:
        if _seal is not _POSTGRES_EMERGENCY_TRANSITION_SEAL:
            raise TypeError('PostgreSQL emergency cleanup transition changed')
        self._require_emergency_cleanup_capability(capability)
        return self._emergency_cleanup_record(capability)

    def _begin_emergency_cleanup_ticket(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        *,
        _seal: object,
    ) -> _PostgresRuntimeExclusiveTicket:
        with self._condition:
            return self._begin_emergency_cleanup_ticket_locked(
                capability,
                _seal=_seal,
            )

    def _begin_emergency_cleanup_ticket_locked(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        *,
        _seal: object,
    ) -> _PostgresRuntimeExclusiveTicket:
        record = self._emergency_cleanup_record(capability)
        self._require_emergency_transition(capability, _seal=_seal)
        if record.cleanup_ticket is not None or record.cleanup_generation is not None:
            raise TypeError('PostgreSQL emergency cleanup transition changed')
        self._exclusive_ticket_sequence += 1
        ticket = _PostgresRuntimeExclusiveTicket(
            sequence=self._exclusive_ticket_sequence,
            owner_thread_id=record.owner_thread_id,
            purpose='cleanup',
        )
        replacement: _PostgresEmergencyCleanupRecord | None = None
        try:
            self._exclusive_tickets.append(ticket)
            self._emergency_cleanup_transition_checkpoint('after_ticket_append')
            self._emergency_cleanup_transition_checkpoint('before_ticket_record')
            replacement = record._replace(
                revision=record.revision + 1,
                cleanup_ticket=ticket,
            )
            self._emergency_cleanup_records[id(capability)] = replacement
            self._emergency_cleanup_transition_checkpoint('after_ticket_record')
            return ticket
        except BaseException:
            self._cancel_exclusive_ticket(ticket)
            if replacement is not None:
                current = self._emergency_cleanup_records.get(id(capability))
                if current is replacement:
                    self._emergency_cleanup_records[id(capability)] = record
            raise

    def _claim_emergency_cleanup_ticket(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        ticket: _PostgresRuntimeExclusiveTicket,
        *,
        _seal: object,
    ) -> _PostgresCleanupOwnerCapability:
        with self._condition:
            return self._claim_emergency_cleanup_ticket_locked(
                capability,
                ticket,
                _seal=_seal,
            )

    def _claim_emergency_cleanup_ticket_locked(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        ticket: _PostgresRuntimeExclusiveTicket,
        *,
        _seal: object,
    ) -> _PostgresCleanupOwnerCapability:
        record = self._require_emergency_transition(capability, _seal=_seal)
        if (
            record.cleanup_ticket is not ticket
            or record.owner_thread_id != ticket.owner_thread_id
            or not self._is_head_exclusive_ticket(ticket)
        ):
            raise TypeError('PostgreSQL emergency cleanup ticket changed')
        generation = self._exclusive_generation_sequence + 1
        owner_capability = _PostgresCleanupOwnerCapability(
            health=self,
            owner_thread_id=record.owner_thread_id,
            generation=generation,
            _seal=_POSTGRES_CLEANUP_OWNER_SEAL,
        )
        owner_state = _PostgresExclusiveOwnerState(
            owner_thread_id=record.owner_thread_id,
            depth=1,
            generation=generation,
            capability=owner_capability,
            operation_lease_id=id(record.operation_lease),
        )
        replacement = record._replace(
            revision=record.revision + 1,
            cleanup_ticket=None,
            cleanup_generation=generation,
        )
        claimed = True
        owner_published = False
        record_published = False
        try:
            self._claim_head_exclusive_ticket(ticket)
            self._emergency_cleanup_transition_checkpoint('after_head_removal')
            self._emergency_cleanup_transition_checkpoint(
                'before_owner_state_publication'
            )
            self._exclusive_generation_sequence = generation
            owner_published = True
            self._exclusive_owner_state = owner_state
            self._emergency_cleanup_transition_checkpoint(
                'after_owner_state_publication'
            )
            record_published = True
            self._emergency_cleanup_records[id(capability)] = replacement
            self._emergency_cleanup_transition_checkpoint('after_claim')
            return owner_capability
        except BaseException:
            if owner_published and self._exclusive_owner_state is owner_state:
                owner_capability.active = False
                self._exclusive_owner_state = None
            if claimed and ticket not in self._exclusive_tickets:
                self._exclusive_tickets.appendleft(ticket)
            current = self._emergency_cleanup_records.get(id(capability))
            if record_published and current is replacement:
                self._emergency_cleanup_records[id(capability)] = record
            self._condition.notify_all()
            raise

    def _rollback_emergency_cleanup_transition(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        ticket: _PostgresRuntimeExclusiveTicket,
        *,
        _seal: object,
    ) -> None:
        with self._condition:
            self._rollback_emergency_cleanup_transition_locked(
                capability,
                ticket,
                _seal=_seal,
            )

    def _rollback_emergency_cleanup_transition_locked(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        ticket: _PostgresRuntimeExclusiveTicket,
        *,
        _seal: object,
    ) -> None:
        record = self._require_emergency_transition(capability, _seal=_seal)
        if (
            record.cleanup_ticket is not ticket
            and record.cleanup_generation is None
            and ticket not in self._exclusive_tickets
        ):
            return
        self._cancel_exclusive_ticket(ticket)
        owner_state = self._exclusive_owner_state
        if (
            record.cleanup_generation is not None
            and owner_state is not None
            and owner_state.generation == record.cleanup_generation
            and owner_state.owner_thread_id == record.owner_thread_id
            and owner_state.operation_lease_id == id(record.operation_lease)
        ):
            owner_state.capability.active = False
            self._exclusive_owner_state = None
        replacement = record._replace(
            revision=record.revision + 1,
            cleanup_ticket=None,
            cleanup_generation=None,
        )
        self._emergency_cleanup_records[id(capability)] = replacement
        self._condition.notify_all()

    def _emergency_cleanup_transition_checkpoint(self, _stage: str) -> None:
        """Test seam inside exact registry/FIFO transitions."""

    def _enqueue_exclusive_ticket(
        self,
        *,
        thread_id: int,
        purpose: Literal['cleanup', 'poison'],
    ) -> _PostgresRuntimeExclusiveTicket:
        self._exclusive_ticket_sequence += 1
        ticket = _PostgresRuntimeExclusiveTicket(
            sequence=self._exclusive_ticket_sequence,
            owner_thread_id=thread_id,
            purpose=purpose,
        )
        self._exclusive_tickets.append(ticket)
        return ticket

    def _is_head_exclusive_ticket(
        self,
        ticket: _PostgresRuntimeExclusiveTicket,
    ) -> bool:
        return bool(self._exclusive_tickets) and self._exclusive_tickets[0] is ticket

    def _claim_head_exclusive_ticket(
        self,
        ticket: _PostgresRuntimeExclusiveTicket,
    ) -> None:
        if not self._is_head_exclusive_ticket(ticket):
            raise RuntimeError('PostgreSQL cleanup authority ticket changed')
        self._exclusive_tickets.popleft()

    def _cancel_exclusive_ticket(
        self,
        ticket: _PostgresRuntimeExclusiveTicket,
    ) -> None:
        queued_ticket = next(
            (
                queued
                for queued in self._exclusive_tickets
                if queued is ticket
            ),
            None,
        )
        if queued_ticket is None:
            return
        self._exclusive_tickets.remove(queued_ticket)
        self._condition.notify_all()


@dataclass(frozen=True, slots=True, repr=False)
class DatabaseConnectionPolicy:
    """One resolved application/dedicated Engine construction policy."""

    engine_options: Mapping[str, object] = field(
        default_factory=dict,
        repr=False,
    )
    initialize_engine: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        options = dict(self.engine_options)
        if 'poolclass' in options or 'creator' in options:
            raise DatabaseConfigurationError()
        if self.initialize_engine is not None:
            raise DatabaseConfigurationError()
        object.__setattr__(
            self,
            'engine_options',
            _freeze_policy_mapping(options),
        )


@dataclass(frozen=True, slots=True, repr=False)
class _IssuedDedicatedPostgresEngine:
    engine: Engine = field(repr=False)
    policy_capability_id: str
    bootstrap: TrustedPostgresEngineBootstrap = field(repr=False)
    runtime_health: TrustedPostgresRuntimeHealth = field(repr=False)
    _seal: object = field(repr=False, compare=False)


class _TrustedApplicationCheckoutRegistry:
    """Pool-local ownership tracker; callbacks never acquire health locks."""

    __slots__ = (
        '_callback_depths',
        '_captured',
        '_close_owner_thread_id',
        '_condition',
        '_engine',
        '_listeners',
        '_lock',
        '_pending',
        '_remaining_listeners',
        '_runtime_health',
        '_state',
    )

    def __init__(
        self,
        engine: Engine,
        *,
        runtime_health: TrustedPostgresRuntimeHealth,
    ) -> None:
        self._engine = engine
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._runtime_health = runtime_health
        self._callback_depths: dict[int, int] = {}
        self._close_owner_thread_id: int | None = None
        self._pending: dict[int, _PostgresEmergencyCleanupCapability] = {}
        self._captured: dict[
            int,
            tuple[object, object, int],
        ] = {}
        self._state: Literal['open', 'closing', 'closed'] = 'open'

        def on_checkout(
            _dbapi_connection: object,
            connection_record: object,
            connection_proxy: object,
        ) -> None:
            thread_id = get_ident()
            with self._condition:
                self._callback_depths[thread_id] = (
                    self._callback_depths.get(thread_id, 0) + 1
                )
            try:
                with self._condition:
                    if self._state == 'closed':
                        return
                    capability = self._pending.get(thread_id)
                    if capability is not None:
                        self._captured[id(capability)] = (
                            connection_proxy,
                            connection_record,
                            thread_id,
                        )
                    self._condition.notify_all()
            finally:
                with self._condition:
                    depth = self._callback_depths.get(thread_id, 0)
                    if depth <= 1:
                        self._callback_depths.pop(thread_id, None)
                    else:
                        self._callback_depths[thread_id] = depth - 1
                    self._condition.notify_all()

        def on_return(
            _dbapi_connection: object,
            connection_record: object,
            *_args: object,
        ) -> None:
            thread_id = get_ident()
            with self._condition:
                self._callback_depths[thread_id] = (
                    self._callback_depths.get(thread_id, 0) + 1
                )
            try:
                with self._condition:
                    for capability_id, captured in tuple(self._captured.items()):
                        if captured[1] is connection_record:
                            self._captured.pop(capability_id, None)
                    self._condition.notify_all()
            finally:
                with self._condition:
                    depth = self._callback_depths.get(thread_id, 0)
                    if depth <= 1:
                        self._callback_depths.pop(thread_id, None)
                    else:
                        self._callback_depths[thread_id] = depth - 1
                    self._condition.notify_all()

        listeners = (
            (engine.pool, 'checkout', on_checkout),
            (engine.pool, 'checkin', on_return),
            (engine.pool, 'invalidate', on_return),
        )
        self._listeners = listeners
        self._remaining_listeners = list(listeners)
        installed: list[tuple[object, str, object]] = []
        try:
            for target, identifier, callback in listeners:
                event.listen(target, identifier, callback)
                installed.append((target, identifier, callback))
        except BaseException:
            with self._condition:
                self._state = 'closing'
            confirmed = []
            for listener in listeners:
                target, identifier, callback = listener
                try:
                    is_installed = event.contains(target, identifier, callback)
                except BaseException:
                    is_installed = listener in installed
                if is_installed:
                    confirmed.append(listener)
            removal_failed = self._remove_listeners(tuple(reversed(confirmed)))
            with self._condition:
                self._state = 'closed'
                self._condition.notify_all()
            if removal_failed:
                self._fail_stop_runtime()
            raise

    def arm(self, capability: _PostgresEmergencyCleanupCapability) -> None:
        with self._condition:
            thread_id = get_ident()
            if self._state != 'open' or thread_id in self._pending:
                raise TypeError('PostgreSQL application checkout changed')
            self._pending[thread_id] = capability
            self._condition.notify_all()

    def connect(
        self,
        engine: Engine,
        capability: _PostgresEmergencyCleanupCapability,
    ):
        """Install pending responsibility inside the same protected try/finally."""
        try:
            self.arm(capability)
            return engine.connect()
        finally:
            self.disarm(capability)

    def disarm(self, capability: _PostgresEmergencyCleanupCapability) -> None:
        with self._condition:
            thread_id = get_ident()
            if self._pending.get(thread_id) is capability:
                self._pending.pop(thread_id, None)
            self._condition.notify_all()

    def transfer(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        connection: object,
    ) -> None:
        with self._condition:
            captured = self._captured.get(id(capability))
            if (
                self._state == 'closed'
                or captured is None
                or not hasattr(connection, 'connection')
                or connection.connection is not captured[0]
            ):
                raise TypeError('PostgreSQL application checkout changed')
            self._captured.pop(id(capability), None)
            self._condition.notify_all()

    def drain(self, capability: _PostgresEmergencyCleanupCapability) -> bool:
        with self._condition:
            captured = self._captured.pop(id(capability), None)
            for thread_id, pending in tuple(self._pending.items()):
                if pending is capability:
                    self._pending.pop(thread_id, None)
            self._condition.notify_all()
        if captured is None:
            return False
        proxy = captured[0]
        failed = False
        try:
            proxy.invalidate()
        except BaseException:
            failed = True
        try:
            proxy.close()
        except BaseException:
            failed = True
        return failed

    def revoke(self) -> bool:
        thread_id = get_ident()
        with self._condition:
            while self._close_owner_thread_id is not None:
                if self._close_owner_thread_id == thread_id:
                    self._state = 'closing'
                    self._fail_stop_runtime()
                    return False
                try:
                    self._condition.wait()
                except BaseException:
                    self._condition.notify_all()
                    self._fail_stop_runtime()
                    raise
            if self._state == 'closed':
                return True
            self._state = 'closing'
            self._close_owner_thread_id = thread_id
            deferred = (
                self._pending.get(thread_id) is not None
                or any(
                    captured[2] == thread_id
                    for captured in self._captured.values()
                )
                or self._callback_depths.get(thread_id, 0) > 0
            )
            if not deferred:
                try:
                    while self._pending or self._captured or self._callback_depths:
                        self._condition.wait()
                except BaseException:
                    self._close_owner_thread_id = None
                    self._condition.notify_all()
                    self._fail_stop_runtime()
                    raise
        if deferred:
            with self._condition:
                self._close_owner_thread_id = None
                self._condition.notify_all()
                self._fail_stop_runtime()
            return False
        removal_error: BaseException | None = None
        removal_uncertain = False
        for listener in tuple(reversed(self._remaining_listeners)):
            target, identifier, callback = listener
            try:
                event.remove(target, identifier, callback)
            except BaseException as exc:
                removal_uncertain = True
                try:
                    removed = not event.contains(target, identifier, callback)
                except BaseException:
                    removed = False
                if removed:
                    self._remaining_listeners.remove(listener)
                removal_error = exc
                break
            else:
                self._remaining_listeners.remove(listener)
        with self._condition:
            if not self._remaining_listeners:
                self._state = 'closed'
            self._close_owner_thread_id = None
            self._condition.notify_all()
        if removal_uncertain:
            self._fail_stop_runtime()
        if removal_error is not None:
            raise removal_error
        return self._state == 'closed'

    def _remove_listeners(
        self,
        listeners: tuple[tuple[object, str, object], ...],
    ) -> bool:
        failed = False
        for target, identifier, callback in listeners:
            removed = False
            for _attempt in range(2):
                try:
                    event.remove(target, identifier, callback)
                    removed = True
                    break
                except BaseException:
                    failed = True
                    try:
                        if not event.contains(target, identifier, callback):
                            removed = True
                            break
                    except BaseException:
                        pass
            if not removed:
                failed = True
        return failed

    def _fail_stop_runtime(self) -> None:
        with self._runtime_health._condition:
            self._runtime_health._force_fail_stop_locked()
            self._runtime_health._condition.notify_all()


class TrustedPostgresEngineBootstrap:
    """Bootstrap-minted factory for per-request advisory transport."""

    __slots__ = (
        '_application_engine',
        '_checkout_registry',
        '_dedicated_factory',
        '_policy_capability_id',
        '_revoked',
        '_runtime_health',
        '_seal',
        '_state_lock',
    )

    def __init__(
        self,
        *,
        application_engine: Engine,
        dedicated_factory: Callable[[], Engine],
        policy_capability_id: str,
        runtime_health: TrustedPostgresRuntimeHealth,
        _seal: object,
    ) -> None:
        if (
            _seal is not _POSTGRES_BOOTSTRAP_SEAL
            or not isinstance(application_engine, Engine)
            or application_engine.dialect.name != 'postgresql'
            or not callable(dedicated_factory)
            or len(policy_capability_id) != 64
            or type(runtime_health) is not TrustedPostgresRuntimeHealth
        ):
            raise TypeError('trusted PostgreSQL Engine bootstrap is unavailable')
        self._application_engine = application_engine
        self._dedicated_factory = dedicated_factory
        self._policy_capability_id = policy_capability_id
        self._revoked = False
        self._runtime_health = runtime_health
        self._seal = _seal
        self._state_lock = RLock()

        self._checkout_registry = _TrustedApplicationCheckoutRegistry(
            application_engine,
            runtime_health=runtime_health,
        )

    def _connect_registered_application(
        self,
        application_engine: Engine,
        capability: _PostgresEmergencyCleanupCapability,
    ):
        with self._state_lock:
            if (
                self._revoked
                or self._seal is not _POSTGRES_BOOTSTRAP_SEAL
                or application_engine is not self._application_engine
            ):
                raise TypeError('PostgreSQL Engine bootstrap authority changed')
        self._runtime_health._require_emergency_cleanup_capability(capability)
        return self._checkout_registry.connect(application_engine, capability)

    def _release_registered_application_checkout(
        self,
        capability: _PostgresEmergencyCleanupCapability,
        connection: object,
    ) -> None:
        self._runtime_health._require_emergency_cleanup_capability(capability)
        self._checkout_registry.transfer(capability, connection=connection)

    def _drain_registered_application_checkout(
        self,
        capability: _PostgresEmergencyCleanupCapability,
    ) -> bool:
        """Physically discard a checkout whose Connection was never exposed."""
        try:
            self._runtime_health._require_emergency_cleanup_capability_any_phase(
                capability
            )
        except BaseException:
            return True
        return self._checkout_registry.drain(capability)

    def _issue(self, application_engine: Engine) -> _IssuedDedicatedPostgresEngine:
        with self._runtime_health._operation('bootstrap_issue') as health_lease:
            with self._runtime_health._guard(health_lease), self._state_lock:
                if (
                    self._revoked
                    or self._seal is not _POSTGRES_BOOTSTRAP_SEAL
                    or application_engine is not self._application_engine
                ):
                    raise TypeError('PostgreSQL Engine bootstrap authority changed')
                dedicated_factory = self._dedicated_factory
            dedicated_engine = dedicated_factory()
            try:
                with self._runtime_health._guard(health_lease), self._state_lock:
                    accepted = (
                        not self._revoked
                        and self._seal is _POSTGRES_BOOTSTRAP_SEAL
                        and application_engine is self._application_engine
                        and isinstance(dedicated_engine, Engine)
                        and dedicated_engine is not application_engine
                        and dedicated_engine.dialect.name == 'postgresql'
                        and type(dedicated_engine.pool) is NullPool
                    )
                    if accepted:
                        return _IssuedDedicatedPostgresEngine(
                            engine=dedicated_engine,
                            policy_capability_id=self._policy_capability_id,
                            bootstrap=self,
                            runtime_health=self._runtime_health,
                            _seal=_POSTGRES_BOOTSTRAP_SEAL,
                        )
            except BaseException:
                self._dispose_rejected_engine(dedicated_engine)
                raise
            self._dispose_rejected_engine(dedicated_engine)
            raise TypeError('trusted dedicated PostgreSQL Engine is unavailable')

    def _dispose_rejected_engine(self, engine: object) -> None:
        if not isinstance(engine, Engine):
            return
        try:
            engine.dispose()
        except BaseException:
            self._runtime_health._poison()

    @property
    def runtime_health_snapshot(self) -> PostgresRuntimeHealthSnapshot:
        return self._runtime_health.snapshot

    def _runtime_effect_authority(
        self,
        application_engine: Engine,
    ) -> TrustedPostgresRuntimeHealth:
        with self._state_lock:
            if (
                self._revoked
                or self._seal is not _POSTGRES_BOOTSTRAP_SEAL
                or application_engine is not self._application_engine
            ):
                raise TypeError('PostgreSQL Engine bootstrap authority changed')
            return self._runtime_health

    def _revoke(self) -> bool:
        with self._state_lock:
            self._revoked = True
        return self._checkout_registry.revoke()

def _is_database_availability_error(error: Exception) -> bool:
    if isinstance(error, (ModuleNotFoundError, ImportError, OSError)):
        return True
    if isinstance(error, DBAPIError) and bool(error.connection_invalidated):
        return True
    return isinstance(
        error,
        (
            OperationalError,
            InterfaceError,
            SQLAlchemyTimeoutError,
            DisconnectionError,
        ),
    )


@dataclass(slots=True, repr=False)
class DatabaseRuntime:
    engine: Engine
    session_factory: sessionmaker[Session]
    rag_postgres_bootstrap: TrustedPostgresEngineBootstrap | None = field(
        default=None,
        repr=False,
    )
    _dispose_condition: Condition = field(init=False, repr=False)
    _dispose_lock: object = field(default_factory=RLock, init=False, repr=False)
    _dispose_owner_thread_id: int | None = field(default=None, init=False, repr=False)
    _dispose_state: Literal[
        'OPEN',
        'IN_PROGRESS',
        'RETRY_REQUIRED',
        'DONE',
    ] = field(default='OPEN', init=False)
    _engine_disposed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._dispose_condition = Condition(self._dispose_lock)  # type: ignore[arg-type]

    def dispose(self) -> None:
        thread_id = get_ident()
        with self._dispose_condition:
            while self._dispose_state == 'IN_PROGRESS':
                if self._dispose_owner_thread_id == thread_id:
                    self._dispose_state = 'RETRY_REQUIRED'
                    self._dispose_condition.notify_all()
                    return
                self._dispose_condition.wait()
            if self._dispose_state == 'DONE':
                return
            self._dispose_state = 'IN_PROGRESS'
            self._dispose_owner_thread_id = thread_id
        try:
            if (
                self.rag_postgres_bootstrap is not None
                and not self.rag_postgres_bootstrap._revoke()
            ):
                with self._dispose_condition:
                    self._dispose_state = 'RETRY_REQUIRED'
                return
        except BaseException:
            with self._dispose_condition:
                self._dispose_state = 'RETRY_REQUIRED'
            raise
        finally:
            with self._dispose_condition:
                if self._dispose_state != 'IN_PROGRESS':
                    self._dispose_owner_thread_id = None
                    self._dispose_condition.notify_all()
        with self._dispose_condition:
            if self._engine_disposed:
                self._dispose_state = 'DONE'
                self._dispose_owner_thread_id = None
                self._dispose_condition.notify_all()
                return
        cleanup_failure: Exception | None = None
        try:
            self.engine.dispose()
        except Exception as error:
            cleanup_failure = error
        finally:
            with self._dispose_condition:
                self._engine_disposed = True
                self._dispose_state = 'DONE'
                self._dispose_owner_thread_id = None
                self._dispose_condition.notify_all()
        if cleanup_failure is None:
            return
        if _is_database_availability_error(cleanup_failure):
            raise DatabaseInitializationError()
        raise cleanup_failure


def initialize_database_runtime(
    database_url: str,
    *,
    connection_policy: DatabaseConnectionPolicy | None = None,
) -> DatabaseRuntime:
    configuration_failure = False
    try:
        parsed_url = make_url(database_url)
    except ArgumentError:
        configuration_failure = True
    if configuration_failure:
        raise DatabaseConfigurationError()

    configuration_failure = False
    engine_failure: Exception | None = None
    policy = connection_policy or DatabaseConnectionPolicy()
    engine_options = {'pool_pre_ping': True, **dict(policy.engine_options)}
    try:
        engine = create_engine(
            database_url,
            **_engine_options_for_create(engine_options),
        )
    except (NoSuchModuleError, ArgumentError):
        configuration_failure = True
    except Exception as error:
        engine_failure = error
    if configuration_failure:
        raise DatabaseConfigurationError()
    if engine_failure is not None:
        if _is_database_availability_error(engine_failure):
            raise DatabaseInitializationError()
        raise engine_failure

    session_factory: sessionmaker[Session] | None = None
    session_factory_failure: Exception | None = None
    try:
        session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=True,
        )
    except Exception as error:
        session_factory_failure = error

    if session_factory_failure is not None:
        cleanup_failure: Exception | None = None
        try:
            engine.dispose()
        except Exception as error:
            cleanup_failure = error
        if cleanup_failure is not None:
            if _is_database_availability_error(cleanup_failure):
                raise DatabaseInitializationError()
            raise cleanup_failure
        raise session_factory_failure

    assert session_factory is not None
    postgres_bootstrap: TrustedPostgresEngineBootstrap | None = None
    if parsed_url.get_backend_name() == 'postgresql':
        policy_capability_id = secrets.token_hex(32)
        runtime_health = TrustedPostgresRuntimeHealth(
            _seal=_POSTGRES_RUNTIME_HEALTH_SEAL
        )

        def dedicated_factory() -> Engine:
            dedicated_options = _engine_options_for_create(engine_options)
            dedicated_options['poolclass'] = NullPool
            return create_engine(database_url, **dedicated_options)

        try:
            postgres_bootstrap = TrustedPostgresEngineBootstrap(
                application_engine=engine,
                dedicated_factory=dedicated_factory,
                policy_capability_id=policy_capability_id,
                runtime_health=runtime_health,
                _seal=_POSTGRES_BOOTSTRAP_SEAL,
            )
        except BaseException:
            with suppress(BaseException):
                engine.dispose()
            raise
    return DatabaseRuntime(
        engine=engine,
        session_factory=session_factory,
        rag_postgres_bootstrap=postgres_bootstrap,
    )


def _engine_options_for_create(
    engine_options: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: _thaw_policy_value(value)
        for key, value in engine_options.items()
    }


def _freeze_policy_mapping(
    value: Mapping[object, object],
) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise DatabaseConfigurationError()
        frozen[key] = _freeze_policy_value(item)
    return MappingProxyType(frozen)


def _freeze_policy_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if isinstance(value, Mapping):
        return _freeze_policy_mapping(value)
    if type(value) is tuple:
        return tuple(_freeze_policy_value(item) for item in value)
    raise DatabaseConfigurationError()


def _thaw_policy_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_policy_value(item)
            for key, item in value.items()
        }
    if type(value) is tuple:
        return tuple(_thaw_policy_value(item) for item in value)
    return value
