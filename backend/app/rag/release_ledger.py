from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TypeAlias
from uuid import UUID

from sqlalchemy import Connection, Table, insert, select, update
from sqlalchemy.sql.dml import Insert, Update
from sqlalchemy.sql.elements import BindParameter

from backend.app.agent_runtime.durable_file_authority import (
    DurableFileAuthority,
    DurableFileAuthorityError,
)
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_safety_identity import (
    rag_identity_hmac,
    require_lower_hmac,
)
from backend.app.rag.release_authority import (
    RagReleaseAuthority,
    RagReleaseAuthorityError,
    RagReleaseSnapshot,
    ValidationDatabaseIdentity,
)
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables

AffectedRow: TypeAlias = tuple[str, str]
ObservedRow: TypeAlias = tuple[str, str, str]

_ROW_IDENTITY_REGISTRY: Mapping[str, tuple[str, tuple[str, ...]]] = {
    'authorization': (
        'rag-release-row-identity:authorization:v1',
        ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac'),
    ),
    'case': (
        'rag-release-row-identity:case:v1',
        ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac', 'case_id_hmac'),
    ),
    'dispatch': (
        'rag-release-row-identity:dispatch:v1',
        (
            'ledger_uuid',
            'ledger_epoch',
            'approval_id_hmac',
            'case_id_hmac',
            'component',
        ),
    ),
    'agent_run': ('rag-release-row-identity:agent-run:v1', ('agent_run_id',)),
    'cost_component': (
        'rag-release-row-identity:cost-component:v1',
        ('agent_run_id', 'component'),
    ),
    'provider_safety_authority': (
        'rag-release-row-identity:provider-safety-authority:v1',
        ('authority_uuid',),
    ),
    'provider_readiness': (
        'rag-release-row-identity:provider-readiness:v1',
        (
            'component',
            'provider_bytes',
            'model_bytes',
            'reasoning_or_config_identity_bytes',
        ),
    ),
    'quality_report': (
        'rag-release-row-identity:quality-report:v1',
        ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac'),
    ),
    'release_ledger': (
        'rag-release-row-identity:release-ledger:v1',
        ('ledger_uuid', 'ledger_epoch'),
    ),
    'release_transition': (
        'rag-release-row-identity:release-transition:v1',
        ('ledger_uuid', 'ledger_epoch', 'to_generation'),
    ),
}


@dataclass(frozen=True, slots=True)
class ReleaseRowPrimaryKey:
    row_kind: str
    primary_key: Mapping[str, object]


_MUTATION_SET_SEAL = object()


@dataclass(frozen=True, slots=True)
class _ReleaseMutationPlan:
    statement: object
    row: ReleaseRowPrimaryKey


class RagReleaseMutationSet:
    """Plan row mutations, then capture them only under the authority barrier."""

    __slots__ = (
        '_after_snapshots',
        '_before_snapshots',
        '_connection',
        '_executed',
        '_observation_rows',
        '_observation_snapshots',
        '_plans',
        '_rows',
        '_seal',
        '_transaction',
    )

    def __init__(self, connection: Connection, *, seal: object) -> None:
        if seal is not _MUTATION_SET_SEAL:
            raise RagReleaseLedgerError('release mutation set is invalid')
        self._connection = connection
        self._rows: list[ReleaseRowPrimaryKey] = []
        self._plans: list[_ReleaseMutationPlan] = []
        self._before_snapshots: list[dict[str, object] | None] = []
        self._after_snapshots: list[dict[str, object]] = []
        self._observation_rows: list[ReleaseRowPrimaryKey] = []
        self._observation_snapshots: list[dict[str, object]] = []
        self._transaction = None
        self._executed = False
        self._seal = seal

    @staticmethod
    def _table(row_kind: str) -> tuple[Table, Mapping[str, str]]:
        if row_kind in {'authorization', 'case', 'dispatch', 'quality_report'}:
            release = release_tables(build_rag_release_metadata())
        if row_kind == 'authorization':
            return release.authorizations, {}
        if row_kind == 'case':
            return release.cases, {}
        if row_kind == 'dispatch':
            return release.dispatches, {}
        if row_kind == 'quality_report':
            return release.quality_reports, {}
        if row_kind == 'agent_run':
            from backend.app.models.agent_runs import AgentRun

            # The release ledger's canonical identity deliberately stays
            # ``agent_run_id`` while the application table's physical primary
            # key is ``agent_runs.id``.  Never leak the physical spelling into
            # the signed row-identity domain.
            return AgentRun.__table__, {'agent_run_id': 'id'}
        if row_kind == 'cost_component':
            from backend.app.models.rag_runtime import AgentRunCostComponent

            return AgentRunCostComponent.__table__, {}
        if row_kind == 'provider_safety_authority':
            from backend.app.models.rag_runtime import RagProviderSafetyAuthority

            return RagProviderSafetyAuthority.__table__, {}
        if row_kind == 'provider_readiness':
            from backend.app.models.rag_runtime import RagProviderReadiness

            return RagProviderReadiness.__table__, {
                'provider_bytes': 'provider',
                'model_bytes': 'model',
                'reasoning_or_config_identity_bytes': ('reasoning_or_config_identity'),
            }
        raise RagReleaseLedgerError('release mutation row kind is invalid')

    @classmethod
    def _snapshot(
        cls,
        connection: Connection,
        row: ReleaseRowPrimaryKey,
        *,
        for_update: bool = False,
    ) -> dict[str, object] | None:
        table, aliases = cls._table(row.row_kind)
        predicates = []
        for key, value in row.primary_key.items():
            column_name = aliases.get(key, key)
            if column_name not in table.c:
                raise RagReleaseLedgerError('release row primary key is invalid')
            predicates.append(table.c[column_name] == value)
        statement = select(table).where(*predicates)
        if for_update and connection.dialect.name == 'postgresql':
            statement = statement.with_for_update()
        result = connection.execute(statement).mappings().all()
        if len(result) > 1:
            raise RagReleaseLedgerError('release row identity is not unique')
        # SQLAlchemy may expose schema labels as quoted_name (a str subclass).
        # Normalize only these trusted table labels at the database boundary;
        # never coerce caller keys or any stored scalar/JSON values.
        return (
            None
            if not result
            else {str(column.name): result[0][column.name] for column in table.c}
        )

    def plan(self, statement: object, row: ReleaseRowPrimaryKey) -> None:
        if type(row) is not ReleaseRowPrimaryKey:
            raise RagReleaseLedgerError('release row primary key is invalid')
        if self._executed or self._transaction is not None:
            raise RagReleaseLedgerError('release mutation plan was already executed')
        stored = ReleaseRowPrimaryKey(row.row_kind, dict(row.primary_key))
        if any(plan.row == stored for plan in self._plans) or stored in (
            self._observation_rows
        ):
            raise RagReleaseLedgerError('release row mutated more than once')
        self._plans.append(_ReleaseMutationPlan(statement, stored))

    def observe(self, row: ReleaseRowPrimaryKey) -> None:
        """Plan a locked, read-only peer snapshot for this release barrier."""
        if (
            type(row) is not ReleaseRowPrimaryKey
            or self._executed
            or self._transaction is not None
        ):
            raise RagReleaseLedgerError('release observation is invalid')
        stored = ReleaseRowPrimaryKey(row.row_kind, dict(row.primary_key))
        if (
            stored in self._observation_rows
            or any(plan.row == stored for plan in self._plans)
            or stored.row_kind in {'release_ledger', 'release_transition'}
        ):
            raise RagReleaseLedgerError('release observation is invalid')
        # Validate both the row kind and the exact logical primary-key spelling
        # before it can enter a signed observation set.
        release_row_identity_hmac(
            stored.row_kind,
            stored.primary_key,
            identity_secret=b'0' * 32,
        )
        self._observation_rows.append(stored)

    def _capture_provider_incident(self, evidence: object) -> None:
        from backend.app.agent_runtime.rag_provider_safety import (
            _RELEASE_INCIDENT_PLAN_SEAL,
            _AppliedReleaseProviderIncident,
        )

        if (
            type(evidence) is not _AppliedReleaseProviderIncident
            or evidence._seal is not _RELEASE_INCIDENT_PLAN_SEAL
            or self._executed
        ):
            raise RagReleaseLedgerError('provider incident evidence is invalid')
        authority_uuid = str(evidence.new_body['authority_uuid'])
        readiness = evidence.readiness_after
        rows = (
            ReleaseRowPrimaryKey(
                'provider_safety_authority', {'authority_uuid': authority_uuid}
            ),
            ReleaseRowPrimaryKey(
                'provider_readiness',
                {
                    'component': readiness['component'],
                    'provider_bytes': readiness['provider'],
                    'model_bytes': readiness['model'],
                    'reasoning_or_config_identity_bytes': readiness[
                        'reasoning_or_config_identity'
                    ],
                },
            ),
        )
        if any(plan.row in rows for plan in self._plans):
            raise RagReleaseLedgerError('provider incident rows cannot be SQL planned')
        self._rows.extend(rows)
        self._before_snapshots.extend(
            [dict(evidence.authority_before), dict(evidence.readiness_before)]
        )
        self._after_snapshots.extend(
            [dict(evidence.authority_after), dict(evidence.readiness_after)]
        )

    def _capture_observations_under_barrier(
        self,
        connection: Connection,
        *,
        authority: RagReleaseAuthority,
        barrier_guard: object,
    ) -> None:
        authority._assert_barrier_guard(barrier_guard, connection)
        if connection is not self._connection or self._executed:
            raise RagReleaseLedgerError('release mutation plan is invalid')
        transaction = connection.get_transaction()
        if transaction is None:
            raise RagReleaseLedgerError('release mutation transaction is unavailable')
        if self._transaction is not None:
            if self._transaction is not transaction:
                raise RagReleaseLedgerError('release observation transaction changed')
            return
        for row in self._observation_rows:
            snapshot = self._snapshot(connection, row, for_update=True)
            if snapshot is None:
                raise RagReleaseLedgerError('release observation row is missing')
            self._observation_snapshots.append(snapshot)
        self._transaction = transaction

    def _execute_under_barrier(
        self,
        connection: Connection,
        *,
        authority: RagReleaseAuthority,
        barrier_guard: object,
    ) -> None:
        self._capture_observations_under_barrier(
            connection, authority=authority, barrier_guard=barrier_guard
        )
        if not self._plans:
            raise RagReleaseLedgerError('release mutation plan is empty')
        transaction = self._transaction
        for plan in self._plans:
            before = self._snapshot(connection, plan.row, for_update=True)
            result = connection.execute(plan.statement)  # type: ignore[call-overload]
            after = self._snapshot(connection, plan.row)
            ignored = {'updated_at'}
            if (
                result.rowcount != 1
                or after is None
                or after == before
                or (
                    plan.row.row_kind == 'agent_run'
                    and before is not None
                    and before['started_at'] != after['started_at']
                )
                or (
                    before is not None
                    and not _meaningful_changed(
                        before, after, ignored=frozenset(ignored)
                    )
                )
            ):
                raise RagReleaseLedgerError('release row mutation was not exact')
            if connection.get_transaction() is not transaction:
                raise RagReleaseLedgerError('release mutation transaction changed')
            self._rows.append(plan.row)
            self._before_snapshots.append(before)
            self._after_snapshots.append(after)
        transaction = self._connection.get_transaction()
        if transaction is None:
            raise RagReleaseLedgerError('release mutation transaction changed')
        self._transaction = transaction
        self._executed = True

    def _preflight_runtime_mutations(
        self,
        payload: Mapping[str, object],
        *,
        identity_secret: bytes,
        approved_case_claim: object = None,
        barrier_guard: object = None,
    ) -> None:
        """Validate literal runtime before/after images before *any* write/incident.

        Runtime inserts supply every column explicitly. This removes implicit
        defaults, expression evaluation and server clocks from the signed image.
        Actual SQL after-images are checked against the same HMAC after execution.
        """
        projected = []
        for plan in self._plans:
            if plan.row.row_kind not in {'agent_run', 'cost_component'}:
                continue
            table, _aliases = self._table(plan.row.row_kind)
            statement = plan.statement
            if (
                not isinstance(statement, (Insert, Update))
                or statement.table.name != table.name
            ):
                raise RagReleaseLedgerError('runtime mutation statement is invalid')
            values = {}
            for field, binding in (statement._values or {}).items():
                name = str(field)
                if (
                    name not in table.c
                    or not isinstance(binding, BindParameter)
                    or binding.callable
                ):
                    raise RagReleaseLedgerError(
                        'runtime mutation must use explicit values'
                    )
                values[name] = binding.value
            before = self._snapshot(self._connection, plan.row, for_update=True)
            if (before is None) != isinstance(statement, Insert):
                raise RagReleaseLedgerError('runtime mutation before-image is invalid')
            after = {**(before or {}), **values}
            projected.append((plan.row, before, after))
        self._assert_approved_case_claim(
            payload,
            [(row.row_kind, after) for row, _, after in projected],
            approved_case_claim=approved_case_claim,
            identity_secret=identity_secret,
            barrier_guard=barrier_guard,
        )
        costs = [
            after
            for row, _before, after in projected
            if row.row_kind == 'cost_component'
        ] + [
            snapshot
            for row, snapshot in zip(
                self._observation_rows, self._observation_snapshots, strict=True
            )
            if row.row_kind == 'cost_component'
        ]
        for row, before, after in projected:
            _assert_runtime_column_delta(payload, row.row_kind, before, after, costs)
            self._assert_runtime_mutation_hmac(
                payload, row, before, after, identity_secret=identity_secret
            )

    def _assert_approved_case_claim(
        self,
        payload,
        runtime_rows,
        *,
        approved_case_claim,
        identity_secret,
        barrier_guard=None,
    ):
        if payload['transition_kind'] != 'case_claim':
            if approved_case_claim is not None:
                raise RagReleaseLedgerError('case claim projection kind is invalid')
            return
        from backend.app.rag.release_review import validate_case_claim_projection

        if self._executed:
            case_rows = [
                after
                for row, after in zip(self._rows, self._after_snapshots, strict=True)
                if row.row_kind == 'case'
            ]
        else:
            case_rows = []
            for plan in self._plans:
                if plan.row.row_kind != 'case':
                    continue
                if not isinstance(plan.statement, Insert):
                    raise RagReleaseLedgerError('case claim must insert case')
                values = {}
                for field, binding in (plan.statement._values or {}).items():
                    if not isinstance(binding, BindParameter) or binding.callable:
                        raise RagReleaseLedgerError(
                            'case claim must use literal values'
                        )
                    values[str(field)] = binding.value
                case_rows.append(values)
        validate_case_claim_projection(
            approved_case_claim,
            connection=self._connection,
            payload=payload,
            runtime_rows=runtime_rows,
            case_rows=case_rows,
            after_execution=self._executed,
            observations=[
                (row.row_kind, snapshot)
                for row, snapshot in zip(
                    self._observation_rows, self._observation_snapshots, strict=True
                )
            ],
            identity_secret=identity_secret,
            barrier_guard=barrier_guard,
        )

    @staticmethod
    def _assert_runtime_mutation_hmac(payload, row, before, after, *, identity_secret):
        identity = release_row_identity_hmac(
            row.row_kind, row.primary_key, identity_secret=identity_secret
        )
        signed = next(
            (
                item
                for item in payload['affected_rows']
                if item['row_kind'] == row.row_kind
                and item['row_identity_hmac'] == identity
            ),
            None,
        )
        if signed is None or signed.get(
            'row_mutation_hmac'
        ) != release_runtime_mutation_hmac(
            row.row_kind, before, after, identity_secret=identity_secret
        ):
            raise RagReleaseLedgerError(
                'runtime mutation projection differs from payload'
            )

    @property
    def rows(self) -> tuple[ReleaseRowPrimaryKey, ...]:
        return tuple(self._rows)

    @property
    def observation_rows(self) -> tuple[ReleaseRowPrimaryKey, ...]:
        return tuple(self._observation_rows)

    def _captured_rows(self, row_kind: str) -> list[dict[str, object]]:
        return [
            snapshot
            for row, snapshot in (
                *zip(self._rows, self._after_snapshots, strict=True),
                *zip(self._observation_rows, self._observation_snapshots, strict=True),
            )
            if row.row_kind == row_kind
        ]

    def _assert_roster_rows(self, row_kind: str, rows: list[dict[str, object]]) -> None:
        captured = self._captured_rows(row_kind)
        if len(captured) != len(rows) or any(row not in captured for row in rows):
            raise RagReleaseLedgerError(
                f'terminal roster {row_kind} observation differs'
            )

    def assert_current(self, connection: Connection) -> None:
        if (
            connection is not self._connection
            or self._transaction is None
            or connection.get_transaction() is not self._transaction
        ):
            raise RagReleaseLedgerError(
                'same-transaction release mutation set is required'
            )
        for row, expected in zip(self._rows, self._after_snapshots, strict=True):
            if self._snapshot(connection, row) != expected:
                raise RagReleaseLedgerError(
                    'release mutation row changed after capture'
                )
        for row, expected in zip(
            self._observation_rows, self._observation_snapshots, strict=True
        ):
            if self._snapshot(connection, row) != expected:
                raise RagReleaseLedgerError(
                    'observed release peer changed after capture'
                )

    def assert_payload_projection(
        self,
        payload: Mapping[str, object],
        *,
        identity_secret: bytes,
        approved_case_claim: object = None,
        barrier_guard: object = None,
    ) -> None:
        by_kind: dict[str, list[dict[str, object]]] = {}
        before_by_kind: dict[str, list[dict[str, object] | None]] = {}
        mutated_by_kind: dict[str, list[bool]] = {}
        for row, before, snapshot in zip(
            self._rows,
            self._before_snapshots,
            self._after_snapshots,
            strict=True,
        ):
            by_kind.setdefault(row.row_kind, []).append(snapshot)
            before_by_kind.setdefault(row.row_kind, []).append(before)
            mutated_by_kind.setdefault(row.row_kind, []).append(True)
        for row, snapshot in zip(
            self._observation_rows, self._observation_snapshots, strict=True
        ):
            by_kind.setdefault(row.row_kind, []).append(snapshot)
            before_by_kind.setdefault(row.row_kind, []).append(snapshot)
            mutated_by_kind.setdefault(row.row_kind, []).append(False)
        self._assert_observation_projection(payload, identity_secret=identity_secret)
        # Roster peers share the barrier and digest, but only the current case
        # may participate in this transition's mutable lifecycle.
        for row_kind in ('case', 'agent_run', 'cost_component', 'dispatch'):
            indices = []
            for index, item in enumerate(by_kind.get(row_kind, [])):
                if row_kind in {'case', 'dispatch'}:
                    current = item['case_id_hmac'] == payload['case_id_hmac']
                    if row_kind == 'dispatch':
                        current = current and item['component'] == payload['component']
                else:
                    run_id = (
                        item['id'] if row_kind == 'agent_run' else item['agent_run_id']
                    )
                    current = (
                        rag_identity_hmac(
                            {'agent_run_id': run_id},
                            secret=identity_secret,
                            schema_version='rag-runtime-agent-run-id:v1',
                            policy_version='rag-run:v2',
                        )
                        == payload['runtime_agent_run_id_hmac']
                    )
                if current:
                    indices.append(index)
                elif mutated_by_kind[row_kind][index]:
                    raise RagReleaseLedgerError('unrelated roster mutation is invalid')
            for mapping in (by_kind, before_by_kind, mutated_by_kind):
                mapping[row_kind] = [mapping[row_kind][index] for index in indices]
        authorizations = by_kind.get('authorization', [])
        if len(authorizations) != 1:
            raise RagReleaseLedgerError('authorization peer is required')
        authorization = authorizations[0]
        authorization_before = before_by_kind['authorization'][0]
        expected_authorization_before = payload['authorization_state_before']
        if (
            expected_authorization_before is None and authorization_before is not None
        ) or (
            expected_authorization_before is not None
            and (
                authorization_before is None
                or authorization_before['state'] != expected_authorization_before
            )
        ):
            raise RagReleaseLedgerError(
                'authorization before-image differs from payload'
            )
        exact_authorization = {
            'state': payload['authorization_state_after'],
            'case_claim_count': payload['case_claim_count'],
            'embedding_dispatch_count': payload['embedding_dispatch_count'],
            'generation_dispatch_count': payload['generation_dispatch_count'],
            'total_dispatch_count': payload['total_dispatch_count'],
            'execution_process_instance_hmac': (
                payload['execution_process_instance_hmac']
            ),
            'execution_runner_fence_hmac': payload['execution_runner_fence_hmac'],
        }
        if any(
            authorization[key] != value for key, value in exact_authorization.items()
        ):
            raise RagReleaseLedgerError('authorization payload differs from mutation')
        expected_authorization_identity = {
            'ledger_uuid': payload['ledger_uuid'],
            'ledger_epoch': payload['ledger_epoch'],
            'approval_id_hmac': payload['approval_id_hmac'],
            'approval_hmac': payload['approval_hmac'],
            'approved_corpus_snapshot_hmac': payload['approved_corpus_snapshot_hmac'],
            'approved_provider_safety_snapshot_hmac': payload[
                'approved_provider_safety_snapshot_hmac'
            ],
            'validation_database_identity_hmac': payload[
                'validation_database_identity_hmac'
            ],
        }
        if payload['transition_kind'] == 'authorization_bootstrap':
            expected_authorization_identity.update(
                base_generation=payload['from_generation'],
                provider_safety_envelope_digest=payload[
                    'provider_safety_envelope_digest'
                ],
            )
        if any(
            authorization[key] != value
            for key, value in expected_authorization_identity.items()
        ):
            raise RagReleaseLedgerError(
                'authorization immutable identity differs from payload'
            )
        if any(
            Decimal(str(authorization[column])) != Decimal(str(payload[key]))
            for column, key in (
                ('reserved_cost_usd', 'authorization_reserved_cost_usd'),
                ('charged_cost_usd', 'authorization_charged_cost_usd'),
            )
        ):
            raise RagReleaseLedgerError('authorization cost differs from mutation')
        if authorization_before is not None and any(
            authorization_before[key] != authorization[key]
            for key in (
                'ledger_uuid',
                'ledger_epoch',
                'approval_id_hmac',
                'approval_hmac',
                'base_generation',
                'approved_corpus_snapshot_hmac',
                'approved_provider_safety_snapshot_hmac',
                'provider_safety_envelope_digest',
                'validation_database_identity_hmac',
                'manifest_hmac',
                'baseline_hmac',
                'reviewer_roster_hmac',
            )
        ):
            raise RagReleaseLedgerError('authorization immutable fields changed')
        if authorization_before is not None:
            owner_fields = (
                'execution_process_instance_hmac',
                'execution_runner_fence_hmac',
            )
            first_claim = (
                payload['transition_kind'] == 'case_claim'
                and authorization_before['state'] == 'unused'
                and authorization_before['case_claim_count'] == 0
            )
            if first_claim:
                valid_owner = all(
                    authorization_before[key] is None and authorization[key] is not None
                    for key in owner_fields
                )
            else:
                valid_owner = all(
                    authorization_before[key] == authorization[key]
                    for key in owner_fields
                )
            if not valid_owner:
                raise RagReleaseLedgerError('authorization execution owner changed')
        cases = by_kind.get('case', [])
        if cases:
            if len(cases) != 1:
                raise RagReleaseLedgerError('case mutation count is invalid')
            case = cases[0]
            case_before = before_by_kind['case'][0]
            if (payload['case_state_before'] is None and case_before is not None) or (
                payload['case_state_before'] is not None
                and (
                    case_before is None
                    or case_before['state'] != payload['case_state_before']
                )
            ):
                raise RagReleaseLedgerError('case before-image differs from payload')
            if (
                case['ledger_uuid'] != payload['ledger_uuid']
                or case['ledger_epoch'] != payload['ledger_epoch']
                or case['approval_id_hmac'] != payload['approval_id_hmac']
                or case['case_id_hmac'] != payload['case_id_hmac']
                or case['state'] != payload['case_state_after']
                or case['case_projection_hmac'] != payload['case_projection_hmac']
                or case['runtime_agent_run_id_hmac']
                != payload['runtime_agent_run_id_hmac']
                or any(
                    Decimal(str(case[column])) != Decimal(str(payload[key]))
                    for column, key in (
                        (
                            'embedding_reserved_cost_usd',
                            'case_embedding_reserved_cost_usd',
                        ),
                        (
                            'generation_reserved_cost_usd',
                            'case_generation_reserved_cost_usd',
                        ),
                        ('total_reserved_cost_usd', 'case_total_reserved_cost_usd'),
                    )
                )
            ):
                raise RagReleaseLedgerError('case payload differs from mutation')
            if case_before is not None and any(
                case_before[key] != case[key]
                for key in (
                    'ledger_uuid',
                    'ledger_epoch',
                    'approval_id_hmac',
                    'case_id_hmac',
                    'manifest_ordinal',
                    'runtime_agent_run_id_hmac',
                    'embedding_reserved_cost_usd',
                    'generation_reserved_cost_usd',
                    'total_reserved_cost_usd',
                )
            ):
                raise RagReleaseLedgerError('case immutable fields changed')
        dispatches = by_kind.get('dispatch', [])
        if dispatches:
            if len(dispatches) != 1:
                raise RagReleaseLedgerError('dispatch mutation count is invalid')
            dispatch = dispatches[0]
            dispatch_before = before_by_kind['dispatch'][0]
            if dispatch_before is None:
                if not (
                    payload['transition_kind'] == 'component_claim'
                    and payload['dispatch_state_before'] == 'not_attempted'
                    and payload['dispatch_count_before'] == 0
                ):
                    raise RagReleaseLedgerError(
                        'dispatch before-image differs from payload'
                    )
            elif (
                dispatch_before['state'] != payload['dispatch_state_before']
                or dispatch_before['dispatch_count'] != payload['dispatch_count_before']
            ):
                raise RagReleaseLedgerError(
                    'dispatch before-image differs from payload'
                )
            if (
                dispatch['ledger_uuid'] != payload['ledger_uuid']
                or dispatch['ledger_epoch'] != payload['ledger_epoch']
                or dispatch['approval_id_hmac'] != payload['approval_id_hmac']
                or dispatch['case_id_hmac'] != payload['case_id_hmac']
                or dispatch['component'] != payload['component']
                or dispatch['state'] != payload['dispatch_state_after']
                or dispatch['dispatch_count'] != payload['dispatch_count_after']
                or dispatch['dispatch_fence_hmac'] != payload['dispatch_fence_hmac']
                or dispatch['charge_basis'] != payload['charge_basis_after']
                or any(
                    Decimal(str(dispatch[column])) != Decimal(str(payload[key]))
                    for column, key in (
                        ('reserved_cost_usd', 'reserved_cost_usd'),
                        ('charged_cost_usd', 'charged_cost_usd'),
                    )
                )
            ):
                raise RagReleaseLedgerError('dispatch payload differs from mutation')
            if dispatch_before is not None and any(
                dispatch_before[key] != dispatch[key]
                for key in (
                    'ledger_uuid',
                    'ledger_epoch',
                    'approval_id_hmac',
                    'case_id_hmac',
                    'component',
                    'dispatch_fence_hmac',
                    'reserved_cost_usd',
                )
            ):
                raise RagReleaseLedgerError('dispatch immutable fields changed')
        reports = by_kind.get('quality_report', [])
        if reports and (
            len(reports) != 1
            or before_by_kind['quality_report'][0] is not None
            or reports[0]['ledger_uuid'] != payload['ledger_uuid']
            or reports[0]['ledger_epoch'] != payload['ledger_epoch']
            or reports[0]['approval_id_hmac'] != payload['approval_id_hmac']
            or reports[0]['quality_report_hmac'] != payload['quality_report_hmac']
            or reports[0]['manifest_hmac'] != authorization['manifest_hmac']
            or reports[0]['baseline_hmac'] != authorization['baseline_hmac']
            or reports[0]['reviewer_roster_hmac']
            != authorization['reviewer_roster_hmac']
        ):
            raise RagReleaseLedgerError('quality report differs from mutation')
        provider = by_kind.get('provider_safety_authority', [])
        if (
            len(provider) != 1
            or before_by_kind['provider_safety_authority'][0] is None
            or provider[0]['envelope_digest']
            != payload['provider_safety_envelope_digest']
        ):
            raise RagReleaseLedgerError('provider safety payload differs from mutation')
        all_readiness = by_kind.get('provider_readiness', [])
        active_readiness = [item for item in all_readiness if item['active'] is True]
        if (
            len(active_readiness) != 2
            or {item['component'] for item in active_readiness}
            != {'query_embedding', 'answer_generation'}
            or any(item['authority_id'] != provider[0]['id'] for item in all_readiness)
        ):
            raise RagReleaseLedgerError('provider readiness whole set is required')
        kind = payload['transition_kind']
        approved_digest = authorization['provider_safety_envelope_digest']
        current_digest = provider[0]['envelope_digest']
        ready = all(item['state'] == 'ready' for item in active_readiness)
        drift_kinds = {
            'authorization_abort_control',
            'authorization_abort_component_snapshot',
            'authorization_abort_final',
            'authorization_abort_snapshot',
        }
        if kind in drift_kinds:
            if current_digest == approved_digest and ready:
                raise RagReleaseLedgerError(
                    'provider abort requires drift or non-ready proof'
                )
        elif kind == 'authorization_abort_component':
            if (
                before_by_kind['provider_safety_authority'][0]['envelope_digest']
                != approved_digest
                or current_digest == approved_digest
            ):
                raise RagReleaseLedgerError(
                    'provider incident approved before-image differs'
                )
        elif kind not in {
            'authorization_abort_execution_crash',
            'authorization_abort_corpus_drift',
        } and (current_digest != approved_digest or not ready):
            raise RagReleaseLedgerError(
                'current provider snapshot differs from approval'
            )
        if provider and any(
            before_by_kind['provider_safety_authority'][0][key]  # type: ignore[index]
            != provider[0][key]
            for key in (
                'id',
                'authority_uuid',
                'designated_environment_id',
                'fingerprint_key_version',
                'fingerprint_key_material_verifier',
                'created_at',
            )
        ):
            raise RagReleaseLedgerError('provider safety immutable fields changed')
        run_hmacs: set[str] = set()
        for index, run in enumerate(by_kind.get('agent_run', [])):
            run_before = before_by_kind['agent_run'][index]
            if run.get('run_contract_version') != 'rag-run:v2':
                raise RagReleaseLedgerError('agent run contract differs from mutation')
            if payload['transition_kind'] == 'case_claim':
                if run_before is not None:
                    raise RagReleaseLedgerError('agent run insert before-image exists')
            elif run_before is None:
                raise RagReleaseLedgerError('agent run before-image is missing')
            elif any(
                run_before[key] != run[key]
                for key in (
                    'id',
                    'agent_name',
                    'prompt_version',
                    'source_window',
                    'cache_key',
                    'model_name',
                    'workflow_thread_id',
                    'effect_key',
                    'run_contract_version',
                )
            ):
                raise RagReleaseLedgerError('agent run immutable fields changed')
            run_hmacs.add(
                rag_identity_hmac(
                    {'agent_run_id': run['id']},
                    secret=identity_secret,
                    schema_version='rag-runtime-agent-run-id:v1',
                    policy_version='rag-run:v2',
                )
            )
        if run_hmacs and run_hmacs != {payload['runtime_agent_run_id_hmac']}:
            raise RagReleaseLedgerError('agent run identity differs from mutation')
        for index, component in enumerate(by_kind.get('cost_component', [])):
            component_before = before_by_kind['cost_component'][index]
            component_run_hmac = rag_identity_hmac(
                {'agent_run_id': component['agent_run_id']},
                secret=identity_secret,
                schema_version='rag-runtime-agent-run-id:v1',
                policy_version='rag-run:v2',
            )
            if component_run_hmac != payload['runtime_agent_run_id_hmac'] or component[
                'component'
            ] not in {'query_embedding', 'answer_generation'}:
                raise RagReleaseLedgerError(
                    'cost component identity differs from mutation'
                )
            if payload['transition_kind'] == 'case_claim':
                if component_before is not None:
                    raise RagReleaseLedgerError(
                        'cost component insert before-image exists'
                    )
            elif component_before is None:
                raise RagReleaseLedgerError('cost component before-image is missing')
            elif any(
                component_before[key] != component[key]
                for key in (
                    'id',
                    'agent_run_id',
                    'component',
                    'component_ordinal',
                    'provider',
                    'model',
                    'authorized_model_config_version',
                    'authorized_model_config_snapshot_hmac',
                    'authorized_cost_policy_version',
                    'authorized_token_estimator_version',
                    'authorized_policy_snapshot_hmac',
                )
            ):
                raise RagReleaseLedgerError('cost component immutable fields changed')
            if component_before is not None:
                cancelled = (
                    component_before['dispatch_state'] == 'not_attempted'
                    and component['dispatch_state'] == 'terminal'
                    and component['attempted'] is False
                )
                if any(
                    (
                        component[key] != 0
                        if cancelled
                        else component_before[key] != component[key]
                    )
                    for key in (
                        'reserved_input_tokens',
                        'reserved_output_tokens',
                        'reserved_cost_usd',
                    )
                ):
                    raise RagReleaseLedgerError('cost component reservation changed')
            if component['component'] == payload['component'] and (
                component['dispatch_count'] != payload['dispatch_count_after']
                or (
                    payload['dispatch_state_after'] == 'terminal'
                    and component['dispatch_state']
                    not in {'terminal', 'abandoned_unknown'}
                )
                or (
                    payload['dispatch_state_after'] != 'terminal'
                    and component['dispatch_state'] != payload['dispatch_state_after']
                )
                or Decimal(str(component['charged_cost_usd']))
                != Decimal(str(payload['charged_cost_usd']))
            ):
                raise RagReleaseLedgerError('cost component payload differs')
        if payload['transition_kind'] == 'case_claim':
            runs = by_kind.get('agent_run', [])
            components = by_kind.get('cost_component', [])
            if len(runs) != 1 or len(components) != 2:
                raise RagReleaseLedgerError('case claim runtime roster is invalid')
            run = runs[0]
            component_map = {
                str(component['component']): component for component in components
            }
            if set(component_map) != {'query_embedding', 'answer_generation'}:
                raise RagReleaseLedgerError('case claim runtime roster is invalid')
            expected_reserves = {
                'query_embedding': payload['case_embedding_reserved_cost_usd'],
                'answer_generation': payload['case_generation_reserved_cost_usd'],
            }
            if (
                run['status'] != 'running'
                or run['run_record_phase'] != 'admission'
                or run['completed_at'] is not None
                or run['projection_owner_fence_hmac'] is not None
                or Decimal(str(run['total_charged_cost_usd'])) != Decimal('0.000000')
            ):
                raise RagReleaseLedgerError('case claim runtime parent is invalid')
            terminal_zero = 0
            for component_name, item in component_map.items():
                state = item['dispatch_state']
                if state not in {'not_attempted', 'terminal'}:
                    raise RagReleaseLedgerError('case claim runtime child is invalid')
                if state == 'terminal':
                    terminal_zero += 1
                    if (
                        int(item['reserved_input_tokens']) != 0
                        or int(item['reserved_output_tokens']) != 0
                        or Decimal(str(item['reserved_cost_usd']))
                        != Decimal('0.000000')
                    ):
                        raise RagReleaseLedgerError(
                            'case claim runtime child is invalid'
                        )
                elif Decimal(str(item['reserved_cost_usd'])) != Decimal(
                    str(expected_reserves[component_name])
                ):
                    raise RagReleaseLedgerError('case claim runtime child is invalid')
                if (
                    item['attempted'] is not False
                    or item['dispatch_count'] != 0
                    or item['actual_input_tokens'] is not None
                    or item['actual_output_tokens'] is not None
                    or Decimal(str(item['charged_cost_usd'])) != Decimal('0.000000')
                    or item['charge_basis'] != 'zero'
                    or item['overrun'] is not False
                    or item['dispatch_fence_hmac'] is not None
                    or item['process_instance_hmac'] is not None
                    or item['terminal_outcome'] is not None
                ):
                    raise RagReleaseLedgerError('case claim runtime child is invalid')
            if terminal_zero > 1:
                raise RagReleaseLedgerError('case claim runtime children are invalid')
        readiness = by_kind.get('provider_readiness', [])
        for index, item in enumerate(readiness):
            item_before = before_by_kind['provider_readiness'][index]
            if item_before is None or any(
                item_before[key] != item[key]
                for key in (
                    'id',
                    'authority_id',
                    'component',
                    'provider',
                    'model',
                    'reasoning_or_config_identity',
                    'authorized_model_config_version',
                    'authorized_model_config_snapshot_hmac',
                    'authorized_cost_policy_version',
                    'authorized_token_estimator_version',
                    'authorized_fingerprint_key_version',
                    'authorized_fingerprint_key_material_verifier',
                    'authorized_policy_snapshot_hmac',
                )
            ):
                raise RagReleaseLedgerError('provider readiness differs from mutation')
        if payload['transition_kind'] == 'authorization_abort_component':
            incident_indices = [
                index
                for index, changed in enumerate(
                    mutated_by_kind.get('provider_readiness', [])
                )
                if changed
            ]
            if (
                len(provider) != 1
                or mutated_by_kind.get('provider_safety_authority') != [True]
                or len(incident_indices) != 1
            ):
                raise RagReleaseLedgerError('provider incident mutation is required')
            authority_before = before_by_kind['provider_safety_authority'][0]
            readiness_index = incident_indices[0]
            readiness_before = before_by_kind['provider_readiness'][readiness_index]
            authority_after = provider[0]
            readiness_after = readiness[readiness_index]
            expected_state = (
                'blocked_overrun'
                if payload['outcome'] == 'provider_usage_overrun'
                else 'blocked_remediation'
            )
            if (
                authority_before is None
                or readiness_before is None
                or readiness_after['component'] != payload['component']
                or readiness_after['active'] is not True
                or any(
                    item['state'] != 'ready'
                    for index, item in enumerate(readiness)
                    if index != readiness_index and item['active'] is True
                )
                or authority_after['global_safety_generation']
                != authority_before['global_safety_generation'] + 1
                or readiness_before['state'] != 'ready'
                or readiness_after['state'] != expected_state
                or readiness_after['state_version']
                != readiness_before['state_version'] + 1
                or readiness_after['family_safety_generation']
                != authority_after['global_safety_generation']
                or readiness_after['reviewed_gate_reference_hmac']
                != readiness_before['reviewed_gate_reference_hmac']
                or readiness_after['reset_by'] != readiness_before['reset_by']
                or readiness_after['reset_at'] != readiness_before['reset_at']
            ):
                raise RagReleaseLedgerError('provider incident state differs')
            run_hmac = rag_identity_hmac(
                {'agent_run_id': readiness_after['overrun_agent_run_id']},
                secret=identity_secret,
                schema_version='rag-runtime-agent-run-id:v1',
                policy_version='rag-run:v2',
            )
            if (
                run_hmac != payload['runtime_agent_run_id_hmac']
                or Decimal(str(readiness_after['overrun_cost_usd']))
                != Decimal(str(payload['charged_cost_usd']))
                or readiness_after['overrun_input_tokens'] is None
                or readiness_after['overrun_output_tokens'] is None
                or readiness_after['overrun_observed_at'] is None
            ):
                raise RagReleaseLedgerError('provider incident evidence differs')
        elif any(mutated_by_kind.get('provider_safety_authority', ())) or any(
            mutated_by_kind.get('provider_readiness', ())
        ):
            raise RagReleaseLedgerError('unexpected provider safety mutation')
        _assert_exact_transition_snapshots(
            payload, by_kind, before_by_kind, mutated_by_kind
        )
        self._assert_approved_case_claim(
            payload,
            [
                (row.row_kind, after)
                for row, after in zip(self._rows, self._after_snapshots, strict=True)
                if row.row_kind in {'agent_run', 'cost_component'}
            ],
            approved_case_claim=approved_case_claim,
            identity_secret=identity_secret,
            barrier_guard=barrier_guard,
        )
        for row, before, after in zip(
            self._rows, self._before_snapshots, self._after_snapshots, strict=True
        ):
            if row.row_kind in {'agent_run', 'cost_component'}:
                _assert_runtime_column_delta(
                    payload,
                    row.row_kind,
                    before,
                    after,
                    self._captured_rows('cost_component'),
                )
                self._assert_runtime_mutation_hmac(
                    payload, row, before, after, identity_secret=identity_secret
                )

    def _assert_observation_projection(
        self, payload: Mapping[str, object], *, identity_secret: bytes
    ) -> None:
        actual = tuple(
            sorted(
                (
                    row.row_kind,
                    release_row_identity_hmac(
                        row.row_kind, row.primary_key, identity_secret=identity_secret
                    ),
                    release_observation_projection_hmac(
                        row.row_kind, snapshot, identity_secret=identity_secret
                    ),
                )
                for row, snapshot in zip(
                    self._observation_rows, self._observation_snapshots, strict=True
                )
            )
        )
        if actual != _validate_observation_set(payload['observation_set']):
            raise RagReleaseLedgerError('actual observation set differs from payload')


def _meaningful_changed(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    ignored: frozenset[str] = frozenset({'updated_at'}),
) -> bool:
    keys = (set(before) | set(after)) - ignored
    return any(before.get(key) != after.get(key) for key in keys)


def _assert_runtime_column_delta(payload, row_kind, before, after, costs):
    """Deny every column delta except the exact lifecycle for this kind."""
    kind = payload['transition_kind']
    if before is None:
        if kind != 'case_claim':
            raise RagReleaseLedgerError('runtime insert kind is invalid')
        return
    expected = dict(before)
    if row_kind == 'agent_run':
        if kind == 'component_outcome' and payload['component'] == 'answer_generation':
            expected.update(
                run_record_phase='cost_finalized_pending_projection',
                projection_owner_fence_hmac=payload['execution_runner_fence_hmac'],
            )
        elif kind in {
            'case_failure',
            'case_safe_outcome',
            'case_outcome',
            'authorization_abort_control',
            'authorization_abort_component',
            'authorization_abort_component_snapshot',
            'authorization_abort_corpus_drift',
            'authorization_abort_execution_crash',
        }:
            completed = after.get('completed_at')
            if (
                before.get('completed_at') is not None
                or not isinstance(completed, datetime)
                or _runtime_utc(completed) < _runtime_utc(before['started_at'])
            ):
                raise RagReleaseLedgerError('runtime completion timestamp is invalid')
            expected.update(
                status='complete'
                if kind in {'case_safe_outcome', 'case_outcome'}
                else 'failed',
                run_record_phase='admission_only'
                if kind == 'authorization_abort_execution_crash'
                and before['run_record_phase'] == 'admission'
                else 'final',
                completed_at=completed,
            )
        else:
            raise RagReleaseLedgerError('agent run mutation kind is invalid')
        children = [item for item in costs if item['agent_run_id'] == after['id']]
        if len(children) != 2 or {item['component'] for item in children} != {
            'query_embedding',
            'answer_generation',
        }:
            raise RagReleaseLedgerError('runtime mutation child roster is invalid')
        expected['total_charged_cost_usd'] = sum(
            Decimal(str(item['charged_cost_usd'])) for item in children
        )
    elif before['dispatch_state'] == 'not_attempted':
        if kind == 'component_claim' and after['component'] == payload['component']:
            expected.update(
                dispatch_state='dispatching',
                attempted=True,
                dispatch_count=1,
                process_instance_hmac=payload['execution_process_instance_hmac'],
                dispatch_fence_hmac=payload['dispatch_fence_hmac'],
                charge_basis='reserved',
                charged_cost_usd=before['reserved_cost_usd'],
            )
        elif kind in {
            'case_failure',
            'case_safe_outcome',
            'authorization_abort_control',
            'authorization_abort_component',
            'authorization_abort_component_snapshot',
            'authorization_abort_corpus_drift',
            'authorization_abort_execution_crash',
        }:
            expected.update(
                dispatch_state='terminal',
                reserved_input_tokens=0,
                reserved_output_tokens=0,
                reserved_cost_usd=Decimal(0),
            )
        else:
            raise RagReleaseLedgerError('cost component mutation kind is invalid')
    elif (
        before['dispatch_state'] == 'dispatching'
        and kind
        in {
            'component_outcome',
            'case_failure',
            'authorization_abort_component',
            'authorization_abort_component_snapshot',
            'authorization_abort_corpus_drift',
            'authorization_abort_execution_crash',
        }
        and after['component'] == payload['component']
    ):
        crash = kind == 'authorization_abort_execution_crash'
        actual = payload['charge_basis_after'] == 'actual'
        for field in ('actual_input_tokens', 'actual_output_tokens'):
            value = after[field]
            if (actual and (type(value) is not int or value < 0)) or (
                not actual and value is not None
            ):
                raise RagReleaseLedgerError(
                    'cost component token projection is invalid'
                )
            expected[field] = value
        expected.update(
            dispatch_state='abandoned_unknown' if crash else 'terminal',
            charged_cost_usd=Decimal(str(payload['charged_cost_usd'])),
            charge_basis=payload['charge_basis_after'],
            overrun=kind == 'authorization_abort_component'
            and payload['outcome'] == 'provider_usage_overrun',
            terminal_outcome='component_succeeded'
            if kind == 'component_outcome'
            else 'abandoned_unknown'
            if crash
            else payload['outcome'],
        )
    else:
        raise RagReleaseLedgerError('terminal cost component is immutable')
    if _runtime_projection(row_kind, expected) != _runtime_projection(row_kind, after):
        if expected.get('projection_owner_fence_hmac') != after.get(
            'projection_owner_fence_hmac'
        ):
            raise RagReleaseLedgerError('runtime projection owner fence changed')
        if expected.get('overrun') != after.get('overrun'):
            raise RagReleaseLedgerError(
                'component overrun requires exact safety incident'
            )
        raise RagReleaseLedgerError(
            f'{row_kind} immutable fields or exact lifecycle changed'
        )
    if not _meaningful_changed(before, after):
        raise RagReleaseLedgerError('runtime mutation is not semantic')


def _runtime_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _assert_exact_transition_snapshots(
    payload: Mapping[str, object],
    after_by_kind: Mapping[str, list[dict[str, object]]],
    before_by_kind: Mapping[str, list[dict[str, object] | None]],
    mutated_by_kind: Mapping[str, list[bool]],
) -> None:
    """Bind each signed transition kind to its actual mutable DB lifecycle."""
    kind = str(payload['transition_kind'])
    case_bound = payload['case_id_hmac'] is not None
    runs = after_by_kind.get('agent_run', [])
    run_befores = before_by_kind.get('agent_run', [])
    costs = after_by_kind.get('cost_component', [])
    cost_befores = before_by_kind.get('cost_component', [])
    dispatches = after_by_kind.get('dispatch', [])
    dispatch_befores = before_by_kind.get('dispatch', [])

    if len(runs) != (1 if case_bound else 0):
        raise RagReleaseLedgerError('runtime parent mutation matrix is invalid')
    if len(dispatches) != (1 if payload['component'] is not None else 0):
        raise RagReleaseLedgerError('runtime dispatch mutation matrix is invalid')
    for row_before, row_after, mutated in zip(
        run_befores, runs, mutated_by_kind.get('agent_run', []), strict=True
    ):
        if (
            mutated
            and row_before is not None
            and not _meaningful_changed(
                row_before, row_after, ignored=frozenset({'updated_at', 'started_at'})
            )
        ):
            raise RagReleaseLedgerError('agent run mutation is timestamp-only')
    for row_before, row_after, mutated in zip(
        cost_befores, costs, mutated_by_kind.get('cost_component', []), strict=True
    ):
        if (
            mutated
            and row_before is not None
            and not _meaningful_changed(row_before, row_after)
        ):
            raise RagReleaseLedgerError('cost component mutation is timestamp-only')
        if row_before is not None:
            first_claim = (
                kind == 'component_claim'
                and row_after['component'] == payload['component']
            )
            owner_fields = ('process_instance_hmac', 'dispatch_fence_hmac')
            if first_claim:
                valid_owner = all(row_before.get(key) is None for key in owner_fields)
                valid_owner = (
                    valid_owner
                    and row_after.get('process_instance_hmac')
                    == payload['execution_process_instance_hmac']
                    and row_after.get('dispatch_fence_hmac')
                    == payload['dispatch_fence_hmac']
                )
            else:
                valid_owner = all(
                    row_before.get(key) == row_after.get(key) for key in owner_fields
                )
            if not valid_owner:
                raise RagReleaseLedgerError('cost component execution owner changed')
            if (
                row_before['dispatch_state'] in {'terminal', 'abandoned_unknown'}
                and mutated
            ):
                raise RagReleaseLedgerError('terminal cost component is immutable')
    for row_before, row_after, mutated in zip(
        dispatch_befores, dispatches, mutated_by_kind.get('dispatch', []), strict=True
    ):
        if (
            mutated
            and row_before is not None
            and not _meaningful_changed(row_before, row_after)
        ):
            raise RagReleaseLedgerError('dispatch mutation is timestamp-only')

    if not case_bound:
        if costs or dispatches:
            raise RagReleaseLedgerError('case-null runtime mutation is invalid')
        return

    run = runs[0]
    run_before = run_befores[0]
    if kind == 'case_claim':
        return
    if run_before is None:
        raise RagReleaseLedgerError('runtime parent before-image is missing')

    component = payload['component']
    current_after = next(
        (item for item in costs if item['component'] == component), None
    )
    current_before = None
    if current_after is not None:
        current_index = costs.index(current_after)
        current_before = cost_befores[current_index]

    if component is not None:
        if current_after is None or current_before is None:
            raise RagReleaseLedgerError('runtime current component is missing')
        expected_overrun = (
            kind == 'authorization_abort_component'
            and payload['outcome'] == 'provider_usage_overrun'
        )
        if (current_after.get('overrun') is True) != expected_overrun:
            raise RagReleaseLedgerError(
                'component overrun requires exact safety incident'
            )
        dispatch = dispatches[0]
        dispatch_before = dispatch_befores[0]
        if (
            current_after['dispatch_count'] != dispatch['dispatch_count']
            or current_after['dispatch_fence_hmac'] != dispatch['dispatch_fence_hmac']
            or Decimal(str(current_after['reserved_cost_usd']))
            != Decimal(str(dispatch['reserved_cost_usd']))
            or Decimal(str(current_after['charged_cost_usd']))
            != Decimal(str(dispatch['charged_cost_usd']))
            or current_after['charge_basis'] != dispatch['charge_basis']
            or (
                current_after['dispatch_state'] == 'abandoned_unknown'
                and dispatch['state'] != 'terminal'
            )
            or (
                current_after['dispatch_state'] != 'abandoned_unknown'
                and current_after['dispatch_state'] != dispatch['state']
            )
        ):
            raise RagReleaseLedgerError('release/runtime dispatch pair differs')
        if kind == 'component_claim':
            if (
                dispatch_before is not None
                or current_before['dispatch_state'] != 'not_attempted'
                or current_after['dispatch_state'] != 'dispatching'
                or current_after['attempted'] is not True
                or current_after['dispatch_count'] != 1
                or current_after['actual_input_tokens'] is not None
                or current_after['actual_output_tokens'] is not None
                or current_after['charge_basis'] != 'reserved'
                or current_after['terminal_outcome'] is not None
                or current_after['overrun'] is not False
                or current_after['process_instance_hmac']
                != payload['execution_process_instance_hmac']
            ):
                raise RagReleaseLedgerError('component claim runtime state is invalid')
        else:
            if (
                dispatch_before is None
                or dispatch_before['state'] != 'dispatching'
                or current_before['dispatch_state'] != 'dispatching'
                or current_after['dispatch_state']
                not in {'terminal', 'abandoned_unknown'}
                or current_after['attempted'] is not True
                or current_after['dispatch_count'] != 1
                or current_after['process_instance_hmac']
                != payload['execution_process_instance_hmac']
            ):
                raise RagReleaseLedgerError(
                    'component outcome runtime state is invalid'
                )
            expected_terminal = (
                'component_succeeded'
                if kind == 'component_outcome'
                else payload['outcome']
            )
            if kind == 'authorization_abort_execution_crash':
                expected_terminal = 'abandoned_unknown'
            if current_after['terminal_outcome'] != expected_terminal:
                raise RagReleaseLedgerError('component terminal outcome differs')

    if kind in {'component_claim', 'component_outcome'}:
        expected_phase = 'admission'
        expected_fence = None
        if kind == 'component_outcome' and component == 'answer_generation':
            expected_phase = 'cost_finalized_pending_projection'
            expected_fence = payload['execution_runner_fence_hmac']
        if (
            run_before['status'] != 'running'
            or run_before['run_record_phase'] != 'admission'
            or run['status'] != 'running'
            or run['run_record_phase'] != expected_phase
            or run['completed_at'] is not None
            or run['projection_owner_fence_hmac'] != expected_fence
        ):
            raise RagReleaseLedgerError('component parent lifecycle is invalid')
        return

    complete_kinds = {'case_safe_outcome', 'case_outcome'}
    crash = kind == 'authorization_abort_execution_crash'
    if kind in complete_kinds:
        expected_status = 'complete'
        expected_phase = 'final'
    else:
        expected_status = 'failed'
        expected_phase = (
            'admission_only'
            if crash and run_before['run_record_phase'] == 'admission'
            else 'final'
        )
    if (
        run_before['status'] != 'running'
        or run['status'] != expected_status
        or run['run_record_phase'] != expected_phase
        or run['completed_at'] is None
    ):
        raise RagReleaseLedgerError('case terminal parent lifecycle is invalid')
    if kind == 'case_outcome' and run_before['run_record_phase'] != (
        'cost_finalized_pending_projection'
    ):
        raise RagReleaseLedgerError('case projection parent phase is invalid')
    if (
        run_before['run_record_phase'] == 'admission'
        and run['projection_owner_fence_hmac'] is not None
    ):
        raise RagReleaseLedgerError(
            'admission finalization projection fence is invalid'
        )
    pending_failure = (
        kind
        in {
            'case_failure',
            'authorization_abort_control',
            'authorization_abort_corpus_drift',
            'authorization_abort_execution_crash',
        }
        and run_before['run_record_phase'] == 'cost_finalized_pending_projection'
    )
    if (
        kind != 'case_outcome'
        and not pending_failure
        and run_before['run_record_phase'] != 'admission'
    ):
        raise RagReleaseLedgerError('case terminal parent phase is invalid')
    if run_before['run_record_phase'] == 'cost_finalized_pending_projection' and (
        run_before['projection_owner_fence_hmac']
        != payload['execution_runner_fence_hmac']
        or run['projection_owner_fence_hmac']
        != run_before['projection_owner_fence_hmac']
        or any(mutated_by_kind.get('cost_component', []))
        or component is not None
    ):
        raise RagReleaseLedgerError('pending projection fence or terminal costs differ')
    for item in costs:
        if item['dispatch_state'] not in {'terminal', 'abandoned_unknown'}:
            raise RagReleaseLedgerError('case terminal child is nonterminal')


def release_row_identity_hmac(
    row_kind: str,
    primary_key: Mapping[str, object],
    *,
    identity_secret: bytes,
) -> str:
    registry = _ROW_IDENTITY_REGISTRY.get(row_kind)
    if registry is None or type(primary_key) is not dict:
        raise RagReleaseLedgerError('release row identity registry is invalid')
    schema_version, keys = registry
    if tuple(primary_key) != keys:
        raise RagReleaseLedgerError('release row primary key is invalid')
    value = dict(primary_key)
    for key, item in value.items():
        if key in {'ledger_uuid', 'authority_uuid'}:
            try:
                parsed = UUID(str(item))
            except (TypeError, ValueError) as exc:
                raise RagReleaseLedgerError(
                    'release row primary key is invalid'
                ) from exc
            if str(parsed) != item:
                raise RagReleaseLedgerError('release row primary key is invalid')
        elif key in {'ledger_epoch', 'to_generation', 'agent_run_id'}:
            if type(item) is not int or item <= 0:
                raise RagReleaseLedgerError('release row primary key is invalid')
        elif key.endswith('_hmac'):
            require_lower_hmac(item)
        elif type(item) is not str or not item:
            raise RagReleaseLedgerError('release row primary key is invalid')
    return rag_identity_hmac(
        value,
        secret=identity_secret,
        schema_version=schema_version,
        policy_version='rag-live-gate:v1',
    )


def _observation_json_value(value: object) -> object:
    """Encode SQL after-images without relying on driver-specific reprs."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RagReleaseLedgerError('release observation projection is invalid')
        return {'binary64_hex': value.hex()}
    if isinstance(value, Decimal):
        return format(value, 'f')
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).isoformat().replace('+00:00', 'Z')
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RagReleaseLedgerError('release observation projection is invalid')
        return {
            str(key): _observation_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence):
        return [_observation_json_value(item) for item in value]
    raise RagReleaseLedgerError('release observation projection is invalid')


def release_observation_projection_hmac(
    row_kind: str,
    snapshot: Mapping[str, object],
    *,
    identity_secret: bytes,
) -> str:
    """Seal a typed read-only DB after-image without exposing raw values."""
    if row_kind not in _ROW_IDENTITY_REGISTRY or type(snapshot) is not dict:
        raise RagReleaseLedgerError('release observation projection is invalid')
    projection = {
        str(key): _observation_json_value(value)
        for key, value in sorted(snapshot.items(), key=lambda pair: str(pair[0]))
        if key not in {'created_at', 'updated_at'}
    }
    return rag_identity_hmac(
        {'row_kind': row_kind, 'projection': projection},
        secret=identity_secret,
        schema_version='rag-release-row-observation-projection:v1',
        policy_version='rag-live-gate:v1',
    )


def _assert_runtime_json_types(value):
    """SQL JSON has native JSON types, never observation-encoding aliases."""
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if math.isfinite(value):
            return
    elif type(value) is dict:
        if all(type(key) is str for key in value):
            for item in value.values():
                _assert_runtime_json_types(item)
            return
    elif type(value) is list:
        for item in value:
            _assert_runtime_json_types(item)
        return
    raise RagReleaseLedgerError('runtime JSON literal type is invalid')


def _runtime_projection(row_kind, snapshot):
    if type(row_kind) is not str or row_kind not in {'agent_run', 'cost_component'}:
        raise RagReleaseLedgerError('runtime mutation kind is invalid')
    table, _aliases = RagReleaseMutationSet._table(row_kind)
    if snapshot is None:
        return None
    if (
        type(snapshot) is not dict
        or any(type(key) is not str for key in snapshot)
        or set(snapshot) != set(table.c.keys())
    ):
        raise RagReleaseLedgerError('runtime mutation requires every column')
    result = {}
    for column in table.c:
        value = snapshot[column.name]
        expected_type = column.type.python_type
        if value is None:
            if not column.nullable:
                raise RagReleaseLedgerError('runtime literal type is invalid')
        elif type(value) is not expected_type:
            raise RagReleaseLedgerError('runtime literal type is invalid')
        if expected_type is dict and value is not None:
            try:
                _assert_runtime_json_types(value)
            except RecursionError:
                raise RagReleaseLedgerError(
                    'runtime JSON literal type is invalid'
                ) from None
            # Preserve native JSON numbers/objects. The observation encoder's
            # float tag would collide with an ordinary JSON object of that shape.
            result[str(column.name)] = value
        else:
            if expected_type is Decimal and value is not None:
                if not value.is_finite():
                    raise RagReleaseLedgerError('runtime money literal is invalid')
                quantized = value.quantize(Decimal('0.000001'))
                if value != quantized:
                    raise RagReleaseLedgerError('runtime money literal is invalid')
                value = quantized
            result[str(column.name)] = _observation_json_value(value)
    return result


def release_runtime_mutation_hmac(
    row_kind, before, after, *, identity_secret: bytes
) -> str:
    """Seal complete typed runtime images; no raw audit or metadata is serialized."""
    return rag_identity_hmac(
        {
            'row_kind': row_kind,
            'before': _runtime_projection(row_kind, before),
            'after': _runtime_projection(row_kind, after),
        },
        secret=identity_secret,
        schema_version='rag-release-runtime-mutation:v1',
        policy_version='rag-live-gate:v1',
    )


_TRANSITION_KEYS = frozenset(
    {
        'affected_rows',
        'approval_hmac',
        'approval_id_hmac',
        'approved_corpus_snapshot_hmac',
        'approved_provider_safety_snapshot_hmac',
        'authorization_charged_cost_usd',
        'authorization_reserved_cost_usd',
        'authorization_state_after',
        'authorization_state_before',
        'case_claim_count',
        'case_embedding_reserved_cost_usd',
        'case_generation_reserved_cost_usd',
        'case_id_hmac',
        'case_projection_hmac',
        'case_state_after',
        'case_state_before',
        'case_total_reserved_cost_usd',
        'charge_basis_after',
        'charged_cost_usd',
        'component',
        'current_corpus_snapshot_hmac',
        'dispatch_count_after',
        'dispatch_count_before',
        'dispatch_fence_hmac',
        'dispatch_state_after',
        'dispatch_state_before',
        'execution_crash_attestation_hmac',
        'execution_process_instance_hmac',
        'execution_runner_fence_hmac',
        'embedding_dispatch_count',
        'from_generation',
        'generation_dispatch_count',
        'ledger_epoch',
        'ledger_uuid',
        'observation_set',
        'outcome',
        'provider_safety_envelope_digest',
        'quality_report_hmac',
        'reserved_cost_usd',
        'runtime_agent_run_id_hmac',
        'to_generation',
        'total_dispatch_count',
        'transition_kind',
        'validation_database_identity_hmac',
    }
)
_AFFECTED_ROW_KINDS = frozenset(
    {
        'authorization',
        'case',
        'dispatch',
        'agent_run',
        'cost_component',
        'provider_safety_authority',
        'provider_readiness',
        'quality_report',
        'release_ledger',
        'release_transition',
    }
)
_OUTCOMES: Mapping[str, frozenset[str | None]] = {
    'authorization_bootstrap': frozenset({None}),
    'case_claim': frozenset({None}),
    'component_claim': frozenset({None}),
    'component_outcome': frozenset({'component_succeeded'}),
    'case_safe_outcome': frozenset({'no_match', 'hidden_only', 'safety_filter_empty'}),
    'case_outcome': frozenset(
        {'supported', 'insufficient_evidence', 'evidence_unavailable'}
    ),
    'case_failure': frozenset(
        {
            'budget_exceeded',
            'retriever_unavailable',
            'model_unavailable',
            'model_provider_failed',
            'structured_output_invalid',
            'citation_validation_failed',
            'persistence_failed',
            'unexpected_internal_error',
        }
    ),
    'authorization_abort_control': frozenset({'provider_safety_unavailable'}),
    'authorization_abort_component_snapshot': frozenset(
        {'provider_safety_unavailable'}
    ),
    'authorization_abort_final': frozenset({'provider_safety_unavailable'}),
    'authorization_abort_snapshot': frozenset({'provider_safety_unavailable'}),
    'authorization_abort_component': frozenset(
        {
            'provider_usage_overrun',
            'provider_response_identity_invalid',
            'provider_embedding_payload_invalid',
            'provider_safety_unavailable',
        }
    ),
    'authorization_abort_corpus_drift': frozenset({'live_corpus_snapshot_changed'}),
    'authorization_abort_execution_crash': frozenset({'abandoned_unknown'}),
    'authorization_complete': frozenset({'quality_gate_green'}),
    'authorization_finish_failed': frozenset(
        {'ordinary_execution_failed', 'execution_contract_failed'}
    ),
    'authorization_finish_quality_failed': frozenset({'quality_gate_failed'}),
}
_AUTHORIZATION_STATES = frozenset(
    {
        'unused',
        'started',
        'complete',
        'finished_failed',
        'aborted_corpus_drift',
        'aborted_execution_crash',
        'aborted_overrun',
        'aborted_provider_safety',
    }
)
_MONEY = re.compile(r'^(?:0|[1-9][0-9]{0,11})\.[0-9]{6}$', re.ASCII)
_MONEY_KEYS = (
    'authorization_charged_cost_usd',
    'authorization_reserved_cost_usd',
    'case_embedding_reserved_cost_usd',
    'case_generation_reserved_cost_usd',
    'case_total_reserved_cost_usd',
    'charged_cost_usd',
    'reserved_cost_usd',
)
_HMAC_KEYS = (
    'approval_hmac',
    'approval_id_hmac',
    'approved_corpus_snapshot_hmac',
    'approved_provider_safety_snapshot_hmac',
    'case_id_hmac',
    'case_projection_hmac',
    'dispatch_fence_hmac',
    'execution_crash_attestation_hmac',
    'execution_process_instance_hmac',
    'execution_runner_fence_hmac',
    'provider_safety_envelope_digest',
    'quality_report_hmac',
    'runtime_agent_run_id_hmac',
    'current_corpus_snapshot_hmac',
    'validation_database_identity_hmac',
)
_CASE_FIELDS = (
    'case_embedding_reserved_cost_usd',
    'case_generation_reserved_cost_usd',
    'case_id_hmac',
    'case_projection_hmac',
    'case_state_after',
    'case_state_before',
    'case_total_reserved_cost_usd',
    'runtime_agent_run_id_hmac',
)
_DISPATCH_FIELDS = (
    'charge_basis_after',
    'charged_cost_usd',
    'component',
    'dispatch_count_after',
    'dispatch_count_before',
    'dispatch_fence_hmac',
    'dispatch_state_after',
    'dispatch_state_before',
    'reserved_cost_usd',
)


class RagReleaseLedgerError(RagReleaseAuthorityError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedReleaseTransition:
    transition_kind: str
    from_generation: int
    to_generation: int
    affected_rows: tuple[AffectedRow, ...]
    observation_set: tuple[ObservedRow, ...]
    payload_canonical_bytes: bytes
    transition_digest: str


def _require_count(value: object, *, key: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise RagReleaseLedgerError(f'{key} count is invalid')
    return value


def _validate_affected_rows(value: object) -> tuple[AffectedRow, ...]:
    if type(value) is not list or not value:
        raise RagReleaseLedgerError('affected rows are invalid')
    rows: list[AffectedRow] = []
    for item in value:
        expected_keys = {'row_identity_hmac', 'row_kind'}
        if type(item) is dict and item.get('row_kind') in {
            'agent_run',
            'cost_component',
        }:
            expected_keys.add('row_mutation_hmac')
        if type(item) is not dict or set(item) != expected_keys:
            raise RagReleaseLedgerError('affected row keys are invalid')
        if 'row_mutation_hmac' in item:
            require_lower_hmac(item['row_mutation_hmac'])
        kind = item['row_kind']
        identity = item['row_identity_hmac']
        if kind not in _AFFECTED_ROW_KINDS:
            raise RagReleaseLedgerError('affected row kind is invalid')
        require_lower_hmac(identity)
        rows.append((kind, identity))
    if rows != sorted(set(rows)):
        raise RagReleaseLedgerError('affected rows must be unique lexical order')
    required = {'release_ledger', 'release_transition'}
    if not required <= {kind for kind, _identity in rows}:
        raise RagReleaseLedgerError('affected rows omit release authority mutation')
    return tuple(rows)


def _validate_observation_set(value: object) -> tuple[ObservedRow, ...]:
    if type(value) is not list:
        raise RagReleaseLedgerError('observation set is invalid')
    rows: list[ObservedRow] = []
    for item in value:
        if type(item) is not dict or set(item) != {
            'row_identity_hmac',
            'row_kind',
            'row_projection_hmac',
        }:
            raise RagReleaseLedgerError('observation set keys are invalid')
        kind = item['row_kind']
        identity = item['row_identity_hmac']
        projection = item['row_projection_hmac']
        if kind not in _AFFECTED_ROW_KINDS or kind in {
            'release_ledger',
            'release_transition',
        }:
            raise RagReleaseLedgerError('observation row kind is invalid')
        require_lower_hmac(identity)
        require_lower_hmac(projection)
        rows.append((kind, identity, projection))
    if rows != sorted(set(rows)):
        raise RagReleaseLedgerError('observation set must be unique lexical order')
    if len({(kind, identity) for kind, identity, _projection in rows}) != len(rows):
        raise RagReleaseLedgerError('observation identity must be unique')
    return tuple(rows)


def _derived_payload_rows(
    payload: Mapping[str, object], *, identity_secret: bytes
) -> tuple[AffectedRow, ...]:
    common = {
        'ledger_uuid': payload['ledger_uuid'],
        'ledger_epoch': payload['ledger_epoch'],
    }
    keys: list[ReleaseRowPrimaryKey] = [
        ReleaseRowPrimaryKey(
            'authorization',
            {**common, 'approval_id_hmac': payload['approval_id_hmac']},
        ),
        ReleaseRowPrimaryKey('release_ledger', dict(common)),
        ReleaseRowPrimaryKey(
            'release_transition',
            {**common, 'to_generation': payload['to_generation']},
        ),
    ]
    if payload['case_id_hmac'] is not None:
        keys.append(
            ReleaseRowPrimaryKey(
                'case',
                {
                    **common,
                    'approval_id_hmac': payload['approval_id_hmac'],
                    'case_id_hmac': payload['case_id_hmac'],
                },
            )
        )
    if payload['component'] is not None:
        keys.append(
            ReleaseRowPrimaryKey(
                'dispatch',
                {
                    **common,
                    'approval_id_hmac': payload['approval_id_hmac'],
                    'case_id_hmac': payload['case_id_hmac'],
                    'component': payload['component'],
                },
            )
        )
    if payload['quality_report_hmac'] is not None:
        keys.append(
            ReleaseRowPrimaryKey(
                'quality_report',
                {**common, 'approval_id_hmac': payload['approval_id_hmac']},
            )
        )
    return tuple(
        sorted(
            (
                item.row_kind,
                release_row_identity_hmac(
                    item.row_kind,
                    item.primary_key,
                    identity_secret=identity_secret,
                ),
            )
            for item in keys
        )
    )


def _validate_bootstrap(payload: Mapping[str, object]) -> None:
    null_keys = (
        'case_embedding_reserved_cost_usd',
        'case_generation_reserved_cost_usd',
        'case_id_hmac',
        'case_projection_hmac',
        'case_state_after',
        'case_state_before',
        'case_total_reserved_cost_usd',
        'charge_basis_after',
        'charged_cost_usd',
        'component',
        'dispatch_count_after',
        'dispatch_count_before',
        'dispatch_fence_hmac',
        'dispatch_state_after',
        'dispatch_state_before',
        'execution_crash_attestation_hmac',
        'execution_process_instance_hmac',
        'execution_runner_fence_hmac',
        'quality_report_hmac',
        'reserved_cost_usd',
        'runtime_agent_run_id_hmac',
    )
    if any(payload[key] is not None for key in null_keys):
        raise RagReleaseLedgerError('authorization bootstrap null matrix is invalid')
    if (
        payload['authorization_state_before'] is not None
        or payload['authorization_state_after'] != 'unused'
        or payload['authorization_charged_cost_usd'] != '0.000000'
        or payload['authorization_reserved_cost_usd'] != '0.000000'
        or any(
            payload[key] != 0
            for key in (
                'case_claim_count',
                'embedding_dispatch_count',
                'generation_dispatch_count',
                'total_dispatch_count',
            )
        )
    ):
        raise RagReleaseLedgerError('authorization bootstrap matrix is invalid')


def _require_null_matrix(
    payload: Mapping[str, object], keys: Sequence[str], label: str
) -> None:
    if any(payload[key] is not None for key in keys):
        raise RagReleaseLedgerError(f'{label} null matrix is invalid')


def _require_present_matrix(
    payload: Mapping[str, object], keys: Sequence[str], label: str
) -> None:
    if any(payload[key] is None for key in keys):
        raise RagReleaseLedgerError(f'{label} required matrix is invalid')


def _validate_transition_matrix(
    payload: Mapping[str, object],
    rows: tuple[AffectedRow, ...],
    observations: tuple[ObservedRow, ...],
) -> None:
    kind = payload['transition_kind']
    common_hmacs = (
        'approval_hmac',
        'approval_id_hmac',
        'approved_corpus_snapshot_hmac',
        'approved_provider_safety_snapshot_hmac',
        'current_corpus_snapshot_hmac',
        'provider_safety_envelope_digest',
        'validation_database_identity_hmac',
    )
    _require_present_matrix(payload, common_hmacs, 'transition identity')
    _require_present_matrix(
        payload,
        ('authorization_charged_cost_usd', 'authorization_reserved_cost_usd'),
        'authorization cost',
    )
    authorization_reserved = Decimal(str(payload['authorization_reserved_cost_usd']))
    authorization_charged = Decimal(str(payload['authorization_charged_cost_usd']))
    if (
        authorization_reserved < 0
        or authorization_charged < 0
        or (
            authorization_charged > authorization_reserved
            and not (
                kind == 'authorization_abort_component'
                and payload['outcome'] == 'provider_usage_overrun'
            )
        )
    ):
        raise RagReleaseLedgerError('authorization cost matrix is invalid')
    for key in _MONEY_KEYS:
        if payload[key] is not None and Decimal(str(payload[key])) < 0:
            raise RagReleaseLedgerError(f'{key} cost is invalid')

    state_pair = (
        payload['authorization_state_before'],
        payload['authorization_state_after'],
    )
    if kind == 'authorization_bootstrap':
        allowed_states = {(None, 'unused')}
    elif kind == 'case_claim':
        allowed_states = {('unused', 'started'), ('started', 'started')}
    elif kind in {
        'component_claim',
        'component_outcome',
        'case_failure',
        'case_safe_outcome',
        'case_outcome',
    }:
        allowed_states = {('started', 'started')}
    elif kind == 'authorization_abort_component':
        expected = (
            'aborted_overrun'
            if payload['outcome'] == 'provider_usage_overrun'
            else 'aborted_provider_safety'
        )
        allowed_states = {('started', expected)}
    elif kind in {
        'authorization_abort_control',
        'authorization_abort_component_snapshot',
        'authorization_abort_final',
    }:
        allowed_states = {('started', 'aborted_provider_safety')}
    elif kind == 'authorization_abort_snapshot':
        allowed_states = {
            ('unused', 'aborted_provider_safety'),
            ('started', 'aborted_provider_safety'),
        }
    elif kind == 'authorization_abort_corpus_drift':
        allowed_states = {
            ('unused', 'aborted_corpus_drift'),
            ('started', 'aborted_corpus_drift'),
        }
    elif kind == 'authorization_abort_execution_crash':
        allowed_states = {('started', 'aborted_execution_crash')}
    elif kind == 'authorization_complete':
        allowed_states = {('started', 'complete')}
    else:
        allowed_states = {('started', 'finished_failed')}
    if state_pair not in allowed_states:
        raise RagReleaseLedgerError('authorization state matrix is invalid')

    owner_keys = ('execution_process_instance_hmac', 'execution_runner_fence_hmac')
    owner_may_be_null = kind == 'authorization_bootstrap' or (
        kind
        in {
            'authorization_abort_snapshot',
            'authorization_abort_corpus_drift',
        }
        and payload['authorization_state_before'] == 'unused'
    )
    if owner_may_be_null:
        _require_null_matrix(payload, owner_keys, 'execution owner')
    else:
        _require_present_matrix(payload, owner_keys, 'execution owner')

    if kind == 'authorization_abort_execution_crash':
        _require_present_matrix(
            payload, ('execution_crash_attestation_hmac',), 'crash attestation'
        )
    else:
        _require_null_matrix(
            payload, ('execution_crash_attestation_hmac',), 'crash attestation'
        )

    case_required = kind in {
        'case_claim',
        'component_claim',
        'component_outcome',
        'case_failure',
        'case_safe_outcome',
        'case_outcome',
        'authorization_abort_control',
        'authorization_abort_component',
        'authorization_abort_component_snapshot',
    }
    case_optional = kind in {
        'authorization_abort_corpus_drift',
        'authorization_abort_execution_crash',
    }
    has_case = payload['case_id_hmac'] is not None
    if case_required and not has_case:
        raise RagReleaseLedgerError('case required matrix is invalid')
    if not case_required and not case_optional and has_case:
        raise RagReleaseLedgerError('case null matrix is invalid')
    if not has_case:
        _require_null_matrix(payload, _CASE_FIELDS, 'case')
    else:
        _require_present_matrix(
            payload,
            (
                'case_embedding_reserved_cost_usd',
                'case_generation_reserved_cost_usd',
                'case_id_hmac',
                'case_state_after',
                'case_total_reserved_cost_usd',
                'runtime_agent_run_id_hmac',
            ),
            'case',
        )
        total = Decimal(str(payload['case_total_reserved_cost_usd']))
        parts = Decimal(str(payload['case_embedding_reserved_cost_usd'])) + Decimal(
            str(payload['case_generation_reserved_cost_usd'])
        )
        if total != parts:
            raise RagReleaseLedgerError('case cost matrix is invalid')
        case_state_pair = (
            payload['case_state_before'],
            payload['case_state_after'],
        )
        if kind == 'case_claim':
            allowed_case_states = {(None, 'claimed')}
        elif kind in {'component_claim', 'component_outcome'}:
            allowed_case_states = {('claimed', 'claimed')}
        elif kind in {'case_safe_outcome', 'case_outcome'}:
            allowed_case_states = {('claimed', 'complete')}
        else:
            allowed_case_states = {('claimed', 'failed')}
        if case_state_pair not in allowed_case_states:
            raise RagReleaseLedgerError('case state matrix is invalid')
        if kind in {'case_safe_outcome', 'case_outcome'}:
            _require_present_matrix(
                payload, ('case_projection_hmac',), 'case projection'
            )
        else:
            _require_null_matrix(payload, ('case_projection_hmac',), 'case projection')

    component_required = kind in {
        'component_claim',
        'component_outcome',
        'authorization_abort_component',
        'authorization_abort_component_snapshot',
    }
    component_optional = kind in {
        'case_failure',
        'authorization_abort_corpus_drift',
        'authorization_abort_execution_crash',
    }
    has_component = payload['component'] is not None
    if component_required and not has_component:
        raise RagReleaseLedgerError('dispatch required matrix is invalid')
    if has_component and not (component_required or component_optional):
        raise RagReleaseLedgerError('dispatch null matrix is invalid')
    if not has_component:
        _require_null_matrix(payload, _DISPATCH_FIELDS, 'dispatch')
    else:
        _require_present_matrix(payload, _DISPATCH_FIELDS, 'dispatch')
        if payload['component'] not in {'query_embedding', 'answer_generation'}:
            raise RagReleaseLedgerError('dispatch component matrix is invalid')
        dispatch_pair = (
            payload['dispatch_state_before'],
            payload['dispatch_state_after'],
            payload['dispatch_count_before'],
            payload['dispatch_count_after'],
        )
        if kind == 'component_claim':
            if dispatch_pair != ('not_attempted', 'dispatching', 0, 1):
                raise RagReleaseLedgerError('dispatch claim matrix is invalid')
            if (
                payload['charge_basis_after'] != 'reserved'
                or payload['charged_cost_usd'] != payload['reserved_cost_usd']
            ):
                raise RagReleaseLedgerError('dispatch charge matrix is invalid')
        elif dispatch_pair != ('dispatching', 'terminal', 1, 1):
            raise RagReleaseLedgerError('dispatch outcome matrix is invalid')
        elif payload['charge_basis_after'] not in {'reserved', 'actual'}:
            raise RagReleaseLedgerError('dispatch charge matrix is invalid')

    if kind in {'authorization_complete', 'authorization_finish_quality_failed'}:
        _require_present_matrix(payload, ('quality_report_hmac',), 'quality report')
    elif kind != 'authorization_finish_failed':
        _require_null_matrix(payload, ('quality_report_hmac',), 'quality report')
    if kind in {'authorization_complete', 'authorization_finish_quality_failed'} and (
        payload['case_claim_count'],
        payload['embedding_dispatch_count'],
        payload['generation_dispatch_count'],
        payload['total_dispatch_count'],
    ) != (30, 10, 30, 40):
        raise RagReleaseLedgerError('terminal aggregate matrix is invalid')
    if kind == 'authorization_finish_failed':
        counts = (
            payload['case_claim_count'],
            payload['embedding_dispatch_count'],
            payload['generation_dispatch_count'],
            payload['total_dispatch_count'],
        )
        if counts[0] != 30:
            raise RagReleaseLedgerError('terminal aggregate matrix is invalid')
        if payload['outcome'] == 'execution_contract_failed' and counts == (
            30,
            10,
            30,
            40,
        ):
            raise RagReleaseLedgerError('terminal aggregate matrix is invalid')
    if has_case and payload['case_claim_count'] == 0:
        raise RagReleaseLedgerError('case aggregate matrix is invalid')

    affected = [row_kind for row_kind, _identity in rows]
    observed = [row_kind for row_kind, _identity, _projection in observations]
    if (
        affected.count('release_ledger') != 1
        or affected.count('release_transition') != 1
    ):
        raise RagReleaseLedgerError('affected row matrix is invalid')
    if any(kind in observed for kind in {'release_ledger', 'release_transition'}):
        raise RagReleaseLedgerError('observation row matrix is invalid')

    def _require_peer(row_kind: str, *, mode: str) -> None:
        actual = affected.count(row_kind)
        read_only = observed.count(row_kind)
        valid = (
            (actual == 1 and read_only == 0)
            if mode == 'affected'
            else (actual == 0 and read_only == 1)
            if mode == 'observed'
            else (actual + read_only == 1)
        )
        if not valid:
            raise RagReleaseLedgerError(f'{row_kind} peer matrix is invalid')

    authorization_mode = (
        'observed'
        if kind in {'case_safe_outcome', 'case_outcome'}
        else 'either'
        if kind in {'component_outcome', 'case_failure'}
        else 'affected'
    )
    _require_peer('authorization', mode=authorization_mode)

    case_mutation = int(
        has_case and kind not in {'component_claim', 'component_outcome'}
    )
    parent_mutation = int(
        has_case
        and (
            kind not in {'component_claim', 'component_outcome'}
            or kind == 'component_outcome'
            and payload['component'] == 'answer_generation'
        )
    )
    for row_kind, mutations, total in (
        ('case', case_mutation, payload['case_claim_count']),
        ('agent_run', parent_mutation, payload['case_claim_count']),
        ('dispatch', int(has_component), payload['total_dispatch_count']),
    ):
        if (
            affected.count(row_kind) != mutations
            or affected.count(row_kind) + observed.count(row_kind) != total
        ):
            raise RagReleaseLedgerError(
                f'terminal roster {row_kind} peer matrix is invalid'
            )

    cost_range = {
        'case_claim': (2, 2),
        'component_claim': (1, 1),
        'component_outcome': (1, 1),
        'case_failure': (0, 2),
        'case_safe_outcome': (1, 2),
        'authorization_abort_control': (0, 2),
        'authorization_abort_component': (1, 2),
        'authorization_abort_component_snapshot': (1, 2),
        'authorization_abort_corpus_drift': (0, 2),
        'authorization_abort_execution_crash': (0, 2),
    }.get(str(kind), (0, 0))
    cost_count = affected.count('cost_component')
    if not cost_range[0] <= cost_count <= cost_range[1] or (
        cost_count + observed.count('cost_component') != 2 * payload['case_claim_count']
    ):
        raise RagReleaseLedgerError('cost component matrix is invalid')
    if kind == 'authorization_abort_component':
        _require_peer('provider_safety_authority', mode='affected')
        if (
            affected.count('provider_readiness') != 1
            or observed.count('provider_readiness') < 1
        ):
            raise RagReleaseLedgerError('provider readiness peer matrix is invalid')
    else:
        _require_peer('provider_safety_authority', mode='observed')
        if (
            affected.count('provider_readiness')
            or observed.count('provider_readiness') < 2
        ):
            raise RagReleaseLedgerError('provider readiness peer matrix is invalid')

    quality_expected = payload['quality_report_hmac'] is not None
    if affected.count('quality_report') != int(quality_expected) or observed.count(
        'quality_report'
    ):
        raise RagReleaseLedgerError('quality report row matrix is invalid')

    allowed = {
        'authorization',
        'case',
        'dispatch',
        'agent_run',
        'cost_component',
        'provider_safety_authority',
        'provider_readiness',
        'quality_report',
        'release_ledger',
        'release_transition',
    }
    if set(affected + observed) - allowed:
        raise RagReleaseLedgerError('release peer matrix is invalid')


def validate_transition_payload(
    payload: Mapping[str, object],
    *,
    identity_secret: bytes,
) -> ValidatedReleaseTransition:
    if type(payload) is not dict or set(payload) != _TRANSITION_KEYS:
        raise RagReleaseLedgerError('transition payload keys are invalid')
    if type(identity_secret) is not bytes or len(identity_secret) < 32:
        raise RagReleaseLedgerError('transition signer is unavailable')
    kind = payload['transition_kind']
    if type(kind) is not str or kind not in _OUTCOMES:
        raise RagReleaseLedgerError('transition kind is invalid')
    if payload['outcome'] not in _OUTCOMES[kind]:
        raise RagReleaseLedgerError('transition outcome is invalid')
    try:
        ledger_uuid = UUID(str(payload['ledger_uuid']))
    except (TypeError, ValueError) as exc:
        raise RagReleaseLedgerError('transition ledger identity is invalid') from exc
    if str(ledger_uuid) != payload['ledger_uuid']:
        raise RagReleaseLedgerError('transition ledger identity is invalid')
    epoch = payload['ledger_epoch']
    if type(epoch) is not int or epoch <= 0:
        raise RagReleaseLedgerError('transition ledger epoch is invalid')
    from_generation = payload['from_generation']
    to_generation = payload['to_generation']
    if (
        type(from_generation) is not int
        or from_generation < 0
        or type(to_generation) is not int
        or to_generation != from_generation + 1
    ):
        raise RagReleaseLedgerError('transition generation is not gapless')
    _require_count(payload['case_claim_count'], key='case claim', maximum=30)
    embedding_count = _require_count(
        payload['embedding_dispatch_count'], key='embedding dispatch', maximum=10
    )
    generation_count = _require_count(
        payload['generation_dispatch_count'], key='generation dispatch', maximum=30
    )
    total_count = _require_count(
        payload['total_dispatch_count'], key='total dispatch', maximum=40
    )
    if total_count != embedding_count + generation_count:
        raise RagReleaseLedgerError('dispatch count total is invalid')
    for key in ('dispatch_count_before', 'dispatch_count_after'):
        if payload[key] is not None and payload[key] not in {0, 1}:
            raise RagReleaseLedgerError('dispatch count is invalid')
        if type(payload[key]) is bool:
            raise RagReleaseLedgerError('dispatch count is invalid')
    for key in _MONEY_KEYS:
        value = payload[key]
        if value is not None and (
            type(value) is not str or _MONEY.fullmatch(value) is None
        ):
            raise RagReleaseLedgerError(f'{key} must be a six-place decimal')
    for key in _HMAC_KEYS:
        value = payload[key]
        if value is not None:
            require_lower_hmac(value)
    before = payload['authorization_state_before']
    after = payload['authorization_state_after']
    if before is not None and before not in _AUTHORIZATION_STATES:
        raise RagReleaseLedgerError('authorization state is invalid')
    if after not in _AUTHORIZATION_STATES:
        raise RagReleaseLedgerError('authorization state is invalid')
    approved = payload['approved_corpus_snapshot_hmac']
    current = payload['current_corpus_snapshot_hmac']
    if kind == 'authorization_abort_corpus_drift':
        if approved == current:
            raise RagReleaseLedgerError(
                'corpus drift transition requires different snapshots'
            )
    elif approved != current:
        raise RagReleaseLedgerError('current corpus snapshot differs from approval')
    rows = _validate_affected_rows(payload['affected_rows'])
    observations = _validate_observation_set(payload['observation_set'])
    affected_identities = {(kind, identity) for kind, identity in rows}
    observed_identities = {
        (kind, identity) for kind, identity, _projection in observations
    }
    if affected_identities & observed_identities:
        raise RagReleaseLedgerError('affected and observed rows overlap')
    derived = _derived_payload_rows(payload, identity_secret=identity_secret)
    bound_identities = affected_identities | observed_identities
    if any(item not in bound_identities for item in derived):
        raise RagReleaseLedgerError(
            'affected or observed release row identity is unbound'
        )
    if kind == 'authorization_bootstrap':
        _validate_bootstrap(payload)
    _validate_transition_matrix(payload, rows, observations)
    canonical = canonical_json_bytes(dict(payload))
    digest = rag_identity_hmac(
        dict(payload),
        secret=identity_secret,
        schema_version='rag-release-ledger-transition:v2',
        policy_version='rag-live-gate:v1',
    )
    return ValidatedReleaseTransition(
        transition_kind=kind,
        from_generation=from_generation,
        to_generation=to_generation,
        affected_rows=rows,
        observation_set=observations,
        payload_canonical_bytes=canonical,
        transition_digest=digest,
    )


class RagReleaseLedger:
    def __init__(
        self,
        *,
        authority: RagReleaseAuthority,
        identity_secret: bytes,
    ) -> None:
        if type(authority) is not RagReleaseAuthority:
            raise TypeError('release authority is required')
        if type(identity_secret) is not bytes or len(identity_secret) < 32:
            raise ValueError('release ledger signer is unavailable')
        self._authority = authority
        self._secret = identity_secret

    @staticmethod
    def mutation_set(connection: Connection) -> RagReleaseMutationSet:
        return RagReleaseMutationSet(connection, seal=_MUTATION_SET_SEAL)

    def _assert_database_roster(
        self,
        connection: Connection,
        payload: Mapping[str, object],
        actual_mutations: RagReleaseMutationSet | None = None,
    ) -> None:
        """Substantiate aggregate transition fields from the locked DB roster."""
        tables = release_tables(build_rag_release_metadata())
        scope = (
            tables.cases.c.ledger_uuid == payload['ledger_uuid'],
            tables.cases.c.ledger_epoch == payload['ledger_epoch'],
            tables.cases.c.approval_id_hmac == payload['approval_id_hmac'],
        )
        case_statement = select(tables.cases).where(*scope)
        dispatch_statement = select(tables.dispatches).where(
            tables.dispatches.c.ledger_uuid == payload['ledger_uuid'],
            tables.dispatches.c.ledger_epoch == payload['ledger_epoch'],
            tables.dispatches.c.approval_id_hmac == payload['approval_id_hmac'],
        )
        report_statement = select(tables.quality_reports).where(
            tables.quality_reports.c.ledger_uuid == payload['ledger_uuid'],
            tables.quality_reports.c.ledger_epoch == payload['ledger_epoch'],
            tables.quality_reports.c.approval_id_hmac == payload['approval_id_hmac'],
        )
        if connection.dialect.name == 'postgresql':
            case_statement = case_statement.with_for_update()
            dispatch_statement = dispatch_statement.with_for_update()
            report_statement = report_statement.with_for_update()
        cases = [dict(row) for row in connection.execute(case_statement).mappings()]
        dispatches = [
            dict(row) for row in connection.execute(dispatch_statement).mappings()
        ]
        reports = [dict(row) for row in connection.execute(report_statement).mappings()]
        if len(cases) != payload['case_claim_count'] or sorted(
            row['manifest_ordinal'] for row in cases
        ) != list(range(len(cases))):
            raise RagReleaseLedgerError('terminal roster case count differs')
        dispatch_counts = {
            component: sum(
                int(row['dispatch_count'])
                for row in dispatches
                if row['component'] == component
            )
            for component in ('query_embedding', 'answer_generation')
        }
        if (
            dispatch_counts['query_embedding'] != payload['embedding_dispatch_count']
            or dispatch_counts['answer_generation']
            != payload['generation_dispatch_count']
            or sum(dispatch_counts.values()) != payload['total_dispatch_count']
        ):
            raise RagReleaseLedgerError('terminal roster dispatch count differs')
        case_reserved = sum(
            (Decimal(str(row['total_reserved_cost_usd'])) for row in cases),
            Decimal('0.000000'),
        )
        dispatch_charged = sum(
            (Decimal(str(row['charged_cost_usd'])) for row in dispatches),
            Decimal('0.000000'),
        )
        if case_reserved != Decimal(
            str(payload['authorization_reserved_cost_usd'])
        ) or dispatch_charged != Decimal(
            str(payload['authorization_charged_cost_usd'])
        ):
            raise RagReleaseLedgerError('terminal roster cost aggregate differs')

        expected_report = payload['quality_report_hmac']
        if expected_report is None:
            if reports:
                raise RagReleaseLedgerError('terminal roster report differs')
        elif len(reports) != 1 or reports[0]['quality_report_hmac'] != expected_report:
            raise RagReleaseLedgerError('terminal roster report differs')

        terminal_kind = str(payload['transition_kind']).startswith(
            'authorization_abort_'
        ) or payload['transition_kind'] in {
            'authorization_complete',
            'authorization_finish_failed',
            'authorization_finish_quality_failed',
        }
        if terminal_kind and any(row['state'] == 'claimed' for row in cases):
            raise RagReleaseLedgerError('terminal roster contains claimed case')
        if payload['transition_kind'] in {
            'authorization_complete',
            'authorization_finish_quality_failed',
        } and (
            len(cases) != 30
            or any(row['state'] != 'complete' for row in cases)
            or dispatch_counts != {'query_embedding': 10, 'answer_generation': 30}
            or any(row['state'] != 'terminal' for row in dispatches)
        ):
            raise RagReleaseLedgerError('terminal roster is incomplete')
        if payload['transition_kind'] == 'authorization_finish_failed':
            if len(cases) != 30 or any(
                row['state'] not in {'complete', 'failed'} for row in cases
            ):
                raise RagReleaseLedgerError('terminal roster is incomplete')
            if payload['outcome'] == 'ordinary_execution_failed' and not any(
                row['state'] == 'failed' for row in cases
            ):
                raise RagReleaseLedgerError('terminal roster lacks failed case')

        if actual_mutations is None:
            raise RagReleaseLedgerError('terminal roster observation set is required')
        for row_kind in ('provider_safety_authority', 'provider_readiness'):
            table, _aliases = actual_mutations._table(row_kind)
            statement = select(table)
            if connection.dialect.name == 'postgresql':
                statement = statement.with_for_update()
            actual_mutations._assert_roster_rows(
                row_kind,
                [dict(row) for row in connection.execute(statement).mappings()],
            )
        for row_kind, roster in (
            ('case', cases),
            ('dispatch', dispatches),
            ('quality_report', reports),
        ):
            actual_mutations._assert_roster_rows(row_kind, roster)

        if not cases:
            actual_mutations._assert_roster_rows('agent_run', [])
            actual_mutations._assert_roster_rows('cost_component', [])
            return
        from backend.app.models.agent_runs import AgentRun
        from backend.app.models.rag_runtime import AgentRunCostComponent

        run_statement = select(AgentRun.__table__).where(
            AgentRun.__table__.c.id.in_(
                [row['id'] for row in actual_mutations._captured_rows('agent_run')]
            )
        )
        if connection.dialect.name == 'postgresql':
            run_statement = run_statement.with_for_update()
        candidate_runs = [
            dict(row) for row in connection.execute(run_statement).mappings()
        ]
        actual_mutations._assert_roster_rows('agent_run', candidate_runs)
        runs_by_hmac: dict[str, dict[str, object]] = {}
        for run in candidate_runs:
            run_hmac = rag_identity_hmac(
                {'agent_run_id': run['id']},
                secret=self._secret,
                schema_version='rag-runtime-agent-run-id:v1',
                policy_version='rag-run:v2',
            )
            if run_hmac in runs_by_hmac:
                raise RagReleaseLedgerError('terminal roster runtime identity collides')
            runs_by_hmac[run_hmac] = run
        required_hmacs = {str(row['runtime_agent_run_id_hmac']) for row in cases}
        if len(required_hmacs) != len(cases) or required_hmacs != set(runs_by_hmac):
            raise RagReleaseLedgerError('terminal roster runtime parent is missing')
        run_ids = [runs_by_hmac[item]['id'] for item in required_hmacs]
        cost_statement = select(AgentRunCostComponent.__table__).where(
            AgentRunCostComponent.__table__.c.agent_run_id.in_(run_ids)
        )
        if connection.dialect.name == 'postgresql':
            cost_statement = cost_statement.with_for_update()
        costs = [dict(row) for row in connection.execute(cost_statement).mappings()]
        actual_mutations._assert_roster_rows('cost_component', costs)
        costs_by_run: dict[int, list[dict[str, object]]] = {}
        for row in costs:
            costs_by_run.setdefault(int(row['agent_run_id']), []).append(row)
        if any(
            {row['component'] for row in costs_by_run.get(int(run_id), [])}
            != {'query_embedding', 'answer_generation'}
            or len(costs_by_run.get(int(run_id), [])) != 2
            for run_id in run_ids
        ):
            raise RagReleaseLedgerError('terminal roster runtime costs are incomplete')
        cases_by_run = {str(row['runtime_agent_run_id_hmac']): row for row in cases}
        for run_hmac in required_hmacs:
            run = runs_by_hmac[run_hmac]
            case = cases_by_run[run_hmac]
            components = costs_by_run[int(run['id'])]
            component_map = {str(row['component']): row for row in components}
            if any(
                Decimal(str(component_map[component]['reserved_cost_usd']))
                != Decimal(str(case[column]))
                and not (
                    component_map[component]['dispatch_state'] == 'terminal'
                    and component_map[component]['attempted'] is False
                    and Decimal(str(component_map[component]['reserved_cost_usd'])) == 0
                )
                for component, column in (
                    ('query_embedding', 'embedding_reserved_cost_usd'),
                    ('answer_generation', 'generation_reserved_cost_usd'),
                )
            ) or (
                run['run_record_phase'] != 'admission'
                and sum(
                    (Decimal(str(row['charged_cost_usd'])) for row in components),
                    Decimal('0.000000'),
                )
                != Decimal(str(run['total_charged_cost_usd']))
            ):
                raise RagReleaseLedgerError('terminal roster runtime cost differs')
            if run['run_contract_version'] != 'rag-run:v2':
                raise RagReleaseLedgerError('terminal roster runtime contract differs')
            if run['run_record_phase'] == 'cost_finalized_pending_projection' and (
                run['projection_owner_fence_hmac']
                != payload['execution_runner_fence_hmac']
                or any(row['dispatch_state'] != 'terminal' for row in components)
            ):
                raise RagReleaseLedgerError('terminal roster pending parent differs')
            for child in components:
                paired = [
                    row
                    for row in dispatches
                    if row['case_id_hmac'] == case['case_id_hmac']
                    and row['component'] == child['component']
                ]
                if child['dispatch_count'] == 0:
                    if (
                        paired
                        or child['process_instance_hmac'] is not None
                        or child['dispatch_fence_hmac'] is not None
                        or Decimal(str(child['charged_cost_usd'])) != 0
                    ):
                        raise RagReleaseLedgerError(
                            'terminal roster zero child differs'
                        )
                elif (
                    len(paired) != 1
                    or child['process_instance_hmac']
                    != payload['execution_process_instance_hmac']
                    or any(
                        child[key] != paired[0][key]
                        for key in (
                            'dispatch_count',
                            'dispatch_fence_hmac',
                            'reserved_cost_usd',
                            'charged_cost_usd',
                            'charge_basis',
                        )
                    )
                    or paired[0]['state']
                    != (
                        'terminal'
                        if child['dispatch_state'] == 'abandoned_unknown'
                        else child['dispatch_state']
                    )
                ):
                    raise RagReleaseLedgerError(
                        'terminal roster dispatch owner or pairing differs'
                    )
            if case['state'] == 'claimed' and (
                run['status'] != 'running'
                or run['run_record_phase']
                not in {'admission', 'cost_finalized_pending_projection'}
            ):
                raise RagReleaseLedgerError('terminal roster claimed run differs')
            if case['state'] == 'complete' and (
                run['status'] != 'complete'
                or run['run_record_phase'] != 'final'
                or any(row['dispatch_state'] != 'terminal' for row in components)
            ):
                raise RagReleaseLedgerError('terminal roster complete run differs')
            if case['state'] == 'failed' and (
                run['status'] != 'failed'
                or run['run_record_phase'] not in {'final', 'admission_only'}
                or any(
                    row['dispatch_state'] not in {'terminal', 'abandoned_unknown'}
                    for row in components
                )
            ):
                raise RagReleaseLedgerError('terminal roster failed run differs')
        runtime_counts = {
            component: sum(
                int(row['dispatch_count'])
                for row in costs
                if row['component'] == component
            )
            for component in ('query_embedding', 'answer_generation')
        }
        runtime_charged = sum(
            (Decimal(str(row['charged_cost_usd'])) for row in costs),
            Decimal('0.000000'),
        )
        if runtime_counts != dispatch_counts or runtime_charged != dispatch_charged:
            raise RagReleaseLedgerError('terminal roster runtime aggregate differs')
        if terminal_kind and (
            any(
                runs_by_hmac[item]['run_record_phase']
                not in {'final', 'admission_only'}
                for item in required_hmacs
            )
            or any(
                row['dispatch_state'] not in {'terminal', 'abandoned_unknown'}
                for row in costs
            )
        ):
            raise RagReleaseLedgerError('terminal roster runtime is nonterminal')

    def append(
        self,
        connection: Connection,
        payload: Mapping[str, object],
        *,
        actual_mutations: RagReleaseMutationSet,
        database_identity: ValidationDatabaseIdentity | None = None,
        provider_incident: object = None,
        approved_case_claim: object = None,
    ) -> RagReleaseSnapshot:
        validated = validate_transition_payload(payload, identity_secret=self._secret)
        from backend.app.admin.rag_provider_safety import (
            RagProviderSafetyIncidentPlan,
        )

        if provider_incident is not None and (
            type(provider_incident) is not RagProviderSafetyIncidentPlan
            or provider_incident._seal is None
        ):
            raise RagReleaseLedgerError('provider incident capability is invalid')
        if (payload['transition_kind'] == 'authorization_abort_component') != (
            provider_incident is not None
        ):
            raise RagReleaseLedgerError('provider incident capability is required')
        if provider_incident is not None:
            expected_run_hmac = rag_identity_hmac(
                {'agent_run_id': provider_incident.agent_run_id},
                secret=self._secret,
                schema_version='rag-runtime-agent-run-id:v1',
                policy_version='rag-run:v2',
            )
            if (
                provider_incident.component != payload['component']
                or provider_incident.category != payload['outcome']
                or expected_run_hmac != payload['runtime_agent_run_id_hmac']
                or Decimal(str(provider_incident.cost_usd))
                != Decimal(str(payload['charged_cost_usd']))
                or provider_incident.new_envelope_digest
                != payload['provider_safety_envelope_digest']
            ):
                raise RagReleaseLedgerError('provider incident capability differs')
        if (
            type(actual_mutations) is not RagReleaseMutationSet
            or actual_mutations._seal is not _MUTATION_SET_SEAL
            or actual_mutations._connection is not connection
        ):
            raise RagReleaseLedgerError(
                'same-transaction release mutation set is required'
            )
        marker = DurableFileAuthority.open_runtime(self._authority.marker_path)
        try:
            with self._authority._authority_barrier(
                connection, marker=marker
            ) as barrier_guard:
                _body, current = self._authority._parse(marker._read_bytes_unlocked())
                self._authority._inspect_locked(
                    connection,
                    current,
                    database_identity=database_identity,
                )
                if (
                    payload['ledger_uuid'] != str(current.ledger_uuid)
                    or payload['ledger_epoch'] != current.ledger_epoch
                    or payload['validation_database_identity_hmac']
                    != current.validation_database_identity_hmac
                    or validated.from_generation != current.generation
                ):
                    raise RagReleaseLedgerError(
                        'release transition generation CAS failed'
                    )
                actual_mutations._capture_observations_under_barrier(
                    connection, authority=self._authority, barrier_guard=barrier_guard
                )
                actual_mutations._assert_observation_projection(
                    payload, identity_secret=self._secret
                )
                actual_mutations._preflight_runtime_mutations(
                    payload,
                    identity_secret=self._secret,
                    approved_case_claim=approved_case_claim,
                    barrier_guard=barrier_guard,
                )
                if provider_incident is not None:
                    authorization_before = actual_mutations._snapshot(
                        connection,
                        ReleaseRowPrimaryKey(
                            'authorization',
                            {
                                'ledger_uuid': payload['ledger_uuid'],
                                'ledger_epoch': payload['ledger_epoch'],
                                'approval_id_hmac': payload['approval_id_hmac'],
                            },
                        ),
                        for_update=True,
                    )
                    if (
                        authorization_before is None
                        or authorization_before['provider_safety_envelope_digest']
                        != provider_incident._prepared.old_envelope_digest
                    ):
                        raise RagReleaseLedgerError(
                            'provider incident approved before-image differs'
                        )
                    incident_evidence = barrier_guard.apply_provider_incident(
                        provider_incident
                    )
                    if (
                        incident_evidence.new_body['envelope_digest']
                        != payload['provider_safety_envelope_digest']
                    ):
                        raise RagReleaseLedgerError(
                            'provider incident envelope differs from payload'
                        )
                    actual_mutations._capture_provider_incident(incident_evidence)
                actual_mutations._execute_under_barrier(
                    connection,
                    authority=self._authority,
                    barrier_guard=barrier_guard,
                )
                actual_mutations.assert_current(connection)
                actual_mutations.assert_payload_projection(
                    payload,
                    identity_secret=self._secret,
                    approved_case_claim=approved_case_claim,
                    barrier_guard=barrier_guard,
                )
                self._assert_database_roster(connection, payload, actual_mutations)
                derived_actual = tuple(
                    sorted(
                        (
                            row.row_kind,
                            release_row_identity_hmac(
                                row.row_kind,
                                row.primary_key,
                                identity_secret=self._secret,
                            ),
                        )
                        for row in actual_mutations.rows
                    )
                )
                internal = tuple(
                    row
                    for row in validated.affected_rows
                    if row[0] in {'release_ledger', 'release_transition'}
                )
                derived_actual = tuple(sorted((*derived_actual, *internal)))
                if (
                    len(derived_actual) != len(actual_mutations.rows) + len(internal)
                    or derived_actual != validated.affected_rows
                ):
                    raise RagReleaseLedgerError(
                        'actual affected rows differ from payload'
                    )
                next_body = self._authority._body(
                    ledger_uuid=current.ledger_uuid,
                    ledger_epoch=current.ledger_epoch,
                    generation=validated.to_generation,
                    last_transition_digest=validated.transition_digest,
                    predecessor_marker_digest=current.predecessor_marker_digest,
                    rebootstrap_reason_hmac=current.rebootstrap_reason_hmac,
                    database_identity_hmac=(current.validation_database_identity_hmac),
                    database_locator_hmac=(current.validation_database_locator_hmac),
                    review_envelope_hmac=(current.bootstrap_review_envelope_hmac),
                    review_nonce_hmac=current.bootstrap_review_nonce_hmac,
                    review_operation=current.bootstrap_operation,
                )
                next_envelope = self._authority._wrap(next_body)
                _next_body, next_snapshot = self._authority._parse(
                    canonical_json_bytes(next_envelope)
                )
                tables = release_tables(build_rag_release_metadata())
                marker._replace_unlocked(next_envelope)
                if self._authority._after_marker_replace is not None:
                    self._authority._after_marker_replace()
                result = connection.execute(
                    update(tables.ledgers)
                    .where(
                        tables.ledgers.c.ledger_uuid == str(current.ledger_uuid),
                        tables.ledgers.c.ledger_epoch == current.ledger_epoch,
                        tables.ledgers.c.generation == current.generation,
                        tables.ledgers.c.last_transition_digest
                        == current.last_transition_digest,
                        tables.ledgers.c.marker_file_digest
                        == current.marker_file_digest,
                    )
                    .values(
                        generation=next_snapshot.generation,
                        last_transition_digest=next_snapshot.last_transition_digest,
                        marker_file_digest=next_snapshot.marker_file_digest,
                    )
                )
                if result.rowcount != 1:
                    raise RagReleaseLedgerError(
                        'release transition generation CAS failed'
                    )
                transition_result = connection.execute(
                    insert(tables.transitions).values(
                        ledger_uuid=str(current.ledger_uuid),
                        ledger_epoch=current.ledger_epoch,
                        generation=validated.to_generation,
                        transition_kind=validated.transition_kind,
                        transition_digest=validated.transition_digest,
                        payload_canonical_bytes=validated.payload_canonical_bytes,
                    )
                )
                if transition_result.rowcount != 1:
                    raise RagReleaseLedgerError('release transition insert failed')
                connection.commit()
                return next_snapshot
        except DurableFileAuthorityError as exc:
            connection.rollback()
            raise RagReleaseLedgerError('release authority lock failed') from exc
        except Exception:
            connection.rollback()
            raise
