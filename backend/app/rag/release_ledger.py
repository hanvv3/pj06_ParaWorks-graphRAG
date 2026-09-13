from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias
from uuid import UUID

from sqlalchemy import Connection, insert, update

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
    'case_safe_outcome': frozenset(
        {'no_match', 'hidden_only', 'safety_filter_empty'}
    ),
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
    'authorization_abort_corpus_drift': frozenset(
        {'live_corpus_snapshot_changed'}
    ),
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
_MONEY = re.compile(r'^(?:0|[1-9][0-9]*)\.[0-9]{6}$', re.ASCII)
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
        if type(item) is not dict or set(item) != {
            'row_identity_hmac',
            'row_kind',
        }:
            raise RagReleaseLedgerError('affected row keys are invalid')
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
    payload: Mapping[str, object], rows: tuple[AffectedRow, ...]
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
        kind in {
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

    row_kinds = [row_kind for row_kind, _identity in rows]
    if kind == 'authorization_bootstrap':
        if sorted(row_kinds) != [
            'authorization',
            'release_ledger',
            'release_transition',
        ]:
            raise RagReleaseLedgerError('affected row matrix is invalid')
    elif not {
        'authorization',
        'release_ledger',
        'release_transition',
    } <= set(row_kinds):
        raise RagReleaseLedgerError('affected row matrix is invalid')
    if has_case and not {'case', 'agent_run'} <= set(row_kinds):
        raise RagReleaseLedgerError('affected row matrix is invalid')
    if component_required and not {'dispatch', 'cost_component'} <= set(row_kinds):
        raise RagReleaseLedgerError('affected row matrix is invalid')
    if kind == 'case_claim' and row_kinds.count('cost_component') != 2:
        raise RagReleaseLedgerError('affected row matrix is invalid')
    if kind in {'authorization_complete', 'authorization_finish_quality_failed'} and (
        'quality_report' not in row_kinds
    ):
        raise RagReleaseLedgerError('affected row matrix is invalid')


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
            raise RagReleaseLedgerError('corpus drift transition requires different snapshots')
    elif approved != current:
        raise RagReleaseLedgerError('current corpus snapshot differs from approval')
    rows = _validate_affected_rows(payload['affected_rows'])
    if kind == 'authorization_bootstrap':
        _validate_bootstrap(payload)
    _validate_transition_matrix(payload, rows)
    canonical = canonical_json_bytes(dict(payload))
    digest = rag_identity_hmac(
        dict(payload),
        secret=identity_secret,
        schema_version='rag-release-ledger-transition:v1',
        policy_version='rag-live-gate:v1',
    )
    return ValidatedReleaseTransition(
        transition_kind=kind,
        from_generation=from_generation,
        to_generation=to_generation,
        affected_rows=rows,
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

    def append(
        self,
        connection: Connection,
        payload: Mapping[str, object],
        *,
        actual_affected_rows: Sequence[AffectedRow],
        database_identity: ValidationDatabaseIdentity | None = None,
    ) -> RagReleaseSnapshot:
        validated = validate_transition_payload(payload, identity_secret=self._secret)
        if tuple(actual_affected_rows) != validated.affected_rows:
            raise RagReleaseLedgerError('actual affected rows differ from payload')
        marker = DurableFileAuthority.open_runtime(self._authority.marker_path)
        try:
            with marker.locked(), self._authority._registered_advisory(connection):
                _body, current = self._authority._parse(
                    marker._read_bytes_unlocked()
                )
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
                next_body = self._authority._body(
                    ledger_uuid=current.ledger_uuid,
                    ledger_epoch=current.ledger_epoch,
                    generation=validated.to_generation,
                    last_transition_digest=validated.transition_digest,
                    predecessor_marker_digest=current.predecessor_marker_digest,
                    rebootstrap_reason_hmac=current.rebootstrap_reason_hmac,
                    database_identity_hmac=(
                        current.validation_database_identity_hmac
                    ),
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
                        tables.ledgers.c.ledger_uuid
                        == str(current.ledger_uuid),
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
                    raise RagReleaseLedgerError(
                        'release transition insert failed'
                    )
                connection.commit()
                return next_snapshot
        except DurableFileAuthorityError as exc:
            raise RagReleaseLedgerError('release authority lock failed') from exc
