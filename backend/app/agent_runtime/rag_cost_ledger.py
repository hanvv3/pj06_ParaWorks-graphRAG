from __future__ import annotations

import hmac
import secrets
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_runtime_contracts import (
    ADMISSION_SOURCE_WINDOWS,
    AuthorizedProviderPolicySnapshot,
    CommittedRagDispatchGrant,
    RagComponentFinal,
    RagRunAdmission,
    RagRunTerminal,
    StrictProviderOutcome,
)
from backend.app.agent_runtime.rag_safety_identity import (
    admission_identity,
    dispatch_fence_identity,
    process_instance_identity,
    provider_safety_snapshot_identity,
    rag_identity_hmac,
    require_lower_hmac,
    runtime_cost_identity,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.app.rag.retrieval import PreparedPaidCallBudget, RagPaidComponent

_ZERO = Decimal('0.000000')
_COMPONENT_ORDER = ('query_embedding', 'answer_generation')


class RagCostLedgerError(RuntimeError):
    """A fail-closed durable cost authority refusal."""


def _make_grant(
    *,
    agent_run_id: int,
    component: RagPaidComponent,
    reserved_cost_usd: Decimal,
    dispatch_fence_hmac: str,
    provider_safety_snapshot_hmac: str,
    on_consume: Callable[[], None],
) -> CommittedRagDispatchGrant:
    lock = threading.Lock()
    consumed = False

    class _Grant:
        __slots__ = ()

        @property
        def agent_run_id(self) -> int:
            return agent_run_id

        @property
        def component(self) -> RagPaidComponent:
            return component

        @property
        def reserved_cost_usd(self) -> Decimal:
            return reserved_cost_usd

        @property
        def dispatch_fence_hmac(self) -> str:
            return dispatch_fence_hmac

        @property
        def provider_safety_snapshot_hmac(self) -> str:
            return provider_safety_snapshot_hmac

        @property
        def consumed(self) -> bool:
            with lock:
                return consumed

        def consume_at_dispatch(self) -> None:
            nonlocal consumed
            with lock:
                if consumed:
                    raise RagCostLedgerError('dispatch grant is already consumed')
                on_consume()
                consumed = True

        def __copy__(self):
            raise TypeError('dispatch grants cannot be copied')

        def __deepcopy__(self, _memo: object):
            raise TypeError('dispatch grants cannot be copied')

        def __reduce_ex__(self, _protocol: int):
            raise TypeError('dispatch grants cannot be serialized')

        def __repr__(self) -> str:
            return '<CommittedRagDispatchGrant opaque>'

    return cast(CommittedRagDispatchGrant, _Grant())


class RagCostLedger:
    """Transactional exact-two RAG cost ledger.

    Pricing is deliberately absent: this class persists only frozen Task 9
    reserves and caller-supplied strict actual charges.
    """

    def __init__(
        self,
        session: Session,
        *,
        identity_secret: bytes,
        after_commit: Callable[[], None] | None = None,
        actual_cost_authority: Callable[[RagPaidComponent, object], Decimal] | None = None,
        apply_safety_action: Callable[[StrictProviderOutcome], None] | None = None,
    ) -> None:
        if type(identity_secret) is not bytes or not identity_secret:
            raise ValueError('cost-ledger identity secret is required')
        self._session = session
        self._secret = identity_secret
        self._after_commit = after_commit
        self._actual_cost_authority = actual_cost_authority
        self._apply_safety_action = apply_safety_action
        self._active_grants: dict[tuple[int, str], object] = {}
        self._process_nonce = secrets.token_hex(32)

    def _commit(self) -> None:
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        if self._after_commit is not None:
            self._after_commit()

    def create_admission(
        self,
        *,
        agent_run_id: int,
        surface: str,
        mode: str,
        cutover_stage: str,
        configured_backend: str,
        query_context_version: str,
        current_text_hmac: str,
        retrieval_query_hmac: str,
        security_scope_fingerprint: str,
        admission_cache_identity_hmac: str | None,
        source_window: str,
        components: tuple[
            tuple[AuthorizedProviderPolicySnapshot, PreparedPaidCallBudget],
            tuple[AuthorizedProviderPolicySnapshot, PreparedPaidCallBudget],
        ],
    ) -> RagRunAdmission:
        if (
            type(agent_run_id) is not int
            or agent_run_id <= 0
            or source_window not in ADMISSION_SOURCE_WINDOWS
            or source_window != f'rag-v2:admission:{mode}:{surface}:{configured_backend}'
            or surface not in {'ask', 'search', 'assistant'}
            or mode not in {'shadow', 'enforce'}
            or cutover_stage not in {'none', 'ask', 'search', 'assistant'}
            or configured_backend not in {'keyword', 'pgvector'}
            or query_context_version not in {'direct-query:v1', 'assistant-context:v1'}
            or type(components) is not tuple
            or len(components) != 2
        ):
            raise ValueError('RAG admission is invalid')
        for value in (
            current_text_hmac, retrieval_query_hmac, security_scope_fingerprint,
        ):
            require_lower_hmac(value)
        normalized = []
        for ordinal, expected in enumerate(_COMPONENT_ORDER):
            pair = components[ordinal]
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError('RAG admission components are invalid')
            snapshot, budget = pair
            if (
                type(snapshot) is not AuthorizedProviderPolicySnapshot
                or type(budget) is not PreparedPaidCallBudget
                or snapshot.component != expected
                or budget.component != expected
                or not hmac.compare_digest(
                    snapshot.authorized_policy_snapshot_hmac,
                    budget.cost_policy_snapshot_hmac,
                )
            ):
                raise ValueError('RAG admission components are misaligned')
            normalized.append((snapshot, budget))
        total_reserved = sum(
            (budget.reserved_cost_usd for _, budget in normalized), _ZERO
        )
        computed_admission_hmac = admission_identity(
            {
                'answer_provider_policy_snapshot_hmac': (
                    normalized[1][0].authorized_policy_snapshot_hmac
                    if mode == 'enforce' and surface in {'ask', 'assistant'}
                    else None
                ),
                'configured_backend': configured_backend,
                'current_text_hmac': current_text_hmac,
                'cutover_stage': cutover_stage,
                'graph_version': 'company-memory-rag-answer-v2.0',
                'mode': mode,
                'query_context_version_bytes': exact_utf8_bytes(query_context_version),
                'query_embedding_provider_policy_snapshot_hmac': (
                    normalized[0][0].authorized_policy_snapshot_hmac
                    if configured_backend == 'pgvector' else None
                ),
                'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
                'retrieval_query_hmac': retrieval_query_hmac,
                'security_scope_fingerprint': security_scope_fingerprint,
                'surface': surface,
            }, secret=self._secret,
        )
        if (
            admission_cache_identity_hmac is not None
            and admission_cache_identity_hmac != computed_admission_hmac
        ):
            raise ValueError('caller admission identity does not match server authority')
        runtime_hmac = runtime_cost_identity(
            {
                'agent_run_id': agent_run_id,
                'components': [
                    {
                        'component': budget.component,
                        'estimated_input_tokens': budget.estimated_input_tokens,
                        'maximum_output_tokens': budget.maximum_output_tokens,
                        'reserved_cost_usd': format(budget.reserved_cost_usd, 'f'),
                        'policy_snapshot_hmac': budget.cost_policy_snapshot_hmac,
                        'estimator_input_hmac': budget.estimator_input_hmac,
                    }
                    for _, budget in normalized
                ],
            },
            secret=self._secret,
        )
        parent = AgentRun(
            id=agent_run_id,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v2',
            status='running',
            source_window=source_window,
            cache_key='rag-v2-admission:' + computed_admission_hmac,
            model_name='rag-v2-admission',
            generation_provider=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=float(total_reserved),
            permission_level='restricted',
            metadata_={'runtime_cost_snapshot_hmac': runtime_hmac},
            run_contract_version='rag-run:v2',
            run_record_phase='admission',
            total_charged_cost_usd=_ZERO,
            projection_owner_fence_hmac=None,
            completed_at=None,
        )
        self._session.add(parent)
        for ordinal, (snapshot, budget) in enumerate(normalized):
            self._session.add(AgentRunCostComponent(
                agent_run_id=agent_run_id,
                component=budget.component,
                component_ordinal=ordinal,
                dispatch_state='not_attempted',
                dispatch_fence_hmac=None,
                process_instance_hmac=None,
                attempted=False,
                dispatch_count=0,
                reserved_input_tokens=budget.estimated_input_tokens,
                reserved_output_tokens=budget.maximum_output_tokens,
                actual_input_tokens=None,
                actual_output_tokens=None,
                reserved_cost_usd=budget.reserved_cost_usd,
                charged_cost_usd=_ZERO,
                charge_basis='zero',
                overrun=False,
                provider=snapshot.provider,
                model=snapshot.model,
                authorized_model_config_version=snapshot.authorized_model_config_version,
                authorized_model_config_snapshot_hmac=snapshot.authorized_model_config_snapshot_hmac,
                authorized_cost_policy_version=snapshot.authorized_cost_policy_version,
                authorized_token_estimator_version=snapshot.authorized_token_estimator_version,
                authorized_policy_snapshot_hmac=snapshot.authorized_policy_snapshot_hmac,
                terminal_outcome=None,
            ))
        self._commit()
        return RagRunAdmission(
            agent_run_id=agent_run_id,
            surface=surface,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            cutover_stage=cutover_stage,  # type: ignore[arg-type]
            configured_backend=configured_backend,  # type: ignore[arg-type]
            query_context_version=query_context_version,  # type: ignore[arg-type]
            current_text_hmac=current_text_hmac,
            retrieval_query_hmac=retrieval_query_hmac,
            security_scope_fingerprint=security_scope_fingerprint,
            query_embedding_provider_policy_snapshot_hmac=normalized[0][0].authorized_policy_snapshot_hmac,
            answer_provider_policy_snapshot_hmac=normalized[1][0].authorized_policy_snapshot_hmac,
            admission_cache_identity_hmac=computed_admission_hmac,
            source_window=source_window,  # type: ignore[arg-type]
            status='running',
            run_record_phase='admission',
            component_order=_COMPONENT_ORDER,
            total_reserved_cost_usd=total_reserved,
            total_charged_cost_usd=_ZERO,
            runtime_cost_snapshot_hmac=runtime_hmac,
        )

    def claim_component(
        self,
        *,
        run_id: int,
        component: RagPaidComponent,
        prepared: PreparedPaidCallBudget,
    ) -> CommittedRagDispatchGrant:
        if component not in _COMPONENT_ORDER or type(prepared) is not PreparedPaidCallBudget:
            raise ValueError('component claim is invalid')
        parent = self._session.get(AgentRun, run_id, with_for_update=True)
        rows = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
            .with_for_update()
        ))
        if (
            parent is None
            or parent.status != 'running'
            or parent.run_record_phase != 'admission'
            or tuple(row.component for row in rows) != _COMPONENT_ORDER
        ):
            raise RagCostLedgerError('committed exact-two admission is unavailable')
        row = rows[_COMPONENT_ORDER.index(component)]
        if (
            row.dispatch_state != 'not_attempted'
            or prepared.component != component
            or row.reserved_input_tokens != prepared.estimated_input_tokens
            or row.reserved_output_tokens != prepared.maximum_output_tokens
            or Decimal(row.reserved_cost_usd) != prepared.reserved_cost_usd
            or not hmac.compare_digest(
                row.authorized_policy_snapshot_hmac,
                prepared.cost_policy_snapshot_hmac,
            )
        ):
            raise RagCostLedgerError('component claim authority is stale')
        fence = dispatch_fence_identity(
            {
                'agent_run_id': run_id,
                'component': component,
                'dispatch_count': 1,
                'estimator_input_hmac': prepared.estimator_input_hmac,
            }, secret=self._secret,
        )
        process = process_instance_identity(
            {'process_nonce': self._process_nonce}, secret=self._secret
        )
        safety = provider_safety_snapshot_identity(
            {
                'component': component,
                'provider': row.provider,
                'model': row.model,
                'policy_snapshot_hmac': row.authorized_policy_snapshot_hmac,
            }, secret=self._secret,
        )
        row.dispatch_state = 'dispatching'
        row.dispatch_fence_hmac = fence
        row.process_instance_hmac = process
        row.attempted = True
        row.dispatch_count = 1
        row.charged_cost_usd = row.reserved_cost_usd
        row.charge_basis = 'reserved'
        row.updated_at = row.updated_at
        self._commit()

        key = (run_id, component)
        holder: list[CommittedRagDispatchGrant] = []

        def consume() -> None:
            if not holder or self._active_grants.get(key) is not holder[0]:
                raise RagCostLedgerError('dispatch grant authority is unavailable')

        grant = _make_grant(
            agent_run_id=run_id,
            component=component,
            reserved_cost_usd=Decimal(row.reserved_cost_usd),
            dispatch_fence_hmac=fence,
            provider_safety_snapshot_hmac=safety,
            on_consume=consume,
        )
        holder.append(grant)
        self._active_grants[key] = grant
        return grant

    def consume_committed_grant(self, grant: object) -> None:
        """Authenticate and consume the exact process-local store capability."""
        try:
            key = (grant.agent_run_id, grant.component)
        except Exception:
            raise RagCostLedgerError('dispatch grant is invalid') from None
        if self._active_grants.get(key) is not grant:
            raise RagCostLedgerError('dispatch grant is not store-owned')
        try:
            grant.consume_at_dispatch()
        except RagCostLedgerError:
            raise
        except Exception:
            raise RagCostLedgerError('dispatch grant consumption failed') from None

    def finalize_component(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        outcome: StrictProviderOutcome,
    ) -> RagComponentFinal:
        if type(outcome) is not StrictProviderOutcome or outcome.component != grant.component:
            raise ValueError('provider outcome is invalid')
        key = (grant.agent_run_id, grant.component)
        if self._active_grants.get(key) is not grant:
            raise RagCostLedgerError('dispatch grant is not active')
        if getattr(grant, 'consumed', None) is not True:
            raise RagCostLedgerError('dispatch grant was not consumed by transport')
        self._validate_outcome(outcome)
        parent = self._session.get(AgentRun, grant.agent_run_id, with_for_update=True)
        row = self._session.scalar(
            select(AgentRunCostComponent).where(
                AgentRunCostComponent.agent_run_id == grant.agent_run_id,
                AgentRunCostComponent.component == grant.component,
            ).with_for_update()
        )
        if (
            parent is None or row is None or row.dispatch_state != 'dispatching'
            or not hmac.compare_digest(row.dispatch_fence_hmac or '', grant.dispatch_fence_hmac)
        ):
            raise RagCostLedgerError('dispatch row is not finalizable')
        if not outcome.provider_dispatch_started:
            raise ValueError('claimed provider component must be attempted')
        actual = outcome.actual_cost_usd
        usage = outcome.strict_usage
        if (actual is None) != (usage is None):
            raise ValueError('actual cost and strict usage must appear together')
        row.dispatch_state = (
            'abandoned_unknown'
            if outcome.terminal_outcome == 'abandoned_unknown'
            else 'terminal'
        )
        row.terminal_outcome = outcome.terminal_outcome
        if actual is None:
            row.actual_input_tokens = None
            row.actual_output_tokens = None
            row.charged_cost_usd = row.reserved_cost_usd
            row.charge_basis = 'reserved'
            row.overrun = False
        else:
            if type(actual) is not Decimal or actual < 0:
                raise ValueError('actual provider cost is invalid')
            if self._actual_cost_authority is None:
                raise RagCostLedgerError('actual provider cost authority is unavailable')
            expected_actual = self._actual_cost_authority(outcome.component, usage)
            if type(expected_actual) is not Decimal or expected_actual != actual:
                raise ValueError('actual provider cost does not match authority')
            row.actual_input_tokens = usage.input_tokens
            row.actual_output_tokens = usage.output_tokens
            row.charged_cost_usd = actual
            row.charge_basis = 'actual'
            row.overrun = actual > Decimal(row.reserved_cost_usd)
            if row.overrun != (outcome.classification == 'known_overrun'):
                raise ValueError('provider overrun classification is inconsistent')
        if outcome.safety_action != 'unchanged':
            if self._apply_safety_action is None:
                raise RagCostLedgerError('provider safety authority is unavailable')
            self._apply_safety_action(outcome)
        self._session.flush()
        parent.total_charged_cost_usd = self.total_charged_cost(grant.agent_run_id)
        siblings = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == grant.agent_run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
        ))
        if all(item.dispatch_state in {'terminal', 'abandoned_unknown'} for item in siblings):
            owner_process = next(
                item.process_instance_hmac for item in siblings
                if item.process_instance_hmac is not None
            )
            parent.run_record_phase = 'cost_finalized_pending_projection'
            parent.projection_owner_fence_hmac = rag_identity_hmac(
                {
                    'advisory_lock_identity_digest': None,
                    'agent_run_id': grant.agent_run_id,
                    'lock_backend': 'sqlite_process_mutex',
                    'lock_namespace': 'sqlite_rag_smoke_projection_owner',
                    'owner_nonce_hex': secrets.token_hex(32),
                    'owner_phase': 'cost_finalized_pending_projection',
                    'process_instance_hmac': owner_process,
                }, secret=self._secret,
                schema_version='rag-projection-owner-fence:v1',
                policy_version='rag-run:v2',
            )
        self._commit()
        self._active_grants.pop(key, None)
        return self._component_final(parent, row)

    @staticmethod
    def _validate_outcome(outcome: StrictProviderOutcome) -> None:
        matrix = {
            'validated_success': ('component_succeeded', True, True, 'actual', 'unchanged'),
            'response_less_failure': ('model_provider_failed', True, False, 'reserved', 'unchanged'),
            'response_identity_invalid': (
                'provider_response_identity_invalid', True, True, 'reserved',
                'block_remediation',
            ),
            'embedding_payload_invalid': (
                'provider_embedding_payload_invalid', True, True, 'actual',
                'block_remediation',
            ),
            'usage_contract_invalid': (
                'provider_safety_unavailable', True, True, 'reserved',
                'block_remediation',
            ),
            'usage_storage_invalid': (
                'provider_safety_unavailable', True, True, 'reserved',
                'block_remediation',
            ),
            'known_overrun': (
                'provider_usage_overrun', True, True, 'actual', 'block_overrun'
            ),
            'structured_output_invalid': (
                'structured_output_invalid', True, True, 'actual', 'unchanged'
            ),
            'citation_validation_failed': (
                'citation_validation_failed', True, True, 'actual', 'unchanged'
            ),
            'evidence_validation_failed': (
                'evidence_unavailable', True, True, 'actual', 'unchanged'
            ),
        }
        expected = matrix.get(outcome.classification)
        if expected is None or outcome.classification == 'pre_send_refusal':
            raise ValueError('provider outcome classification is not finalizable')
        terminal, started, received, basis, action = expected
        has_actual = outcome.strict_usage is not None and outcome.actual_cost_usd is not None
        if (
            outcome.terminal_outcome != terminal
            or outcome.provider_dispatch_started is not started
            or outcome.provider_response_received is not received
            or outcome.safety_action != action
            or has_actual != (basis == 'actual')
            or (outcome.strict_usage is None) != (outcome.actual_cost_usd is None)
        ):
            raise ValueError('provider outcome classification is inconsistent')

    def total_charged_cost(self, run_id: int) -> Decimal:
        rows = self._session.scalars(select(AgentRunCostComponent).where(
            AgentRunCostComponent.agent_run_id == run_id
        ))
        return sum((Decimal(row.charged_cost_usd) for row in rows), _ZERO)

    def finalize_projectionless_failure(
        self,
        *,
        run_id: int,
        outcome: str,
        completed_at: datetime | None = None,
    ) -> RagRunTerminal:
        """Close an admission before any paid dispatch using exact terminal-zero rows."""
        if outcome != 'abandoned_unknown':
            raise ValueError('admission-only outcome must be abandoned_unknown')
        parent = self._session.get(AgentRun, run_id, with_for_update=True)
        rows = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
            .with_for_update()
        ))
        if (
            parent is None
            or parent.status != 'running'
            or parent.run_record_phase != 'admission'
            or tuple(row.component for row in rows) != _COMPONENT_ORDER
            or any(row.dispatch_state != 'not_attempted' for row in rows)
        ):
            raise RagCostLedgerError('projectionless terminal-zero closure is unavailable')
        timestamp = completed_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError('completion timestamp must be timezone-aware')
        for row in rows:
            row.dispatch_state = 'terminal'
            row.attempted = False
            row.dispatch_count = 0
            row.reserved_input_tokens = 0
            row.reserved_output_tokens = 0
            row.actual_input_tokens = None
            row.actual_output_tokens = None
            row.reserved_cost_usd = _ZERO
            row.charged_cost_usd = _ZERO
            row.charge_basis = 'zero'
            row.overrun = False
            row.dispatch_fence_hmac = None
            row.process_instance_hmac = None
            row.terminal_outcome = None
        parent.status = 'failed'
        parent.run_record_phase = 'admission_only'
        parent.total_charged_cost_usd = _ZERO
        parent.completed_at = timestamp
        terminal_runtime_hmac = runtime_cost_identity(
            {
                'agent_run_id': run_id,
                'component_order': list(_COMPONENT_ORDER),
                'outcome': outcome,
                'parent_run_record_phase': 'admission_only',
                'parent_status': 'failed',
                'snapshot_stage': 'terminal',
                'total_charged_cost_usd': '0.000000',
                'total_reserved_cost_usd': '0.000000',
            }, secret=self._secret,
        )
        parent.metadata_ = {
            **parent.metadata_,
            'runtime_cost_snapshot_hmac': terminal_runtime_hmac,
            'outcome': outcome,
        }
        self._commit()
        finals = tuple(self._component_final(parent, row) for row in rows)
        return RagRunTerminal(
            agent_run_id=run_id,
            status='failed',
            run_record_phase='admission_only',
            outcome=outcome,  # type: ignore[arg-type]
            admission_cache_identity_hmac=parent.cache_key.removeprefix(
                'rag-v2-admission:'
            ),
            source_window=parent.source_window,  # type: ignore[arg-type]
            cache_key=parent.cache_key,
            total_reserved_cost_usd=_ZERO,
            total_charged_cost_usd=_ZERO,
            component_finals=cast(tuple[RagComponentFinal, RagComponentFinal], finals),
            projection_owner_fence_hmac=None,
            runtime_cost_snapshot_hmac=terminal_runtime_hmac,
            terminal_identity_hmac=None,
            completed_at=timestamp,
        )

    def _component_final(
        self, parent: AgentRun, row: AgentRunCostComponent
    ) -> RagComponentFinal:
        return RagComponentFinal(
            agent_run_id=parent.id,
            component=row.component,  # type: ignore[arg-type]
            terminal_outcome=row.terminal_outcome,  # type: ignore[arg-type]
            dispatch_state=row.dispatch_state,  # type: ignore[arg-type]
            attempted=row.attempted,
            dispatch_count=row.dispatch_count,  # type: ignore[arg-type]
            reserved_input_tokens=row.reserved_input_tokens,
            reserved_output_tokens=row.reserved_output_tokens,
            actual_input_tokens=row.actual_input_tokens,
            actual_output_tokens=row.actual_output_tokens,
            reserved_cost_usd=Decimal(row.reserved_cost_usd),
            charged_cost_usd=Decimal(row.charged_cost_usd),
            charge_basis=row.charge_basis,  # type: ignore[arg-type]
            overrun=row.overrun,
            provider=row.provider,
            model=row.model,
            authorized_model_config_version=row.authorized_model_config_version,
            authorized_model_config_snapshot_hmac=row.authorized_model_config_snapshot_hmac,
            authorized_cost_policy_version=row.authorized_cost_policy_version,
            authorized_token_estimator_version=row.authorized_token_estimator_version,
            authorized_policy_snapshot_hmac=row.authorized_policy_snapshot_hmac,
            dispatch_fence_hmac=row.dispatch_fence_hmac,
            process_instance_hmac=row.process_instance_hmac,
            parent_status=parent.status,  # type: ignore[arg-type]
            parent_run_record_phase=parent.run_record_phase,  # type: ignore[arg-type]
            parent_total_charged_cost_usd=Decimal(parent.total_charged_cost_usd),
            projection_owner_fence_hmac=parent.projection_owner_fence_hmac,
            runtime_cost_snapshot_hmac=parent.metadata_['runtime_cost_snapshot_hmac'],
        )
