from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.auto_review_cost_policy import (
    EXTRACTION_ROUTE_POLICIES,
    VALIDATION_COST_POLICY,
)
from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.agent_runtime.keyed_mutation_guard import KeyedMutationGuard
from backend.app.core.config import Settings, get_settings
from backend.app.db.session import SessionLocal
from backend.app.models import (
    AuditLog,
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewProviderSafetyEvent,
    AutoReviewProviderSafetyState,
    AutoReviewRolloutControlEvent,
    AutoReviewRolloutState,
)
from backend.app.review.auto_review_quality_revoke import (
    AutoReviewQualityRevokeService,
)
from backend.app.review.auto_review_rollout import (
    AutoReviewRolloutPolicyService,
    RolloutGateError,
    RolloutPercentage,
)

_SYSTEM_OPERATOR_ID = 'system:local-auto-review-rollout-admin'
_SYSTEM_OPERATOR_EMAIL = 'local-auto-review-rollout-admin@paraworks.invalid'


class RolloutAdminRefused(RuntimeError):  # noqa: N818 - bounded domain refusal
    """Bounded local rollout-admin refusal."""


@dataclass(frozen=True, slots=True)
class RolloutAdminStatus:
    scope: str
    policy: str
    state_version: int
    control_epoch: int
    max_authorized_percentage: RolloutPercentage
    authorization_generation: int
    breaker_open: bool


class AutoReviewRolloutAdminService:
    def __init__(self, db: Session, *, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def status(self, scope: str, policy: str) -> RolloutAdminStatus:
        snapshot = AutoReviewRolloutPolicyService(
            self._db, settings=self._settings
        ).peek_or_default(scope, policy)
        return self._status(snapshot)

    def authorize_percentage(
        self,
        *,
        scope: str,
        policy: str,
        percentage: RolloutPercentage,
        expected_state_version: int,
        reason: str,
        gate_ref: str,
    ) -> RolloutAdminStatus:
        normalized_reason = ' '.join(reason.split())
        normalized_gate = gate_ref.strip()
        if not 1 <= len(normalized_reason) <= 500:
            raise RolloutAdminRefused('rollout reason must contain 1 to 500 characters')
        if not 1 <= len(normalized_gate) <= 120:
            raise RolloutAdminRefused('rollout gate reference is invalid')
        with KeyedMutationGuard.generation_barrier(self._db):
            runtime = KeyedMutationGuard.lock_runtime_key_state(
                self._db, for_update=False
            )
            verifier = fingerprint_key_material_verifier(
                self._settings.agent_runtime_fingerprint_secret
            )
            if (
                runtime is None
                or not runtime.ready
                or runtime.fingerprint_key_version
                != self._settings.agent_runtime_fingerprint_key_version
                or runtime.fingerprint_key_material_verifier != verifier
            ):
                raise RolloutAdminRefused('runtime key state is unavailable')
            policy_service = AutoReviewRolloutPolicyService(
                self._db, settings=self._settings
            )
            row = policy_service.ensure_row(scope, policy)
            if row.state_version != expected_state_version:
                raise RolloutAdminRefused('rollout state version changed')
            snapshot = policy_service._snapshot(row)
            try:
                authorized = snapshot.require_authorization_transition(
                    percentage
                )
            except RolloutGateError as exc:
                raise RolloutAdminRefused(str(exc)) from None

            now = datetime.now(UTC)
            prior_state_version = row.state_version
            prior_control_epoch = row.control_epoch
            prior_percentage = row.max_authorized_percentage
            prior_breaker = row.breaker_open
            prior_generation = row.authorization_generation
            row.state_version += 1
            row.control_epoch += 1
            row.max_authorized_percentage = authorized
            row.authorization_generation += 1
            row.authorization_at = now
            row.regression_gate_reference = normalized_gate
            row.updated_at = now
            actor_hmac = build_keyed_fingerprint(
                {'subject_id': _SYSTEM_OPERATOR_ID},
                settings=self._settings,
                schema_version='auto-review-rollout-actor:v1',
                policy_version='auto-review-rollout-actor:v1',
            )
            event = AutoReviewRolloutControlEvent(
                rollout_state_id=row.id,
                security_scope_id=row.security_scope_id,
                policy_version=row.policy_version,
                event_sequence=row.last_event_sequence + 1,
                event_kind='percentage_authorized',
                prior_state_version=prior_state_version,
                new_state_version=row.state_version,
                prior_control_epoch=prior_control_epoch,
                new_control_epoch=row.control_epoch,
                prior_max_authorized_percentage=prior_percentage,
                new_max_authorized_percentage=authorized,
                prior_breaker_open=prior_breaker,
                new_breaker_open=False,
                prior_authorization_generation=prior_generation,
                new_authorization_generation=row.authorization_generation,
                reason_code='operator_authorized',
                regression_gate_reference=normalized_gate,
                actor_subject_hmac=actor_hmac,
                fingerprint_key_version=runtime.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    runtime.fingerprint_key_material_verifier
                ),
                created_at=now,
            )
            self._db.add(event)
            self._db.flush()
            row.last_event_sequence = event.event_sequence
            row.last_event_id = event.id
            self._db.add(
                AuditLog(
                    actor_id=_SYSTEM_OPERATOR_ID,
                    actor_email=_SYSTEM_OPERATOR_EMAIL,
                    actor_role='system',
                    action='auto_review_rollout_authorized',
                    target_type='auto_review_rollout',
                    target_id=f'{scope}:{policy}',
                    status='success',
                    metadata_={
                        'percentage': authorized,
                        'reason': normalized_reason,
                        'gate_ref': normalized_gate,
                    },
                )
            )
            self._db.flush()
            return self._status(policy_service._snapshot(row))

    def close_rollout_breaker(
        self,
        *,
        scope: str,
        policy: str,
        expected_state_version: int,
        reason: str,
        gate_ref: str,
    ) -> RolloutAdminStatus:
        normalized_reason, normalized_gate = self._normalize_control_input(
            reason, gate_ref
        )
        with KeyedMutationGuard.generation_barrier(self._db):
            runtime = KeyedMutationGuard.lock_runtime_key_state(
                self._db, for_update=False
            )
            verifier = fingerprint_key_material_verifier(
                self._settings.agent_runtime_fingerprint_secret
            )
            if (
                runtime is None
                or not runtime.ready
                or runtime.fingerprint_key_version
                != self._settings.agent_runtime_fingerprint_key_version
                or runtime.fingerprint_key_material_verifier != verifier
            ):
                raise RolloutAdminRefused('runtime key state is unavailable')
            statement = select(AutoReviewRolloutState).where(
                AutoReviewRolloutState.security_scope_id == scope,
                AutoReviewRolloutState.policy_version == policy,
            )
            if self._db.get_bind().dialect.name == 'postgresql':
                statement = statement.with_for_update()
            row = self._db.scalar(statement)
            if row is None or row.state_version != expected_state_version:
                raise RolloutAdminRefused('rollout state version changed')
            if not row.breaker_open:
                raise RolloutAdminRefused('rollout breaker is not open')
            if row.corrected_critical_count != 0:
                raise RolloutAdminRefused(
                    'audit correction requires a new reviewed policy'
                )
            unresolved = self._db.scalar(
                select(AutoReviewPostAudit.id)
                .join(
                    AutoReviewPromotionDecision,
                    AutoReviewPromotionDecision.id
                    == AutoReviewPostAudit.promotion_decision_id,
                )
                .where(
                    AutoReviewPromotionDecision.security_scope_id == scope,
                    AutoReviewPromotionDecision.policy_version == policy,
                    AutoReviewPostAudit.status.in_(
                        {'pending', 'remediation_required'}
                    ),
                )
                .limit(1)
            )
            unresolved_correction = self._db.scalar(
                select(AutoReviewAuditCorrection.id)
                .join(
                    AutoReviewPromotionDecision,
                    AutoReviewPromotionDecision.review_item_id
                    == AutoReviewAuditCorrection.review_item_id,
                )
                .where(
                    AutoReviewPromotionDecision.security_scope_id == scope,
                    AutoReviewPromotionDecision.policy_version == policy,
                    AutoReviewAuditCorrection.status == 'remediation_required',
                )
                .limit(1)
            )
            if unresolved is not None or unresolved_correction is not None:
                raise RolloutAdminRefused(
                    'rollout remediation must be resolved before close'
                )

            now = datetime.now(UTC)
            prior_state_version = row.state_version
            prior_control_epoch = row.control_epoch
            prior_percentage = row.max_authorized_percentage
            prior_generation = row.authorization_generation
            row.state_version += 1
            row.control_epoch += 1
            row.breaker_open = False
            row.breaker_reason_code = None
            row.max_authorized_percentage = 0
            row.authorization_generation += 1
            row.regression_gate_reference = normalized_gate
            row.updated_at = now
            actor_hmac = build_keyed_fingerprint(
                {'subject_id': _SYSTEM_OPERATOR_ID},
                settings=self._settings,
                schema_version='auto-review-rollout-actor:v1',
                policy_version='auto-review-rollout-actor:v1',
            )
            event = AutoReviewRolloutControlEvent(
                rollout_state_id=row.id,
                security_scope_id=scope,
                policy_version=policy,
                event_sequence=row.last_event_sequence + 1,
                event_kind='breaker_closed',
                prior_state_version=prior_state_version,
                new_state_version=row.state_version,
                prior_control_epoch=prior_control_epoch,
                new_control_epoch=row.control_epoch,
                prior_max_authorized_percentage=prior_percentage,
                new_max_authorized_percentage=0,
                prior_breaker_open=True,
                new_breaker_open=False,
                prior_authorization_generation=prior_generation,
                new_authorization_generation=row.authorization_generation,
                reason_code='operator_closed',
                regression_gate_reference=normalized_gate,
                actor_subject_hmac=actor_hmac,
                fingerprint_key_version=runtime.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    runtime.fingerprint_key_material_verifier
                ),
                created_at=now,
            )
            self._db.add(event)
            self._db.flush([event])
            row.last_event_sequence = event.event_sequence
            row.last_event_id = event.id
            self._db.add(
                AuditLog(
                    actor_id=_SYSTEM_OPERATOR_ID,
                    actor_email=_SYSTEM_OPERATOR_EMAIL,
                    actor_role='system',
                    action='auto_review_rollout_breaker_closed',
                    target_type='auto_review_rollout',
                    target_id=f'{scope}:{policy}',
                    status='success',
                    metadata_={
                        'reason': normalized_reason,
                        'gate_ref': normalized_gate,
                    },
                )
            )
            self._db.flush()
            return self._status(
                AutoReviewRolloutPolicyService(
                    self._db, settings=self._settings
                )._snapshot(row)
            )

    def authorize_initial_provider_policy(
        self,
        *,
        purpose: str,
        provider: str,
        model: str,
        reasoning_effort: str,
        expected_absent: bool,
        new_cost_policy_version: str,
        reason: str,
        gate_ref: str,
    ) -> AutoReviewProviderSafetyState:
        normalized_reason, normalized_gate = self._normalize_control_input(
            reason, gate_ref
        )
        policy = self._provider_registry_policy(
            purpose=purpose,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            cost_policy_version=new_cost_policy_version,
        )
        if policy is None:
            raise RolloutAdminRefused('provider policy is not in the registry')
        if not expected_absent:
            raise RolloutAdminRefused('initial authorization requires expected absent')
        with KeyedMutationGuard.generation_barrier(self._db):
            runtime = KeyedMutationGuard.lock_runtime_key_state(
                self._db, for_update=False
            )
            if runtime is None or not runtime.ready:
                raise RolloutAdminRefused('runtime key state is unavailable')
            statement = select(AutoReviewProviderSafetyState).where(
                AutoReviewProviderSafetyState.purpose == purpose,
                AutoReviewProviderSafetyState.provider == provider,
                AutoReviewProviderSafetyState.model == model,
                AutoReviewProviderSafetyState.reasoning_effort
                == reasoning_effort,
            )
            if self._db.get_bind().dialect.name == 'postgresql':
                statement = statement.with_for_update()
            if self._db.scalar(statement) is not None:
                raise RolloutAdminRefused('provider safety row already exists')
            now = datetime.now(UTC)
            state = AutoReviewProviderSafetyState(
                purpose=purpose,
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                state_version=1,
                authorized_cost_policy_version=new_cost_policy_version,
                token_estimator_version=policy['estimator'],
                tokenizer_encoding=policy['encoding'],
                reply_priming_tokens=policy['reply_priming'],
                framing_safety_tokens=policy['framing_safety'],
                input_usd_per_1m=policy['input_price'],
                output_usd_per_1m=policy['output_price'],
                breaker_open=False,
                overrun_count=0,
                regression_gate_reference=normalized_gate,
                authorized_at=now,
                created_at=now,
                updated_at=now,
            )
            self._db.add(state)
            self._db.flush([state])
            actor_hmac = self._operator_hmac()
            event = AutoReviewProviderSafetyEvent(
                provider_safety_state_id=state.id,
                purpose=purpose,
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                event_sequence=1,
                event_kind='initial_authorized',
                prior_state_version=0,
                new_state_version=1,
                cost_policy_version=new_cost_policy_version,
                token_estimator_version=policy['estimator'],
                tokenizer_encoding=policy['encoding'],
                reply_priming_tokens=policy['reply_priming'],
                framing_safety_tokens=policy['framing_safety'],
                input_usd_per_1m=policy['input_price'],
                output_usd_per_1m=policy['output_price'],
                prior_breaker_open=False,
                new_breaker_open=False,
                reason_code='operator_initial_authorized',
                regression_gate_reference=normalized_gate,
                actor_subject_hmac=actor_hmac,
                call_hmac=None,
                fingerprint_key_version=runtime.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    runtime.fingerprint_key_material_verifier
                ),
                created_at=now,
            )
            self._db.add(event)
            self._db.flush([event])
            state.last_event_sequence = 1
            state.last_event_id = event.id
            self._db.add(
                AuditLog(
                    actor_id=_SYSTEM_OPERATOR_ID,
                    actor_email=_SYSTEM_OPERATOR_EMAIL,
                    actor_role='system',
                    action='auto_review_provider_policy_authorized',
                    target_type='auto_review_provider_safety',
                    target_id=(
                        f'{purpose}:{provider}:{model}:{reasoning_effort}'
                    ),
                    status='success',
                    metadata_={
                        'reason': normalized_reason,
                        'gate_ref': normalized_gate,
                        'cost_policy_version': new_cost_policy_version,
                    },
                )
            )
            self._db.flush()
            return state

    def clear_provider_breaker(
        self,
        *,
        purpose: str,
        provider: str,
        model: str,
        reasoning_effort: str,
        expected_state_version: int,
        new_cost_policy_version: str,
        reason: str,
        gate_ref: str,
    ) -> AutoReviewProviderSafetyState:
        normalized_reason, normalized_gate = self._normalize_control_input(
            reason, gate_ref
        )
        with KeyedMutationGuard.generation_barrier(self._db):
            runtime = KeyedMutationGuard.lock_runtime_key_state(
                self._db, for_update=False
            )
            if runtime is None or not runtime.ready:
                raise RolloutAdminRefused('runtime key state is unavailable')
            statement = select(AutoReviewProviderSafetyState).where(
                AutoReviewProviderSafetyState.purpose == purpose,
                AutoReviewProviderSafetyState.provider == provider,
                AutoReviewProviderSafetyState.model == model,
                AutoReviewProviderSafetyState.reasoning_effort
                == reasoning_effort,
            )
            if self._db.get_bind().dialect.name == 'postgresql':
                statement = statement.with_for_update()
            state = self._db.scalar(statement)
            if state is None or state.state_version != expected_state_version:
                raise RolloutAdminRefused('provider safety state version changed')
            if not state.breaker_open:
                raise RolloutAdminRefused('provider breaker is not open')
            if new_cost_policy_version == state.authorized_cost_policy_version:
                raise RolloutAdminRefused('provider clear requires a new cost policy')
            policy = self._provider_registry_policy(
                purpose=purpose,
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                cost_policy_version=new_cost_policy_version,
            )
            if policy is None:
                raise RolloutAdminRefused('provider policy is not in the registry')
            now = datetime.now(UTC)
            prior_version = state.state_version
            state.state_version += 1
            state.authorized_cost_policy_version = new_cost_policy_version
            state.token_estimator_version = policy['estimator']
            state.tokenizer_encoding = policy['encoding']
            state.reply_priming_tokens = policy['reply_priming']
            state.framing_safety_tokens = policy['framing_safety']
            state.input_usd_per_1m = policy['input_price']
            state.output_usd_per_1m = policy['output_price']
            state.breaker_open = False
            state.breaker_reason_code = None
            state.regression_gate_reference = normalized_gate
            state.cleared_at = now
            state.updated_at = now
            event = AutoReviewProviderSafetyEvent(
                provider_safety_state_id=state.id,
                purpose=purpose,
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                event_sequence=state.last_event_sequence + 1,
                event_kind='breaker_cleared',
                prior_state_version=prior_version,
                new_state_version=state.state_version,
                cost_policy_version=new_cost_policy_version,
                token_estimator_version=policy['estimator'],
                tokenizer_encoding=policy['encoding'],
                reply_priming_tokens=policy['reply_priming'],
                framing_safety_tokens=policy['framing_safety'],
                input_usd_per_1m=policy['input_price'],
                output_usd_per_1m=policy['output_price'],
                prior_breaker_open=True,
                new_breaker_open=False,
                reason_code='operator_breaker_cleared',
                regression_gate_reference=normalized_gate,
                actor_subject_hmac=self._operator_hmac(),
                call_hmac=None,
                fingerprint_key_version=runtime.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    runtime.fingerprint_key_material_verifier
                ),
                created_at=now,
            )
            self._db.add(event)
            self._db.flush([event])
            state.last_event_sequence = event.event_sequence
            state.last_event_id = event.id
            self._db.add(
                AuditLog(
                    actor_id=_SYSTEM_OPERATOR_ID,
                    actor_email=_SYSTEM_OPERATOR_EMAIL,
                    actor_role='system',
                    action='auto_review_provider_breaker_cleared',
                    target_type='auto_review_provider_safety',
                    target_id=(
                        f'{purpose}:{provider}:{model}:{reasoning_effort}'
                    ),
                    status='success',
                    metadata_={
                        'reason': normalized_reason,
                        'gate_ref': normalized_gate,
                        'cost_policy_version': new_cost_policy_version,
                    },
                )
            )
            self._db.flush()
            return state

    def recover_pending_remediation(self, *, limit: int = 100) -> int:
        return AutoReviewQualityRevokeService(
            self._db, settings=self._settings
        ).recover_pending_remediation(limit=limit)

    @staticmethod
    def _normalize_control_input(reason: str, gate_ref: str) -> tuple[str, str]:
        normalized_reason = ' '.join(reason.split())
        normalized_gate = gate_ref.strip()
        if not 1 <= len(normalized_reason) <= 500:
            raise RolloutAdminRefused(
                'rollout reason must contain 1 to 500 characters'
            )
        if not 1 <= len(normalized_gate) <= 120:
            raise RolloutAdminRefused('rollout gate reference is invalid')
        return normalized_reason, normalized_gate

    def _operator_hmac(self) -> str:
        return build_keyed_fingerprint(
            {'subject_id': _SYSTEM_OPERATOR_ID},
            settings=self._settings,
            schema_version='auto-review-rollout-actor:v1',
            policy_version='auto-review-rollout-actor:v1',
        )

    @staticmethod
    def _provider_registry_policy(
        *,
        purpose: str,
        provider: str,
        model: str,
        reasoning_effort: str,
        cost_policy_version: str,
    ) -> dict[str, object] | None:
        if purpose == 'validation':
            policy = VALIDATION_COST_POLICY
            if (
                policy.provider,
                policy.model,
                policy.reasoning_effort,
                policy.cost_policy_version,
            ) != (provider, model, reasoning_effort, cost_policy_version):
                return None
        elif purpose == 'extraction':
            policy = EXTRACTION_ROUTE_POLICIES[0]
            if any(
                (
                    row.provider,
                    row.model,
                    row.reasoning_effort,
                    row.extraction_cost_policy_version,
                    row.token_estimator_version,
                    row.tokenizer_encoding,
                    row.reply_priming_tokens,
                    row.framing_safety_tokens,
                    row.input_cost_per_1m_tokens,
                    row.output_cost_per_1m_tokens,
                )
                != (
                    provider,
                    model,
                    reasoning_effort,
                    cost_policy_version,
                    policy.token_estimator_version,
                    policy.tokenizer_encoding,
                    policy.reply_priming_tokens,
                    policy.framing_safety_tokens,
                    policy.input_cost_per_1m_tokens,
                    policy.output_cost_per_1m_tokens,
                )
                for row in EXTRACTION_ROUTE_POLICIES
            ):
                return None
        else:
            return None
        return {
            'estimator': policy.token_estimator_version,
            'encoding': policy.tokenizer_encoding,
            'reply_priming': policy.reply_priming_tokens,
            'framing_safety': policy.framing_safety_tokens,
            'input_price': policy.input_cost_per_1m_tokens,
            'output_price': policy.output_cost_per_1m_tokens,
        }

    @staticmethod
    def _status(snapshot) -> RolloutAdminStatus:
        return RolloutAdminStatus(
            scope=snapshot.security_scope_id,
            policy=snapshot.policy_version,
            state_version=snapshot.state_version,
            control_epoch=snapshot.control_epoch,
            max_authorized_percentage=snapshot.max_authorized_percentage,
            authorization_generation=snapshot.authorization_generation,
            breaker_open=snapshot.breaker_open,
        )


def _add_rollout_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--scope', required=True)
    parser.add_argument('--policy', required=True)


def _add_control_reason(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--reason', required=True)
    parser.add_argument('--gate-ref', required=True)


def _add_provider_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--purpose', choices=('extraction', 'validation'), required=True)
    parser.add_argument('--provider', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--reasoning-effort', required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    status = commands.add_parser('status')
    _add_rollout_identity(status)
    authorize = commands.add_parser('authorize')
    _add_rollout_identity(authorize)
    authorize.add_argument('--percentage', type=int, choices=(10, 100), required=True)
    authorize.add_argument('--expected-state-version', type=int, required=True)
    _add_control_reason(authorize)
    close = commands.add_parser('close-breaker')
    _add_rollout_identity(close)
    close.add_argument('--expected-state-version', type=int, required=True)
    _add_control_reason(close)
    initial = commands.add_parser('authorize-initial-provider-policy')
    _add_provider_identity(initial)
    initial.add_argument('--new-cost-policy', required=True)
    _add_control_reason(initial)
    clear = commands.add_parser('clear-provider-breaker')
    _add_provider_identity(clear)
    clear.add_argument('--expected-state-version', type=int, required=True)
    clear.add_argument('--new-cost-policy', required=True)
    _add_control_reason(clear)
    recover = commands.add_parser('recover-remediation')
    recover.add_argument('--limit', type=int, default=100)
    args = parser.parse_args(argv)
    try:
        with SessionLocal() as db:
            service = AutoReviewRolloutAdminService(
                db, settings=get_settings()
            )
            if args.command == 'status':
                result = service.status(args.scope, args.policy)
                print(
                    f'scope={result.scope} policy={result.policy} '
                    f'state_version={result.state_version} '
                    f'control_epoch={result.control_epoch} '
                    f'authorized_percentage={result.max_authorized_percentage} '
                    f'generation={result.authorization_generation} '
                    f'breaker_open={str(result.breaker_open).lower()}'
                )
                return 0
            if args.command == 'authorize':
                result = service.authorize_percentage(
                    scope=args.scope,
                    policy=args.policy,
                    percentage=args.percentage,
                    expected_state_version=args.expected_state_version,
                    reason=args.reason,
                    gate_ref=args.gate_ref,
                )
                db.commit()
                print(
                    f'authorized_percentage={result.max_authorized_percentage} '
                    f'state_version={result.state_version} '
                    f'generation={result.authorization_generation}'
                )
                return 0
            if args.command == 'close-breaker':
                result = service.close_rollout_breaker(
                    scope=args.scope,
                    policy=args.policy,
                    expected_state_version=args.expected_state_version,
                    reason=args.reason,
                    gate_ref=args.gate_ref,
                )
                db.commit()
                print(
                    f'breaker_open={str(result.breaker_open).lower()} '
                    f'authorized_percentage={result.max_authorized_percentage} '
                    f'generation={result.authorization_generation}'
                )
                return 0
            if args.command == 'authorize-initial-provider-policy':
                state = service.authorize_initial_provider_policy(
                    purpose=args.purpose,
                    provider=args.provider,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    expected_absent=True,
                    new_cost_policy_version=args.new_cost_policy,
                    reason=args.reason,
                    gate_ref=args.gate_ref,
                )
                db.commit()
                print(
                    f'purpose={state.purpose} state_version={state.state_version} '
                    f'breaker_open={str(state.breaker_open).lower()}'
                )
                return 0
            if args.command == 'clear-provider-breaker':
                state = service.clear_provider_breaker(
                    purpose=args.purpose,
                    provider=args.provider,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    expected_state_version=args.expected_state_version,
                    new_cost_policy_version=args.new_cost_policy,
                    reason=args.reason,
                    gate_ref=args.gate_ref,
                )
                db.commit()
                print(
                    f'purpose={state.purpose} state_version={state.state_version} '
                    f'breaker_open={str(state.breaker_open).lower()}'
                )
                return 0
            recovered = service.recover_pending_remediation(limit=args.limit)
            print(f'recovered_count={recovered}')
            return 0 if recovered >= 0 else 3
    except (RolloutAdminRefused, RolloutGateError, ValueError):
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
