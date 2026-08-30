from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.orm import Session

from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.core.config import Settings
from backend.app.models import AutoReviewRolloutState

RolloutPercentage = Literal[0, 10, 100]
EffectiveRolloutMode = Literal['disabled', 'shadow', 'enforce']
SelectionDomain = Literal[
    'auto-review-enforce-selection:v1',
    'auto-review-audit-selection:v1',
]


class RolloutGateError(RuntimeError):
    """Bounded rollout gate refusal."""


@dataclass(frozen=True, slots=True)
class RolloutControlSnapshot:
    security_scope_id: str
    policy_version: str
    state_version: int
    control_epoch: int
    shadow_predicted_count: int
    shadow_completed_count: int
    shadow_supported_count: int
    enforce_promotion_ordinal: int
    post_audit_selected_count: int
    post_audit_completed_count: int
    post_audit_critical_count: int
    confirmed_mandatory_audit_count: int
    pending_mandatory_audit_count: int
    invalidated_before_audit_count: int
    corrected_critical_count: int
    breaker_open: bool
    max_authorized_percentage: RolloutPercentage
    authorization_generation: int

    @property
    def shadow_precision(self) -> Decimal:
        if self.shadow_completed_count == 0:
            return Decimal('0.0000')
        return (
            Decimal(self.shadow_supported_count)
            / Decimal(self.shadow_completed_count)
        ).quantize(Decimal('0.0001'), rounding=ROUND_DOWN)

    @property
    def shadow_gate_passed(self) -> bool:
        return bool(
            self.shadow_completed_count >= 500
            and self.shadow_precision >= Decimal('0.9900')
            and self.post_audit_critical_count == 0
            and self.corrected_critical_count == 0
            and not self.breaker_open
        )

    @property
    def canary_gate_passed(self) -> bool:
        if self.post_audit_completed_count == 0:
            audit_precision = Decimal('0')
        else:
            audit_precision = (
                Decimal(
                    self.post_audit_completed_count
                    - self.post_audit_critical_count
                )
                / Decimal(self.post_audit_completed_count)
            )
        return bool(
            self.max_authorized_percentage == 10
            and self.enforce_promotion_ordinal >= 500
            and self.confirmed_mandatory_audit_count == 50
            and self.pending_mandatory_audit_count == 0
            and self.post_audit_selected_count
            == self.post_audit_completed_count
            and self.post_audit_critical_count == 0
            and self.corrected_critical_count == 0
            and audit_precision >= Decimal('0.99')
            and not self.breaker_open
        )

    def require_authorization_transition(
        self,
        percentage: RolloutPercentage,
    ) -> RolloutPercentage:
        if percentage not in {10, 100}:
            raise RolloutGateError('rollout percentage is unavailable')
        if percentage <= self.max_authorized_percentage:
            raise RolloutGateError('rollout percentage cannot be replayed')
        if self.max_authorized_percentage == 0:
            if percentage == 100:
                raise RolloutGateError('rollout cannot jump from zero to full')
            if not self.shadow_gate_passed:
                raise RolloutGateError('shadow rollout gate is incomplete')
            return 10
        if self.max_authorized_percentage == 10:
            if percentage != 100 or not self.canary_gate_passed:
                raise RolloutGateError('canary rollout gate is incomplete')
            return 100
        raise RolloutGateError('rollout percentage cannot increase')


@dataclass(frozen=True, slots=True)
class EffectiveRolloutDecision:
    effective_mode: EffectiveRolloutMode
    effective_percentage: RolloutPercentage
    provider_allowed: bool
    reason_code: Literal[
        'disabled',
        'shadow_only',
        'breaker_open',
        'generation_changed',
        'gate_incomplete',
        'enforce',
    ]


@dataclass(frozen=True, slots=True)
class RolloutSelectionInput:
    security_scope_hmac: str
    workflow_execution_hmac: str
    candidate_key: str
    policy_version: str
    rollout_control_epoch: int
    rollout_authorization_generation: int
    requested_percentage: RolloutPercentage
    authorized_percentage: RolloutPercentage
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str


@dataclass(frozen=True, slots=True)
class RolloutSelection:
    fingerprint: str
    bucket: int
    selected: bool


def stable_rollout_selection(
    value: RolloutSelectionInput,
    *,
    percentage: RolloutPercentage,
    domain: SelectionDomain,
    settings: Settings,
) -> RolloutSelection:
    if percentage not in {0, 10, 100}:
        raise RolloutGateError('rollout sampling percentage is invalid')
    fingerprint = build_keyed_fingerprint(
        asdict(value),
        settings=settings,
        schema_version=domain,
        policy_version=value.policy_version,
    )
    bucket = int(fingerprint, 16) % 100
    return RolloutSelection(
        fingerprint=fingerprint,
        bucket=bucket,
        selected=bucket < percentage,
    )


def resolve_effective_rollout(
    *,
    snapshot: RolloutControlSnapshot,
    stored_mode: Literal['shadow', 'enforce'],
    stored_percentage: RolloutPercentage,
    stored_authorization_generation: int,
    settings: Settings,
) -> EffectiveRolloutDecision:
    if settings.auto_review_mode == 'disabled':
        return EffectiveRolloutDecision('disabled', 0, False, 'disabled')
    if snapshot.breaker_open:
        return EffectiveRolloutDecision('shadow', 0, True, 'breaker_open')
    if settings.auto_review_mode == 'shadow' or stored_mode == 'shadow':
        return EffectiveRolloutDecision('shadow', 0, True, 'shadow_only')
    if stored_authorization_generation != snapshot.authorization_generation:
        return EffectiveRolloutDecision('shadow', 0, True, 'generation_changed')
    if not snapshot.shadow_gate_passed:
        return EffectiveRolloutDecision('shadow', 0, True, 'gate_incomplete')
    effective = min(
        stored_percentage,
        settings.auto_review_enforce_percentage,
        snapshot.max_authorized_percentage,
    )
    if effective not in {10, 100}:
        return EffectiveRolloutDecision('shadow', 0, True, 'shadow_only')
    return EffectiveRolloutDecision('enforce', effective, True, 'enforce')


class AutoReviewRolloutPolicyService:
    def __init__(self, db: Session, *, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    @staticmethod
    def default_snapshot(
        security_scope_id: str,
        policy_version: str,
    ) -> RolloutControlSnapshot:
        return RolloutControlSnapshot(
            security_scope_id=security_scope_id,
            policy_version=policy_version,
            state_version=0,
            control_epoch=0,
            shadow_predicted_count=0,
            shadow_completed_count=0,
            shadow_supported_count=0,
            enforce_promotion_ordinal=0,
            post_audit_selected_count=0,
            post_audit_completed_count=0,
            post_audit_critical_count=0,
            confirmed_mandatory_audit_count=0,
            pending_mandatory_audit_count=0,
            invalidated_before_audit_count=0,
            corrected_critical_count=0,
            breaker_open=False,
            max_authorized_percentage=0,
            authorization_generation=0,
        )

    def peek_or_default(
        self,
        security_scope_id: str,
        policy_version: str,
    ) -> RolloutControlSnapshot:
        row = self._db.scalar(
            select(AutoReviewRolloutState).where(
                AutoReviewRolloutState.security_scope_id == security_scope_id,
                AutoReviewRolloutState.policy_version == policy_version,
            )
        )
        if row is None:
            return self.default_snapshot(security_scope_id, policy_version)
        return self._snapshot(row)

    def ensure_row(
        self,
        security_scope_id: str,
        policy_version: str,
    ) -> AutoReviewRolloutState:
        if self._db.get_bind().dialect.name == 'postgresql':
            self._db.execute(
                postgresql_insert(AutoReviewRolloutState)
                .values(
                    security_scope_id=security_scope_id,
                    policy_version=policy_version,
                )
                .on_conflict_do_nothing(
                    index_elements=['security_scope_id', 'policy_version']
                )
            )
        else:
            existing = self._db.scalar(
                select(AutoReviewRolloutState).where(
                    AutoReviewRolloutState.security_scope_id
                    == security_scope_id,
                    AutoReviewRolloutState.policy_version == policy_version,
                )
            )
            if existing is None:
                self._db.add(
                    AutoReviewRolloutState(
                        security_scope_id=security_scope_id,
                        policy_version=policy_version,
                    )
                )
                self._db.flush()
        statement = select(AutoReviewRolloutState).where(
            AutoReviewRolloutState.security_scope_id == security_scope_id,
            AutoReviewRolloutState.policy_version == policy_version,
        )
        if self._db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update()
        row = self._db.scalar(statement)
        if row is None:
            raise RolloutGateError('rollout state is unavailable')
        return row

    @staticmethod
    def _snapshot(row: AutoReviewRolloutState) -> RolloutControlSnapshot:
        return RolloutControlSnapshot(
            security_scope_id=row.security_scope_id,
            policy_version=row.policy_version,
            state_version=row.state_version,
            control_epoch=row.control_epoch,
            shadow_predicted_count=row.shadow_predicted_count,
            shadow_completed_count=row.shadow_completed_count,
            shadow_supported_count=row.shadow_supported_count,
            enforce_promotion_ordinal=row.enforce_promotion_ordinal,
            post_audit_selected_count=row.post_audit_selected_count,
            post_audit_completed_count=row.post_audit_completed_count,
            post_audit_critical_count=row.post_audit_critical_count,
            confirmed_mandatory_audit_count=(
                row.confirmed_mandatory_audit_count
            ),
            pending_mandatory_audit_count=row.pending_mandatory_audit_count,
            invalidated_before_audit_count=row.invalidated_before_audit_count,
            corrected_critical_count=row.corrected_critical_count,
            breaker_open=row.breaker_open,
            max_authorized_percentage=row.max_authorized_percentage,
            authorization_generation=row.authorization_generation,
        )

