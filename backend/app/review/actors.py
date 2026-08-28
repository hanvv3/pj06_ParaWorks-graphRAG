from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from backend.app.core.demo_auth import DemoUser
from backend.app.core.rbac import PERMISSION_ORDER, REVIEW_APPROVAL_PERMISSIONS

ReviewResolutionCapability: TypeAlias = Literal[
    'human_review',
    'auto_review',
    'auto_review_rollout_admin',
]
ReviewResolutionActorType: TypeAlias = Literal['human', 'auto_policy']

SYSTEM_AUTO_REVIEW_ACTOR_ID = 'system:auto-review'
SYSTEM_AUTO_REVIEW_ACTOR_EMAIL = 'system:auto-review@paraworks.invalid'
SYSTEM_AUTO_REVIEW_ACTOR_ROLE = 'system'


@dataclass(frozen=True)
class ReviewResolutionActor:
    subject_id: str
    actor_type: ReviewResolutionActorType
    allowed_permission_levels: tuple[str, ...]
    capabilities: frozenset[ReviewResolutionCapability]
    policy_version: str | None = None

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError('Review resolution actor requires a subject id')
        if len(set(self.allowed_permission_levels)) != len(
            self.allowed_permission_levels
        ):
            raise ValueError('Review resolution permissions must be unique')
        if self.actor_type == 'human':
            if self.policy_version is not None or 'auto_review' in self.capabilities:
                raise ValueError('Human review actor cannot carry auto-review authority')
            return
        if self.subject_id != SYSTEM_AUTO_REVIEW_ACTOR_ID:
            raise ValueError('Auto-review actor identity is application-owned')
        if self.allowed_permission_levels != ('public', 'internal'):
            raise ValueError('Auto-review actor permissions are fixed')
        if self.capabilities != frozenset({'auto_review'}):
            raise ValueError('Auto-review actor capabilities are fixed')
        if not self.policy_version:
            raise ValueError('Auto-review actor requires a policy version')


@dataclass(frozen=True)
class CreateNewPromotion:
    kind: Literal['create_new'] = 'create_new'


@dataclass(frozen=True)
class ReuseExistingPromotion:
    expected_type: Literal['timeline_event', 'history_event']
    expected_id: int
    expected_claim_fingerprint: str
    expected_companion_id: int | None
    expected_companion_claim_fingerprint: str | None
    kind: Literal['reuse_existing'] = 'reuse_existing'

    def __post_init__(self) -> None:
        if self.expected_id <= 0 or not self.expected_claim_fingerprint:
            raise ValueError('Existing promotion identity is incomplete')
        if (self.expected_companion_id is None) != (
            self.expected_companion_claim_fingerprint is None
        ):
            raise ValueError('Existing companion identity must be complete')
        if self.expected_companion_id is not None and self.expected_companion_id <= 0:
            raise ValueError('Existing companion id must be positive')


ApprovalDirective: TypeAlias = CreateNewPromotion | ReuseExistingPromotion


def human_review_actor(user: DemoUser) -> ReviewResolutionActor:
    capabilities: set[ReviewResolutionCapability] = set()
    if user.role in REVIEW_APPROVAL_PERMISSIONS:
        capabilities.add('human_review')
    if user.role == 'admin':
        capabilities.add('auto_review_rollout_admin')
    return ReviewResolutionActor(
        subject_id=user.id,
        actor_type='human',
        allowed_permission_levels=tuple(
            sorted(
                user.permission_levels,
                key=lambda value: (PERMISSION_ORDER.get(value, 10_000), value),
            )
        ),
        capabilities=frozenset(capabilities),
    )


def auto_review_actor(*, policy_version: str) -> ReviewResolutionActor:
    return ReviewResolutionActor(
        subject_id=SYSTEM_AUTO_REVIEW_ACTOR_ID,
        actor_type='auto_policy',
        allowed_permission_levels=('public', 'internal'),
        capabilities=frozenset({'auto_review'}),
        policy_version=policy_version,
    )
