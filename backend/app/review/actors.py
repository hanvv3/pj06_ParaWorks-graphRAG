from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

from backend.app.core.demo_auth import DemoUser, find_demo_user
from backend.app.core.rbac import (
    PERMISSION_ORDER,
    REVIEW_APPROVAL_PERMISSIONS,
    VALID_PERMISSION_LEVELS,
)

ReviewResolutionCapability: TypeAlias = Literal[
    'human_review',
    'auto_review',
    'auto_review_rollout_admin',
]
ReviewResolutionActorType: TypeAlias = Literal['human', 'auto_policy']

SYSTEM_AUTO_REVIEW_ACTOR_ID = 'system:auto-review'
SYSTEM_AUTO_REVIEW_ACTOR_EMAIL = 'system:auto-review@paraworks.invalid'
SYSTEM_AUTO_REVIEW_ACTOR_ROLE = 'system'

_ACTOR_AUTHORITY = object()
_VALID_ACTOR_TYPES = frozenset({'human', 'auto_policy'})
_VALID_CAPABILITIES = frozenset({
    'human_review',
    'auto_review',
    'auto_review_rollout_admin',
})


@dataclass(frozen=True, init=False)
class ReviewResolutionActor:
    subject_id: str
    actor_type: ReviewResolutionActorType
    allowed_permission_levels: tuple[str, ...]
    capabilities: frozenset[ReviewResolutionCapability]
    policy_version: str | None = None
    _authority: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        subject_id: str,
        actor_type: ReviewResolutionActorType,
        allowed_permission_levels: tuple[str, ...],
        capabilities: frozenset[ReviewResolutionCapability],
        policy_version: str | None = None,
        _authority: object | None = None,
    ) -> None:
        if _authority is not _ACTOR_AUTHORITY:
            raise TypeError(
                'Review resolution actors are minted only by server-owned adapters'
            )
        object.__setattr__(self, 'subject_id', subject_id)
        object.__setattr__(self, 'actor_type', actor_type)
        object.__setattr__(
            self,
            'allowed_permission_levels',
            allowed_permission_levels,
        )
        object.__setattr__(self, 'capabilities', capabilities)
        object.__setattr__(self, 'policy_version', policy_version)
        object.__setattr__(self, '_authority', _authority)
        _assert_review_resolution_actor(self)


@dataclass(frozen=True)
class CreateNewPromotion:
    kind: Literal['create_new'] = 'create_new'

    def __post_init__(self) -> None:
        if self.kind != 'create_new':
            raise ValueError('Create-new promotion directive kind is fixed')


@dataclass(frozen=True)
class ReuseExistingPromotion:
    expected_type: Literal['timeline_event', 'history_event']
    expected_id: int
    expected_claim_fingerprint: str
    expected_companion_id: int | None
    expected_companion_claim_fingerprint: str | None
    kind: Literal['reuse_existing'] = 'reuse_existing'

    def __post_init__(self) -> None:
        if self.kind != 'reuse_existing':
            raise ValueError('Reuse-existing promotion directive kind is fixed')
        if self.expected_type not in {'timeline_event', 'history_event'}:
            raise ValueError('Existing promotion type is unsupported')
        if (
            isinstance(self.expected_id, bool)
            or not isinstance(self.expected_id, int)
            or self.expected_id <= 0
            or not isinstance(self.expected_claim_fingerprint, str)
            or not self.expected_claim_fingerprint
        ):
            raise ValueError('Existing promotion identity is incomplete')
        if (self.expected_companion_id is None) != (
            self.expected_companion_claim_fingerprint is None
        ):
            raise ValueError('Existing companion identity must be complete')
        if self.expected_companion_id is not None and (
            isinstance(self.expected_companion_id, bool)
            or not isinstance(self.expected_companion_id, int)
            or self.expected_companion_id <= 0
        ):
            raise ValueError('Existing companion id must be positive')
        if (
            self.expected_companion_claim_fingerprint is not None
            and (
                not isinstance(
                    self.expected_companion_claim_fingerprint,
                    str,
                )
                or not self.expected_companion_claim_fingerprint
            )
        ):
            raise ValueError('Existing companion identity must be complete')


ApprovalDirective: TypeAlias = CreateNewPromotion | ReuseExistingPromotion


def _assert_approval_directive(directive: ApprovalDirective) -> None:
    if not isinstance(directive, (CreateNewPromotion, ReuseExistingPromotion)):
        raise TypeError('Approval directive has an invalid runtime type')
    directive.__post_init__()


def human_review_actor(user: DemoUser) -> ReviewResolutionActor:
    _assert_known_demo_identity_is_canonical(user)
    capabilities: set[str] = set()
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
        capabilities=cast(
            'frozenset[ReviewResolutionCapability]',
            frozenset(capabilities),
        ),
        _authority=_ACTOR_AUTHORITY,
    )


def auto_review_actor(*, policy_version: str) -> ReviewResolutionActor:
    return ReviewResolutionActor(
        subject_id=SYSTEM_AUTO_REVIEW_ACTOR_ID,
        actor_type='auto_policy',
        allowed_permission_levels=('public', 'internal'),
        capabilities=frozenset({'auto_review'}),
        policy_version=policy_version,
        _authority=_ACTOR_AUTHORITY,
    )


def _assert_review_resolution_actor(actor: ReviewResolutionActor) -> None:
    if not isinstance(actor, ReviewResolutionActor):
        raise TypeError('Review resolution actor has an invalid type')
    if getattr(actor, '_authority', None) is not _ACTOR_AUTHORITY:
        raise TypeError('Review resolution actor authority is not server-owned')
    if actor.actor_type not in _VALID_ACTOR_TYPES:
        raise ValueError('Review resolution actor type is unsupported')
    if not isinstance(actor.subject_id, str) or not actor.subject_id:
        raise ValueError('Review resolution actor requires a subject id')
    if (
        not isinstance(actor.allowed_permission_levels, tuple)
        or not actor.allowed_permission_levels
        or len(set(actor.allowed_permission_levels))
        != len(actor.allowed_permission_levels)
        or any(
            level not in VALID_PERMISSION_LEVELS
            for level in actor.allowed_permission_levels
        )
    ):
        raise ValueError('Review resolution actor permissions are invalid')
    if (
        not isinstance(actor.capabilities, frozenset)
        or any(capability not in _VALID_CAPABILITIES for capability in actor.capabilities)
    ):
        raise ValueError('Review resolution actor capabilities are invalid')
    if actor.actor_type == 'human':
        if actor.subject_id == SYSTEM_AUTO_REVIEW_ACTOR_ID:
            raise ValueError('System identity cannot be projected as a human actor')
        if actor.policy_version is not None or 'auto_review' in actor.capabilities:
            raise ValueError('Human review actor cannot carry auto-review authority')
        return
    if actor.subject_id != SYSTEM_AUTO_REVIEW_ACTOR_ID:
        raise ValueError('Auto-review actor identity is application-owned')
    if actor.allowed_permission_levels != ('public', 'internal'):
        raise ValueError('Auto-review actor permissions are fixed')
    if actor.capabilities != frozenset({'auto_review'}):
        raise ValueError('Auto-review actor capabilities are fixed')
    if not isinstance(actor.policy_version, str) or not actor.policy_version:
        raise ValueError('Auto-review actor requires a policy version')


def _assert_known_demo_identity_is_canonical(user: DemoUser) -> None:
    if not isinstance(user, DemoUser):
        raise TypeError('Human review adapter requires an authenticated DemoUser')
    canonical = find_demo_user(user.id) or find_demo_user(user.email)
    if canonical is not None and canonical != user:
        raise ValueError('Authenticated DemoUser does not match canonical identity')
