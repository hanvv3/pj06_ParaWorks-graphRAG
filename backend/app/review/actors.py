from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Literal, NoReturn, TypeAlias
from weakref import ReferenceType, ref

from backend.app.core.demo_auth import DemoUser, _assert_authenticated_demo_user
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

_VALID_ACTOR_TYPES = frozenset({'human', 'auto_policy'})
_VALID_CAPABILITIES = frozenset({
    'human_review',
    'auto_review',
    'auto_review_rollout_admin',
})


@dataclass(frozen=True, slots=True, weakref_slot=True, init=False)
class ReviewResolutionActor:
    subject_id: str
    actor_type: ReviewResolutionActorType
    allowed_permission_levels: tuple[str, ...]
    capabilities: frozenset[ReviewResolutionCapability]
    policy_version: str | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            'Review resolution actors are minted only by server-owned adapters'
        )

    def __copy__(self) -> NoReturn:
        raise TypeError('Review resolution actors cannot be copied')

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError('Review resolution actors cannot be copied')

    def __reduce__(self) -> NoReturn:
        raise TypeError('Review resolution actors cannot be serialized')

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError('Review resolution actors cannot be serialized')


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


def _assert_actor_shape(actor: ReviewResolutionActor) -> None:
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


def _build_actor_boundary() -> tuple[
    Callable[[DemoUser], ReviewResolutionActor],
    Callable[..., ReviewResolutionActor],
    Callable[[ReviewResolutionActor], None],
]:
    issued: dict[
        int,
        tuple[ReferenceType[ReviewResolutionActor], tuple[object, ...]],
    ] = {}
    lock = RLock()

    def register(actor: ReviewResolutionActor) -> ReviewResolutionActor:
        _assert_actor_shape(actor)
        actor_id = id(actor)

        def discard(
            reference: ReferenceType[ReviewResolutionActor],
            actor_identity: int = actor_id,
        ) -> None:
            with lock:
                entry = issued.get(actor_identity)
                if entry is not None and entry[0] is reference:
                    issued.pop(actor_identity, None)

        reference = ref(actor, discard)
        snapshot: tuple[object, ...] = (
            actor.subject_id,
            actor.actor_type,
            actor.allowed_permission_levels,
            actor.capabilities,
            actor.policy_version,
        )
        with lock:
            existing = issued.get(actor_id)
            if existing is not None and existing[0]() is not None:
                raise RuntimeError('Review actor identity registry collision')
            issued[actor_id] = (reference, snapshot)
        return actor

    def adapt_human(user: DemoUser) -> ReviewResolutionActor:
        _assert_authenticated_demo_user(user)
        capabilities: set[ReviewResolutionCapability] = set()
        if user.role in REVIEW_APPROVAL_PERMISSIONS:
            capabilities.add('human_review')
        if user.role == 'admin':
            capabilities.add('auto_review_rollout_admin')
        actor = object.__new__(ReviewResolutionActor)
        object.__setattr__(actor, 'subject_id', user.id)
        object.__setattr__(actor, 'actor_type', 'human')
        object.__setattr__(
            actor,
            'allowed_permission_levels',
            tuple(
                sorted(
                    user.permission_levels,
                    key=lambda value: (
                        PERMISSION_ORDER.get(value, 10_000),
                        value,
                    ),
                )
            ),
        )
        object.__setattr__(actor, 'capabilities', frozenset(capabilities))
        object.__setattr__(actor, 'policy_version', None)
        return register(actor)

    def create_auto(*, policy_version: str) -> ReviewResolutionActor:
        actor = object.__new__(ReviewResolutionActor)
        object.__setattr__(actor, 'subject_id', SYSTEM_AUTO_REVIEW_ACTOR_ID)
        object.__setattr__(actor, 'actor_type', 'auto_policy')
        object.__setattr__(
            actor,
            'allowed_permission_levels',
            ('public', 'internal'),
        )
        object.__setattr__(actor, 'capabilities', frozenset({'auto_review'}))
        object.__setattr__(actor, 'policy_version', policy_version)
        return register(actor)

    def validate_issued(actor: ReviewResolutionActor) -> None:
        if not isinstance(actor, ReviewResolutionActor):
            raise TypeError('Review resolution actor has an invalid type')
        with lock:
            entry = issued.get(id(actor))
            if entry is None or entry[0]() is not actor:
                raise TypeError(
                    'Review resolution actor was not issued by the server'
                )
            try:
                current: tuple[object, ...] = (
                    actor.subject_id,
                    actor.actor_type,
                    actor.allowed_permission_levels,
                    actor.capabilities,
                    actor.policy_version,
                )
            except AttributeError as exc:
                raise TypeError(
                    'Review resolution actor was not issued by the server'
                ) from exc
            if current != entry[1]:
                raise TypeError(
                    'Review resolution actor was not issued by the server'
                )
        _assert_actor_shape(actor)

    return adapt_human, create_auto, validate_issued


(
    human_review_actor,
    auto_review_actor,
    _assert_review_resolution_actor,
) = _build_actor_boundary()
del _build_actor_boundary
