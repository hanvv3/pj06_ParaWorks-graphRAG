from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Annotated
from weakref import ReferenceType, ref

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import inspect as inspect_sqlalchemy
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.profile import profile_avatar_url
from backend.app.core.session_auth import (
    authenticate_session_cookie,
    serialize_auth_user,
)
from backend.app.db.session import get_db
from backend.app.models import AuthUser


@dataclass(frozen=True)
class DemoUser:
    id: str
    email: str
    role: str
    permission_levels: set[str]
    name: str
    title: str
    department: str
    aliases: tuple[str, ...] = ()


USERS = {
    'admin': DemoUser(
        'demo-admin',
        'admin@paraworks.com',
        'admin',
        {'public', 'internal', 'restricted'},
        'ParaWorks Admin',
        'Workspace Administrator',
        'Platform',
    ),
    'hanvv-admin': DemoUser(
        'google-hanvv-admin',
        'hanvv3@gmail.com',
        'admin',
        {'public', 'internal', 'restricted'},
        'Hanvv Admin',
        'Workspace Administrator',
        'Platform',
        ('한승헌', '승헌', 'SeungHun Han', 'Han SeungHun'),
    ),
    'kjw4work': DemoUser(
        'kjw4work',
        'kjw4work@gmail.com',
        'admin',
        {'public', 'internal', 'restricted'},
        'Kim Jongwoo',
        'COO',
        'platform',
        ('김종우', '종우', '김종우 COO', 'COO 김종우'),
    ),
    'yonghee199702': DemoUser(
        'yonghee199702',
        'yonghee199702@gmail.com',
        'admin',
        {'public', 'internal', 'restricted'},
        'Kim Yonghee',
        'CTO',
        'platform',
        ('김용희', '용희', '김용희 CTO', 'CTO 김용희'),
    ),
    'hanvv-employee': DemoUser(
        'google-hanvv-employee',
        'hanvv3@koreacu.ac.kr',
        'employee',
        {'public', 'internal'},
        'Hanvv Employee',
        'AI Agent Developer',
        'Engineering',
        ('한준혁', '준혁', '한준혁 개발자', 'AI Agent Developer 한준혁'),
    ),
    'viewer': DemoUser(
        'employee-mina',
        'mina@paraworks.com',
        'reviewer',
        {'public', 'internal'},
        'Kim Mina',
        'Product Manager',
        'Product',
        ('김미나', '미나', '김미나 PM', 'PM 김미나'),
    ),
}


def find_demo_user(value: str) -> DemoUser | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in USERS:
        return USERS[normalized]
    return next(
        (
            user
            for user in USERS.values()
            if user.id.lower() == normalized or user.email.lower() == normalized
        ),
        None,
    )


def authenticate_demo_user(email: str) -> DemoUser:
    user = find_demo_user(email)
    if user is None or user.email.lower() != email.strip().lower():
        raise HTTPException(status_code=401, detail='Demo account not found.')
    return user


def serialize_demo_user(user: DemoUser) -> dict:
    return {
        'id': user.id,
        'email': user.email,
        'role': user.role,
        'permission_levels': sorted(user.permission_levels),
        'name': user.name,
        'title': user.title,
        'department': user.department,
        'avatar_url': profile_avatar_url(user.email, user.role),
    }


def demo_user_from_serialized(payload: dict) -> DemoUser:
    return DemoUser(
        id=payload['id'],
        email=payload['email'],
        role=payload['role'],
        permission_levels=set(payload['permission_levels']),
        name=payload['name'],
        title=payload['title'],
        department=payload['department'],
        aliases=tuple(payload.get('aliases', ())),
    )


def _build_authenticated_demo_user_boundary() -> tuple[
    Callable[..., DemoUser],
    Callable[[DemoUser], None],
]:
    authenticated: dict[
        int,
        tuple[ReferenceType[DemoUser], tuple[object, ...]],
    ] = {}
    lock = RLock()

    def snapshot(user: DemoUser) -> tuple[object, ...]:
        return (
            user.id,
            user.email,
            user.role,
            frozenset(user.permission_levels),
            user.name,
            user.title,
            user.department,
            user.aliases,
        )

    static_identities = {
        id(user): (user, snapshot(user)) for user in USERS.values()
    }

    def register_session_projection(user: DemoUser) -> DemoUser:
        user_id = id(user)

        def discard(
            reference: ReferenceType[DemoUser],
            user_identity: int = user_id,
        ) -> None:
            with lock:
                entry = authenticated.get(user_identity)
                if entry is not None and entry[0] is reference:
                    authenticated.pop(user_identity, None)

        reference = ref(user, discard)
        with lock:
            existing = authenticated.get(user_id)
            if existing is not None and existing[0]() is not None:
                raise RuntimeError(
                    'Authenticated DemoUser identity registry collision'
                )
            authenticated[user_id] = (reference, snapshot(user))
        return user

    def get_authenticated_user(
        request: Request,
        db: Annotated[Session, Depends(get_db)],
        x_demo_user: Annotated[str, Header()] = 'admin',
    ) -> DemoUser:
        settings = get_settings()
        session_user = authenticate_session_cookie(
            request.cookies.get(settings.auth_session_cookie_name),
            db,
            settings,
        )
        if session_user is not None:
            state = inspect_sqlalchemy(session_user)
            if (
                not isinstance(session_user, AuthUser)
                or not state.persistent
                or state.detached
                or state.session is not db
                or session_user.id is None
            ):
                raise RuntimeError(
                    'Session authentication did not return a persistent user'
                )
            return register_session_projection(
                demo_user_from_serialized(serialize_auth_user(session_user))
            )
        if not settings.paraworks_demo_mode:
            raise HTTPException(
                status_code=401,
                detail='Authentication required.',
            )
        return find_demo_user(x_demo_user) or USERS['viewer']

    def assert_authenticated(user: DemoUser) -> None:
        if not isinstance(user, DemoUser):
            raise TypeError('Human review adapter requires a DemoUser')
        static_entry = static_identities.get(id(user))
        if static_entry is not None and static_entry[0] is user:
            if snapshot(user) != static_entry[1]:
                raise ValueError(
                    'Authenticated DemoUser claims changed after authentication'
                )
            return
        with lock:
            entry = authenticated.get(id(user))
            if entry is None or entry[0]() is not user:
                raise ValueError(
                    'DemoUser is not an authenticated canonical identity'
                )
            if snapshot(user) != entry[1]:
                raise ValueError(
                    'Authenticated DemoUser claims changed after authentication'
                )

    return get_authenticated_user, assert_authenticated


get_demo_user, _assert_authenticated_demo_user = (
    _build_authenticated_demo_user_boundary()
)
del _build_authenticated_demo_user_boundary


def list_demo_users() -> list[dict]:
    return [serialize_demo_user(user) for user in USERS.values()]


def require_admin_user(user: Annotated[DemoUser, Depends(get_demo_user)]) -> DemoUser:
    if user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin permission required.')
    return user
