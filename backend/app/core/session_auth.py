import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.profile import profile_avatar_url
from backend.app.models import AuthUser, RefreshToken


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def upsert_auth_user_from_demo(db: Session, demo_user) -> AuthUser:
    auth_user = db.scalar(select(AuthUser).where(AuthUser.email == demo_user.email))
    if auth_user is None:
        auth_user = AuthUser(
            external_id=demo_user.id,
            email=demo_user.email,
            display_name=demo_user.name,
            role=demo_user.role,
            department=demo_user.department,
            title=demo_user.title,
            status='active',
            permission_levels=sorted(demo_user.permission_levels),
        )
        db.add(auth_user)
    else:
        auth_user.external_id = demo_user.id
        auth_user.display_name = demo_user.name
        auth_user.role = demo_user.role
        auth_user.department = demo_user.department
        auth_user.title = demo_user.title
        auth_user.status = 'active'
        auth_user.permission_levels = sorted(demo_user.permission_levels)
    db.flush()
    return auth_user


def serialize_auth_user(user: AuthUser) -> dict:
    return {
        'id': user.external_id,
        'email': user.email,
        'role': user.role,
        'permission_levels': sorted(user.permission_levels),
        'name': user.display_name,
        'title': user.title,
        'department': user.department,
        'status': user.status,
        'avatar_url': profile_avatar_url(user.email, user.role),
    }


def issue_auth_cookies(response: Response, db: Session, auth_user: AuthUser, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    session_token = create_session_token(auth_user.id, settings)
    refresh_token = secrets.token_urlsafe(48)
    refresh_record = RefreshToken(
        user_id=auth_user.id,
        token_hash=hash_refresh_token(refresh_token),
        family_id=secrets.token_hex(16),
        expires_at=datetime.now(UTC) + timedelta(seconds=settings.auth_refresh_ttl_seconds),
    )
    db.add(refresh_record)
    set_auth_cookies(response, session_token, refresh_token, settings)
    return refresh_token


def rotate_refresh_token(response: Response, db: Session, refresh_token: str, settings: Settings | None = None) -> AuthUser | None:
    settings = settings or get_settings()
    now = datetime.now(UTC)
    token_record = db.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == hash_refresh_token(refresh_token),
            RefreshToken.revoked_at.is_(None),
        )
    )
    if token_record is None or _as_utc(token_record.expires_at) <= now:
        return None

    auth_user = db.get(AuthUser, token_record.user_id)
    if auth_user is None or auth_user.status != 'active':
        return None

    new_refresh_token = secrets.token_urlsafe(48)
    replacement = RefreshToken(
        user_id=auth_user.id,
        token_hash=hash_refresh_token(new_refresh_token),
        family_id=token_record.family_id,
        expires_at=now + timedelta(seconds=settings.auth_refresh_ttl_seconds),
    )
    db.add(replacement)
    db.flush()
    token_record.revoked_at = now
    token_record.last_used_at = now
    token_record.replaced_by_token_id = replacement.id
    session_token = create_session_token(auth_user.id, settings)
    set_auth_cookies(response, session_token, new_refresh_token, settings)
    return auth_user


def revoke_refresh_token_family(db: Session, refresh_token: str | None) -> None:
    if not refresh_token:
        return
    token_record = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token)))
    if token_record is None:
        return
    now = datetime.now(UTC)
    for family_token in db.scalars(
        select(RefreshToken).where(
            RefreshToken.family_id == token_record.family_id,
            RefreshToken.revoked_at.is_(None),
        )
    ):
        family_token.revoked_at = now


def clear_auth_cookies(response: Response, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    response.delete_cookie(settings.auth_session_cookie_name, path='/')
    response.delete_cookie(settings.auth_refresh_cookie_name, path='/')
    response.delete_cookie(settings.auth_csrf_cookie_name, path='/')


def authenticate_session_cookie(session_token: str | None, db: Session, settings: Settings | None = None) -> AuthUser | None:
    if not session_token:
        return None
    settings = settings or get_settings()
    user_id = verify_session_token(session_token, settings)
    if user_id is None:
        return None
    auth_user = db.get(AuthUser, user_id)
    if auth_user is None or auth_user.status != 'active':
        return None
    return auth_user


def create_session_token(user_id: int, settings: Settings) -> str:
    payload = {
        'sub': user_id,
        'exp': int((datetime.now(UTC) + timedelta(seconds=settings.auth_session_ttl_seconds)).timestamp()),
    }
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    encoded_payload = base64.urlsafe_b64encode(payload_bytes).decode('ascii').rstrip('=')
    signature = _sign(encoded_payload, settings.auth_session_secret)
    return f'{encoded_payload}.{signature}'


def verify_session_token(token: str, settings: Settings) -> int | None:
    try:
        encoded_payload, signature = token.split('.', 1)
    except ValueError:
        return None
    expected = _sign(encoded_payload, settings.auth_session_secret)
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padding = '=' * (-len(encoded_payload) % 4)
        payload = json.loads(base64.urlsafe_b64decode(f'{encoded_payload}{padding}').decode('utf-8'))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(payload.get('exp', 0)) <= int(datetime.now(UTC).timestamp()):
        return None
    try:
        return int(payload['sub'])
    except (KeyError, TypeError, ValueError):
        return None


def set_auth_cookies(response: Response, session_token: str, refresh_token: str, settings: Settings) -> None:
    response.set_cookie(
        settings.auth_session_cookie_name,
        session_token,
        max_age=settings.auth_session_ttl_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite='lax',
        path='/',
    )
    response.set_cookie(
        settings.auth_refresh_cookie_name,
        refresh_token,
        max_age=settings.auth_refresh_ttl_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite='lax',
        path='/',
    )
    set_csrf_cookie(response, settings)


async def check_csrf(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
):
    if request.method in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
        return True
    # Exempt login and refresh from established session CSRF
    if request.url.path in ('/api/v1/auth/login', '/api/v1/auth/refresh'):
        return True
    if not verify_csrf_token(request, settings):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


def set_csrf_cookie(response: Response, settings: Settings) -> str:
    token = secrets.token_urlsafe(32)
    response.set_cookie(
        settings.auth_csrf_cookie_name,
        token,
        max_age=settings.auth_session_ttl_seconds,
        httponly=False,
        secure=settings.auth_cookie_secure,
        samesite='lax',
        path='/',
    )
    return token


def verify_csrf_token(request: Request, settings: Settings) -> bool:
    if request.method in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
        return True
    cookie_token = request.cookies.get(settings.auth_csrf_cookie_name)
    header_token = request.headers.get(settings.auth_csrf_header_name)
    if not cookie_token or not header_token:
        return False
    return hmac.compare_digest(cookie_token, header_token)


def _sign(encoded_payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode('utf-8'), encoded_payload.encode('ascii'), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
