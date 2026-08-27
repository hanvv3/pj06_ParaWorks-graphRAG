import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
from sqlalchemy.orm import Session

from backend.app.connectors.slack_oauth import (
    LocalTokenVault,
    mask_secret,
    pkce_challenge,
)
from backend.app.core.config import Settings
from backend.app.models import IntegrationConnection

GOOGLE_IDENTITY_SCOPES = ('openid', 'email')
GOOGLE_DATA_SCOPES: dict[str, tuple[str, ...]] = {
    'gmail': (
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
    ),
    'drive': ('https://www.googleapis.com/auth/drive.readonly',),
    'calendar': ('https://www.googleapis.com/auth/calendar.readonly',),
}
GOOGLE_OAUTH_SCOPES: dict[str, tuple[str, ...]] = {
    connector_type: (*GOOGLE_IDENTITY_SCOPES, *data_scopes)
    for connector_type, data_scopes in GOOGLE_DATA_SCOPES.items()
}
GOOGLE_OAUTH_CONNECTOR_TYPES = frozenset(GOOGLE_DATA_SCOPES)


@dataclass(frozen=True)
class GoogleOAuthState:
    connector_type: str
    nonce: str
    code_verifier: str | None = None


@dataclass(frozen=True)
class GoogleOAuthInstallUrl:
    connector_type: str
    install_url: str
    state: str
    required_scopes: list[str]
    configured: bool
    code_verifier: str | None = None


@dataclass(frozen=True)
class GoogleOAuthAccess:
    access_token: str
    refresh_token: str | None
    account_id: str
    account_name: str
    scopes: list[str]


class GoogleOAuthConfigurationError(RuntimeError):
    pass


class GoogleOAuthError(RuntimeError):
    pass


class GoogleOAuthStateSigner:
    def __init__(self, secret: str) -> None:
        self.secret = secret.encode()

    def create(
        self,
        *,
        connector_type: str,
        nonce: str | None = None,
        code_verifier: str | None = None,
    ) -> str:
        _validate_google_connector_type(connector_type)
        payload = {
            'connector_type': connector_type,
            'nonce': nonce or secrets.token_urlsafe(18),
        }
        if code_verifier:
            payload['code_verifier'] = code_verifier

        payload_token = _b64encode(json.dumps(payload, separators=(',', ':')).encode())
        signature = _sign(payload_token, self.secret)
        return f'{payload_token}.{signature}'

    def validate(self, state: str) -> GoogleOAuthState:
        try:
            payload_token, signature = state.split('.', 1)
        except ValueError as exc:
            raise GoogleOAuthError('Google OAuth state is malformed') from exc

        expected = _sign(payload_token, self.secret)
        if not hmac.compare_digest(signature, expected):
            raise GoogleOAuthError('Google OAuth state signature is invalid')

        payload = json.loads(_b64decode(payload_token))
        connector_type = str(payload['connector_type'])
        _validate_google_connector_type(connector_type)
        return GoogleOAuthState(
            connector_type=connector_type,
            nonce=str(payload['nonce']),
            code_verifier=payload.get('code_verifier'),
        )


class GoogleOAuthClient:
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        http_client: httpx.Client | None = None,
        token_url: str = 'https://oauth2.googleapis.com/token',
        userinfo_url: str = 'https://openidconnect.googleapis.com/v1/userinfo',
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.http_client = http_client or httpx.Client(timeout=30.0)
        self.token_url = token_url
        self.userinfo_url = userinfo_url

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        scopes: list[str],
        code_verifier: str | None = None,
    ) -> GoogleOAuthAccess:
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': redirect_uri,
        }
        if code_verifier:
            data['code_verifier'] = code_verifier

        response = self.http_client.post(
            self.token_url,
            data=data,
        )
        response.raise_for_status()
        token_payload = response.json()
        access_token = str(token_payload['access_token'])
        userinfo_response = self.http_client.get(
            self.userinfo_url,
            headers={'Authorization': f'Bearer {access_token}'},
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()

        return GoogleOAuthAccess(
            access_token=access_token,
            refresh_token=token_payload.get('refresh_token'),
            account_id=str(userinfo.get('sub') or userinfo.get('email')),
            account_name=str(userinfo.get('email') or userinfo.get('name') or 'Google Workspace'),
            scopes=_parse_scope_string(str(token_payload.get('scope') or ' '.join(scopes))),
        )

    def refresh_access_token(
        self,
        *,
        refresh_token: str,
        scopes: list[str] | None = None,
    ) -> GoogleOAuthAccess:
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token',
        }
        response = self.http_client.post(
            self.token_url,
            data=data,
        )
        response.raise_for_status()
        token_payload = response.json()
        access_token = str(token_payload['access_token'])

        # Refresh response might not include a new refresh token or user info
        userinfo_response = self.http_client.get(
            self.userinfo_url,
            headers={'Authorization': f'Bearer {access_token}'},
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()

        return GoogleOAuthAccess(
            access_token=access_token,
            refresh_token=token_payload.get('refresh_token') or refresh_token,
            account_id=str(userinfo.get('sub') or userinfo.get('email')),
            account_name=str(userinfo.get('email') or userinfo.get('name') or 'Google Workspace'),
            scopes=_parse_scope_string(str(token_payload.get('scope') or ('' if scopes is None else ' '.join(scopes)))),
        )


def build_google_oauth_install_url(
    *,
    settings: Settings,
    connector_type: str,
    nonce: str | None = None,
    redirect_uri: str | None = None,
    use_pkce: bool = True,
) -> GoogleOAuthInstallUrl:
    _validate_google_connector_type(connector_type)
    if not settings.google_client_id:
        raise GoogleOAuthConfigurationError('GOOGLE_CLIENT_ID is required for Google OAuth install URL')

    target_redirect_uri = redirect_uri or settings.google_oauth_redirect_uri
    
    # Non-web URI인 경우 PKCE 강제 사용
    is_non_web = not (target_redirect_uri.startswith('http://') or target_redirect_uri.startswith('https://'))
    actual_use_pkce = use_pkce or is_non_web
    
    code_verifier = secrets.token_urlsafe(64) if actual_use_pkce else None

    state = GoogleOAuthStateSigner(settings.google_oauth_state_secret).create(
        connector_type=connector_type,
        nonce=nonce,
        code_verifier=code_verifier,
    )
    scopes = list(GOOGLE_OAUTH_SCOPES[connector_type])
    params = {
        'client_id': settings.google_client_id,
        'redirect_uri': target_redirect_uri,
        'response_type': 'code',
        'scope': ' '.join(scopes),
        'access_type': 'offline',
        'prompt': 'consent',
        'state': state,
    }

    if code_verifier:
        params['code_challenge'] = pkce_challenge(code_verifier)
        params['code_challenge_method'] = 'S256'

    query = urlencode(params)
    return GoogleOAuthInstallUrl(
        connector_type=connector_type,
        install_url=f'https://accounts.google.com/o/oauth2/v2/auth?{query}',
        state=state,
        required_scopes=scopes,
        configured=True,
        code_verifier=code_verifier,
    )


def complete_google_oauth_callback(
    *,
    db: Session,
    settings: Settings,
    connector_type: str,
    code: str,
    state: str,
    access: GoogleOAuthAccess | None = None,
    token_vault: LocalTokenVault,
    redirect_uri: str | None = None,
) -> IntegrationConnection:
    _validate_google_connector_type(connector_type)
    parsed_state = GoogleOAuthStateSigner(settings.google_oauth_state_secret).validate(state)
    if parsed_state.connector_type != connector_type:
        raise GoogleOAuthError('Google OAuth state connector mismatch')

    scopes = list(GOOGLE_OAUTH_SCOPES[connector_type])
    access_payload = access or _exchange_google_code(
        settings=settings,
        code=code,
        scopes=scopes,
        code_verifier=parsed_state.code_verifier,
        redirect_uri=redirect_uri,
    )
    persisted_token = access_payload.refresh_token or access_payload.access_token
    token_ref = token_vault.store_token(
        connector_type=connector_type,
        workspace_id=access_payload.account_id,
        token=persisted_token,
        token_kind='oauth',
    )

    connection = (
        db.query(IntegrationConnection)
        .filter(
            IntegrationConnection.connector_type == connector_type,
            IntegrationConnection.workspace_id == access_payload.account_id,
        )
        .one_or_none()
    )
    if connection is None:
        connection = IntegrationConnection(
            connector_type=connector_type,
            workspace_id=access_payload.account_id,
            workspace_name=access_payload.account_name,
            workspace_url='https://workspace.google.com',
            token_ref=token_ref,
            masked_bot_token=mask_secret(persisted_token),
        )
        db.add(connection)

    connection.workspace_name = access_payload.account_name
    connection.workspace_url = 'https://workspace.google.com'
    connection.bot_user_id = None
    connection.scopes = access_payload.scopes
    connection.token_ref = token_ref
    connection.masked_bot_token = mask_secret(persisted_token)
    connection.status = 'connected'
    connection.raw_metadata = {
        'state_nonce': parsed_state.nonce,
        'scope_count': len(access_payload.scopes),
        'token_kind': 'refresh_token' if access_payload.refresh_token else 'access_token',
        'pkce_used': parsed_state.code_verifier is not None,
    }

    db.commit()
    db.refresh(connection)
    return connection


def _exchange_google_code(
    *,
    settings: Settings,
    code: str,
    scopes: list[str],
    code_verifier: str | None = None,
    redirect_uri: str | None = None,
) -> GoogleOAuthAccess:
    if not settings.google_client_id or not settings.google_client_secret:
        raise GoogleOAuthConfigurationError('Google OAuth client id and secret are required for callback exchange')
    return GoogleOAuthClient(
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
    ).exchange_code(
        code=code,
        redirect_uri=redirect_uri or settings.google_oauth_redirect_uri,
        scopes=scopes,
        code_verifier=code_verifier,
    )


def _validate_google_connector_type(connector_type: str) -> None:
    if connector_type not in GOOGLE_OAUTH_CONNECTOR_TYPES:
        raise GoogleOAuthError(f'Unsupported Google OAuth connector: {connector_type}')


def _parse_scope_string(scope: str) -> list[str]:
    return [item.strip() for item in scope.replace(',', ' ').split() if item.strip()]


def _sign(payload_token: str, secret: bytes) -> str:
    digest = hmac.new(secret, payload_token.encode(), hashlib.sha256).digest()
    return _b64encode(digest)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip('=')


def _b64decode(value: str) -> bytes:
    padded = value + ('=' * (-len(value) % 4))
    return base64.urlsafe_b64decode(padded.encode())
