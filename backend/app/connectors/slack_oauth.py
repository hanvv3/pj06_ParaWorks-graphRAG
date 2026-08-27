import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
from sqlalchemy.orm import Session

from backend.app.connectors.slack import SLACK_REQUIRED_SCOPES, SlackApiError
from backend.app.core.config import Settings
from backend.app.models import IntegrationConnection

SLACK_OAUTH_BOT_SCOPES = SLACK_REQUIRED_SCOPES


@dataclass(frozen=True)
class SlackOAuthState:
    connector_type: str
    nonce: str
    code_verifier: str | None = None


@dataclass(frozen=True)
class SlackOAuthInstallUrl:
    connector_type: str
    install_url: str
    state: str
    required_scopes: list[str]
    configured: bool
    code_verifier: str | None = None


@dataclass(frozen=True)
class SlackOAuthAccess:
    bot_token: str
    bot_user_id: str | None
    team_id: str
    team_name: str
    scopes: list[str]
    user_token: str | None = None
    user_id: str | None = None


class SlackOAuthConfigurationError(RuntimeError):
    pass


class SlackOAuthStateSigner:
    def __init__(self, secret: str) -> None:
        self.secret = secret.encode()

    def create(
        self,
        *,
        nonce: str | None = None,
        connector_type: str = 'slack',
        code_verifier: str | None = None,
    ) -> str:
        payload = {
            'connector_type': connector_type,
            'nonce': nonce or secrets.token_urlsafe(18),
        }
        if code_verifier:
            payload['code_verifier'] = code_verifier

        payload_token = _b64encode(json.dumps(payload, separators=(',', ':')).encode())
        signature = _sign(payload_token, self.secret)
        return f'{payload_token}.{signature}'

    def validate(self, state: str) -> SlackOAuthState:
        try:
            payload_token, signature = state.split('.', 1)
        except ValueError as exc:
            raise SlackApiError('Slack OAuth state is malformed') from exc

        expected = _sign(payload_token, self.secret)
        if not hmac.compare_digest(signature, expected):
            raise SlackApiError('Slack OAuth state signature is invalid')

        payload = json.loads(_b64decode(payload_token))
        return SlackOAuthState(
            connector_type=str(payload['connector_type']),
            nonce=str(payload['nonce']),
            code_verifier=payload.get('code_verifier'),
        )


class SlackOAuthClient:
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        http_client: httpx.Client | None = None,
        base_url: str = 'https://slack.com/api',
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.http_client = http_client or httpx.Client(timeout=30.0)
        self.base_url = base_url.rstrip('/')

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str | None = None,
    ) -> SlackOAuthAccess:
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': redirect_uri,
        }
        if code_verifier:
            data['code_verifier'] = code_verifier

        response = self.http_client.post(
            f'{self.base_url}/oauth.v2.access',
            data=data,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get('ok'):
            raise SlackApiError(f"Slack oauth.v2.access failed: {payload.get('error', 'unknown_error')}")

        team = payload.get('team') or {}
        authed_user = payload.get('authed_user') or {}
        
        return SlackOAuthAccess(
            bot_token=str(payload.get('access_token') or ''),
            bot_user_id=payload.get('bot_user_id'),
            team_id=str(team.get('id') or payload.get('team_id') or ''),
            team_name=str(team.get('name') or 'Slack workspace'),
            scopes=_parse_scope_string(str(payload.get('scope') or '')),
            user_token=authed_user.get('access_token'),
            user_id=authed_user.get('id'),
        )


class LocalTokenVault:
    def __init__(self, storage_path: str = '.tokens.json') -> None:
        self.storage_path = storage_path
        self._secrets: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path) as f:
                    self._secrets = json.load(f)
        except Exception:
            self._secrets = {}

    def _save(self) -> None:
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self._secrets, f, indent=2)
        except Exception:
            pass

    def store_token(self, *, connector_type: str, workspace_id: str, token: str, token_kind: str) -> str:
        token_ref = f'local:{connector_type}:{workspace_id}:{token_kind}'
        self._secrets[token_ref] = token
        self._save()
        return token_ref

    def store_bot_token(self, *, connector_type: str, workspace_id: str, token: str) -> str:
        return self.store_token(
            connector_type=connector_type,
            workspace_id=workspace_id,
            token=token,
            token_kind='bot',
        )

    def store_user_token(self, *, connector_type: str, workspace_id: str, token: str) -> str:
        return self.store_token(
            connector_type=connector_type,
            workspace_id=workspace_id,
            token=token,
            token_kind='user',
        )

    def resolve(self, token_ref: str) -> str | None:
        return self._secrets.get(token_ref)

    def remove_token(self, token_ref: str) -> bool:
        if token_ref in self._secrets:
            del self._secrets[token_ref]
            self._save()
            return True
        return False


LOCAL_TOKEN_VAULT = LocalTokenVault()


def pkce_challenge(verifier: str) -> str:
    """Generates a code_challenge from a code_verifier using S256."""
    sha256_hash = hashlib.sha256(verifier.encode('ascii')).digest()
    return _b64encode(sha256_hash)


def build_slack_oauth_install_url(
    *,
    settings: Settings,
    nonce: str | None = None,
    redirect_uri: str | None = None,
    use_pkce: bool = False,
) -> SlackOAuthInstallUrl:
    if not settings.slack_client_id:
        raise SlackOAuthConfigurationError('SLACK_CLIENT_ID is required for Slack OAuth install URL')

    target_redirect_uri = redirect_uri or settings.slack_oauth_redirect_uri

    # 로컬 개발 환경(localhost/127.0.0.1) 감지
    is_local_dev = '://localhost' in target_redirect_uri or '://127.0.0.1' in target_redirect_uri

    # 로컬 개발 환경에서는 Slack OAuth가 non-web URI + bot scope 충돌로 불가
    # .env의 SLACK_BOT_TOKEN을 직접 사용하여 연결을 등록하는 방식으로 대체
    if is_local_dev and settings.slack_bot_token:
        return SlackOAuthInstallUrl(
            connector_type='slack',
            # 프론트엔드에서 이 URL을 감지하여 direct-connect API를 호출하도록 함
            install_url='__direct_connect__',
            state='',
            required_scopes=list(SLACK_OAUTH_BOT_SCOPES),
            configured=True,
            code_verifier=None,
        )

    code_verifier = None

    state = SlackOAuthStateSigner(settings.slack_oauth_state_secret).create(
        nonce=nonce,
        code_verifier=code_verifier,
    )

    params = {
        'client_id': settings.slack_client_id,
        'scope': ','.join(SLACK_OAUTH_BOT_SCOPES),
        'user_scope': ','.join(['im:history', 'mpim:history', 'users:read']),
        'redirect_uri': target_redirect_uri,
        'state': state,
    }

    # bot_token에서 team ID를 추출하여 워크스페이스 자동 선택 (xoxb-TEAMID-...)
    team_id = _extract_team_id_from_bot_token(settings.slack_bot_token)
    if team_id:
        params['team'] = team_id

    query = urlencode(params)
    return SlackOAuthInstallUrl(
        connector_type='slack',
        install_url=f'https://slack.com/oauth/v2/authorize?{query}',
        state=state,
        required_scopes=list(SLACK_OAUTH_BOT_SCOPES),
        configured=True,
        code_verifier=code_verifier,
    )


def complete_slack_direct_connect(
    *,
    db: Session,
    settings: Settings,
    token_vault: LocalTokenVault = LOCAL_TOKEN_VAULT,
) -> IntegrationConnection:
    """.env에 설정된 SLACK_BOT_TOKEN을 사용하여 직접 연결을 생성합니다.
    로컬 개발 시 OAuth 제약을 우회하기 위해 사용됩니다.
    """
    if not settings.slack_bot_token:
        raise SlackOAuthConfigurationError('SLACK_BOT_TOKEN is not configured in .env')

    from backend.app.connectors.slack import SlackWebApiClient
    client = SlackWebApiClient(bot_token=settings.slack_bot_token)
    
    try:
        test_info = client.auth_test()
    except SlackApiError as exc:
        raise SlackApiError(f"Failed to verify SLACK_BOT_TOKEN: {str(exc)}") from exc

    access = SlackOAuthAccess(
        bot_token=settings.slack_bot_token,
        bot_user_id=test_info.get('user_id'),
        team_id=test_info.get('team_id', ''),
        team_name=test_info.get('team', 'Slack workspace'),
        scopes=list(SLACK_REQUIRED_SCOPES),  # .env 토큰은 이미 필요한 권한이 있다고 가정
        user_token=settings.slack_user_token,
        user_id=None, # user_token이 있어도 auth.test만으로는 user_id를 알 수 없으나 bot token 중심임
    )

    # dummy state 생성 (validate를 통과하기 위해)
    dummy_state = SlackOAuthStateSigner(settings.slack_oauth_state_secret).create(
        nonce='direct_connect_nonce',
        connector_type='slack'
    )

    return complete_slack_oauth_callback(
        db=db,
        settings=settings,
        code='direct_connect_code',
        state=dummy_state,
        access=access,
        token_vault=token_vault
    )


def complete_slack_oauth_callback(
    *,
    db: Session,
    settings: Settings,
    code: str,
    state: str,
    access: SlackOAuthAccess | None = None,
    token_vault: LocalTokenVault = LOCAL_TOKEN_VAULT,
    redirect_uri: str | None = None,
) -> IntegrationConnection:
    parsed_state = SlackOAuthStateSigner(settings.slack_oauth_state_secret).validate(state)
    if parsed_state.connector_type != 'slack':
        raise SlackApiError('Slack OAuth state connector mismatch')

    access_payload = access or _exchange_slack_code(
        settings=settings,
        code=code,
        code_verifier=parsed_state.code_verifier,
        redirect_uri=redirect_uri,
    )
    
    # Bot Token 저장
    token_ref = token_vault.store_bot_token(
        connector_type='slack',
        workspace_id=access_payload.team_id,
        token=access_payload.bot_token,
    )
    
    # User Token 저장 (있을 경우)
    if access_payload.user_token:
        token_vault.store_user_token(
            connector_type='slack',
            workspace_id=access_payload.team_id,
            token=access_payload.user_token,
        )

    connection = (
        db.query(IntegrationConnection)
        .filter(
            IntegrationConnection.connector_type == 'slack',
            IntegrationConnection.workspace_id == access_payload.team_id,
        )
        .one_or_none()
    )
    if connection is None:
        connection = IntegrationConnection(
            connector_type='slack',
            workspace_id=access_payload.team_id,
            workspace_name=access_payload.team_name,
            token_ref=token_ref,
            masked_bot_token=mask_secret(access_payload.bot_token),
        )
        db.add(connection)

    connection.workspace_name = access_payload.team_name
    connection.workspace_url = settings.slack_workspace_url
    connection.bot_user_id = access_payload.bot_user_id
    connection.scopes = access_payload.scopes
    connection.token_ref = token_ref
    connection.masked_bot_token = mask_secret(access_payload.bot_token)
    connection.status = 'connected'
    connection.raw_metadata = {
        'state_nonce': parsed_state.nonce,
        'scope_count': len(access_payload.scopes),
        'pkce_used': parsed_state.code_verifier is not None,
        'has_user_token': access_payload.user_token is not None,
        'slack_user_id': access_payload.user_id,
    }

    db.commit()
    db.refresh(connection)
    return connection


def mask_secret(secret: str) -> str:
    if not secret:
        return '****'
    if len(secret) <= 8:
        return '****'
    return f'{secret[:4]}...{secret[-4:]}'


def _exchange_slack_code(
    *,
    settings: Settings,
    code: str,
    code_verifier: str | None = None,
    redirect_uri: str | None = None,
) -> SlackOAuthAccess:
    if not settings.slack_client_id or not settings.slack_client_secret:
        raise SlackOAuthConfigurationError('Slack OAuth client id and secret are required for callback exchange')
    return SlackOAuthClient(
        client_id=settings.slack_client_id,
        client_secret=settings.slack_client_secret,
    ).exchange_code(
        code=code,
        redirect_uri=redirect_uri or settings.slack_oauth_redirect_uri,
        code_verifier=code_verifier,
    )


def _parse_scope_string(scope: str) -> list[str]:
    return [item.strip() for item in scope.split(',') if item.strip()]


def _sign(payload_token: str, secret: bytes) -> str:
    digest = hmac.new(secret, payload_token.encode(), hashlib.sha256).digest()
    return _b64encode(digest)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip('=')


def _b64decode(value: str) -> bytes:
    padded = value + ('=' * (-len(value) % 4))
    return base64.urlsafe_b64decode(padded.encode())


def _extract_team_id_from_bot_token(bot_token: str | None) -> str | None:
    """Slack bot token(xoxb-TEAMID-...)에서 team ID를 추출합니다.

    team 파라미터를 OAuth URL에 추가하면 워크스페이스가 자동 선택됩니다.
    토큰이 없거나 형식이 맞지 않으면 None을 반환합니다.
    """
    if not bot_token or not bot_token.startswith('xoxb-'):
        return None
    parts = bot_token.split('-')
    if len(parts) < 3:
        return None
    return parts[1]
