from urllib.parse import parse_qs, urlparse

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.auth.google_identity import (
    build_google_identity_login_url,
)
from backend.app.connectors.google_oauth import (
    build_google_oauth_install_url,
    complete_google_oauth_callback,
)
from backend.app.connectors.slack_oauth import (
    SlackOAuthStateSigner,
    build_slack_oauth_install_url,
    complete_slack_oauth_callback,
    pkce_challenge,
)
from backend.app.core.config import Settings, get_settings


def test_slack_oauth_pkce_generation() -> None:
    settings = Settings(
        slack_client_id='C123',
        slack_oauth_state_secret='state-secret',
    )
    
    # PKCE 사용 설정 (기본값)
    install = build_slack_oauth_install_url(settings=settings)
    parsed = urlparse(install.install_url)
    params = parse_qs(parsed.query)
    
    assert 'code_challenge' in params
    assert params['code_challenge_method'] == ['S256']
    assert install.code_verifier is not None
    
    # code_challenge 검증
    expected_challenge = pkce_challenge(install.code_verifier)
    assert params['code_challenge'] == [expected_challenge]
    
    # state 내에 code_verifier가 포함되어 있는지 확인
    state = SlackOAuthStateSigner('state-secret').validate(install.state)
    assert state.code_verifier == install.code_verifier

def test_slack_oauth_custom_redirect_uri() -> None:
    settings = Settings(
        slack_client_id='C123',
        slack_oauth_redirect_uri='http://localhost:3000/callback',
        slack_oauth_state_secret='state-secret',
    )
    
    custom_uri = 'paraworks://oauth-callback'
    install = build_slack_oauth_install_url(settings=settings, redirect_uri=custom_uri)
    parsed = urlparse(install.install_url)
    params = parse_qs(parsed.query)
    
    assert params['redirect_uri'] == [custom_uri]

def test_google_oauth_pkce_generation() -> None:
    settings = Settings(
        google_client_id='G123',
        google_oauth_state_secret='state-secret',
    )
    
    install = build_google_oauth_install_url(settings=settings, connector_type='gmail')
    parsed = urlparse(install.install_url)
    params = parse_qs(parsed.query)
    
    assert 'code_challenge' in params
    assert params['code_challenge_method'] == ['S256']
    assert install.code_verifier is not None

def test_google_identity_pkce_generation() -> None:
    settings = Settings(
        google_client_id='G123',
        google_identity_state_secret='state-secret',
    )
    
    login = build_google_identity_login_url(settings=settings)
    parsed = urlparse(login.login_url)
    params = parse_qs(parsed.query)
    
    assert 'code_challenge' in params
    assert params['code_challenge_method'] == ['S256']
    assert login.code_verifier is not None

def test_slack_callback_with_custom_redirect_uri_and_pkce(db_session: Session, monkeypatch) -> None:
    settings = Settings(
        slack_client_id='C123',
        slack_client_secret='S123',
        slack_oauth_state_secret='state-secret',
        slack_oauth_redirect_uri='http://localhost:3000/callback'
    )
    
    # 1. Install URL 생성 (PKCE 포함)
    custom_uri = 'http://localhost:9999/callback'
    install = build_slack_oauth_install_url(settings=settings, redirect_uri=custom_uri)
    
    # 2. Mock Slack API exchange
    captured_data = {}
    def mock_post(self, url, *args, **kwargs):
        data = kwargs.get('data')
        captured_data.update(data or {})
        res = httpx.Response(200, json={
            'ok': True,
            'access_token': 'xoxb-token',
            'bot_user_id': 'U123',
            'team': {'id': 'T123', 'name': 'Team'},
            'scope': 'identify'
        })
        res._request = httpx.Request('POST', url)
        return res
    
    monkeypatch.setattr(httpx.Client, 'post', mock_post)
    
    # 3. Callback 수행
    connection = complete_slack_oauth_callback(
        db=db_session,
        settings=settings,
        code='test-code',
        state=install.state,
        redirect_uri=custom_uri
    )
    
    assert captured_data['code'] == 'test-code'
    assert captured_data['redirect_uri'] == custom_uri
    assert captured_data['code_verifier'] == install.code_verifier
    assert connection.raw_metadata['pkce_used'] is True

def test_google_callback_with_custom_redirect_uri_and_pkce(db_session: Session, monkeypatch) -> None:
    settings = Settings(
        google_client_id='G123',
        google_client_secret='S123',
        google_oauth_state_secret='state-secret',
        google_oauth_redirect_uri='http://localhost:3000/callback'
    )
    
    custom_uri = 'http://localhost:9999/callback'
    install = build_google_oauth_install_url(settings=settings, connector_type='gmail', redirect_uri=custom_uri)
    
    # Mock Google Token API
    def mock_post(self, url, data=None, **kwargs):
        res = httpx.Response(200, json={
            'access_token': 'ya29.token',
            'refresh_token': 'refresh-123',
            'expires_in': 3600,
            'scope': 'openid email'
        })
        res._request = httpx.Request('POST', url)
        return res
    
    # Mock Userinfo API
    def mock_get(self, url, **kwargs):
        res = httpx.Response(200, json={'sub': 'user-123', 'email': 'test@example.com'})
        res._request = httpx.Request('GET', url)
        return res
        
    monkeypatch.setattr(httpx.Client, 'post', mock_post)
    monkeypatch.setattr(httpx.Client, 'get', mock_get)
    
    from backend.app.connectors.slack_oauth import LOCAL_TOKEN_VAULT
    
    connection = complete_google_oauth_callback(
        db=db_session,
        settings=settings,
        connector_type='gmail',
        code='test-code',
        state=install.state,
        token_vault=LOCAL_TOKEN_VAULT,
        redirect_uri=custom_uri
    )
    
    assert connection.raw_metadata['pkce_used'] is True
    assert connection.raw_metadata['token_kind'] == 'refresh_token'

def test_api_endpoints_support_redirect_uri(client: TestClient) -> None:
    def override_settings():
        return Settings(
            slack_client_id='C123',
            slack_oauth_state_secret='state-secret',
            google_client_id='G123',
            google_oauth_state_secret='state-secret'
        )
    client.app.dependency_overrides[get_settings] = override_settings
    
    # Slack Install URL
    custom_uri = 'paraworks://slack'
    response = client.get('/api/v1/integrations/slack/oauth/install-url', params={'redirect_uri': custom_uri})
    assert response.status_code == 200
    
    install_url = response.json()['install_url']
    parsed = urlparse(install_url)
    params = parse_qs(parsed.query)
    assert params['redirect_uri'] == [custom_uri]
    assert 'code_challenge' in params
    
    # Google Install URL
    response = client.get('/api/v1/integrations/gmail/oauth/install-url', params={'redirect_uri': custom_uri})
    assert response.status_code == 200
    
    install_url = response.json()['install_url']
    parsed = urlparse(install_url)
    params = parse_qs(parsed.query)
    assert params['redirect_uri'] == [custom_uri]
    assert 'code_challenge' in params
