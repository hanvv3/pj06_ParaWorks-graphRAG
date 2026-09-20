from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import backend.app.api.v1.integrations as integrations_api
from backend.app.connectors.base import ConnectorManifest
from backend.app.connectors.slack import SlackApiError
from backend.app.connectors.slack_oauth import (
    LOCAL_TOKEN_VAULT,
    LocalTokenVault,
    SlackOAuthAccess,
    SlackOAuthClient,
    SlackOAuthStateSigner,
    build_slack_oauth_install_url,
    complete_slack_oauth_callback,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models import IntegrationConnection


class FailingSlackConnector:
    source_type = 'slack'
    manifest = ConnectorManifest(
        connector_type='slack',
        display_name='Slack',
        mode='live',
        auth_type='oauth',
        required_scopes=('channels:history',),
        sync_strategy='incremental',
        cost_policy='Fetch source deltas first.',
    )

    def fetch_events(self):
        raise SlackApiError('Slack conversations.history failed: channel_not_found')


def test_slack_oauth_install_url_contains_signed_state_and_hides_secret() -> None:
    settings = Settings(
        slack_client_id='C123',
        slack_client_secret='super-secret',
        slack_oauth_redirect_uri='http://localhost:3000/integrations/slack/callback',
        slack_oauth_state_secret='state-secret',
    )

    install = build_slack_oauth_install_url(settings=settings, nonce='nonce-1')
    parsed = urlparse(install.install_url)
    params = parse_qs(parsed.query)

    assert parsed.netloc == 'slack.com'
    assert parsed.path == '/oauth/v2/authorize'
    assert params['client_id'] == ['C123']
    assert params['redirect_uri'] == ['http://localhost:3000/integrations/slack/callback']
    assert 'channels:history' in params['scope'][0]
    assert 'groups:history' in params['scope'][0]
    assert 'super-secret' not in install.install_url
    assert install.state == params['state'][0]

    state = SlackOAuthStateSigner('state-secret').validate(install.state)
    assert state.connector_type == 'slack'
    assert state.nonce == 'nonce-1'


def test_slack_oauth_client_exchanges_code_for_access_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'ok': True,
                'access_token': 'xoxb-secret-token',
                'bot_user_id': 'U999',
                'scope': 'channels:history,groups:history',
                'team': {'id': 'T123', 'name': 'ParaWorks'},
            },
        )

    client = SlackOAuthClient(
        client_id='C123',
        client_secret='client-secret',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    access = client.exchange_code(
        code='temporary-code',
        redirect_uri='http://localhost:3000/integrations/slack/callback',
    )

    assert access.bot_token == 'xoxb-secret-token'
    assert access.team_id == 'T123'
    assert access.team_name == 'ParaWorks'
    body = requests[0].content.decode()
    assert 'client_id=C123' in body
    assert 'client_secret=client-secret' in body
    assert 'code=temporary-code' in body


def test_complete_slack_oauth_callback_persists_connection_without_raw_token(
    db_session: Session,
) -> None:
    settings = Settings(
        slack_client_id='C123',
        slack_client_secret='client-secret',
        slack_oauth_redirect_uri='http://localhost:3000/integrations/slack/callback',
        slack_oauth_state_secret='state-secret',
    )
    state = SlackOAuthStateSigner('state-secret').create(nonce='nonce-1')
    vault = LocalTokenVault()
    access = SlackOAuthAccess(
        bot_token='xoxb-secret-token',
        bot_user_id='U999',
        team_id='T123',
        team_name='ParaWorks',
        scopes=['channels:history', 'groups:history'],
    )

    connection = complete_slack_oauth_callback(
        db=db_session,
        settings=settings,
        code='temporary-code',
        state=state,
        access=access,
        token_vault=vault,
    )

    stored = db_session.query(IntegrationConnection).one()
    assert connection.id == stored.id
    assert stored.connector_type == 'slack'
    assert stored.workspace_id == 'T123'
    assert stored.workspace_name == 'ParaWorks'
    assert stored.status == 'connected'
    assert stored.token_ref == 'local:slack:T123:bot'
    assert stored.masked_bot_token == 'xoxb...oken'
    assert vault.resolve('local:slack:T123:bot') == 'xoxb-secret-token'
    assert 'xoxb-secret-token' not in str(stored.raw_metadata)


def test_slack_oauth_install_url_api_uses_settings_without_exposing_secret(
    client: TestClient,
) -> None:
    def override_settings() -> Settings:
        return Settings(
            slack_client_id='C123',
            slack_client_secret='client-secret',
            slack_oauth_redirect_uri='http://localhost:3000/integrations/slack/callback',
            slack_oauth_state_secret='state-secret',
        )

    client.app.dependency_overrides[get_settings] = override_settings

    response = client.get('/api/v1/integrations/slack/oauth/install-url')

    assert response.status_code == 200
    payload = response.json()
    assert payload['configured'] is True
    assert payload['connector_type'] == 'slack'
    assert 'channels:history' in payload['required_scopes']
    assert 'client-secret' not in str(payload)


def test_slack_oauth_callback_api_rejects_invalid_state(client: TestClient) -> None:
    def override_settings() -> Settings:
        return Settings(
            slack_client_id='C123',
            slack_client_secret='client-secret',
            slack_oauth_redirect_uri='http://localhost:3000/integrations/slack/callback',
            slack_oauth_state_secret='state-secret',
        )

    client.app.dependency_overrides[get_settings] = override_settings

    response = client.get(
        '/api/v1/integrations/slack/oauth/callback',
        params={'code': 'temporary-code', 'state': 'not-a-signed-state'},
    )

    assert response.status_code == 400
    assert response.json()['detail'] == 'Slack OAuth state is malformed'


def test_integration_connections_api_hides_token_references(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        IntegrationConnection(
            connector_type='slack',
            workspace_id='T999',
            workspace_name='ParaWorks',
            bot_user_id='U999',
            scopes=['channels:history'],
            token_ref='local:slack:T999:bot',
            masked_bot_token='xoxb...oken',
            status='connected',
            raw_metadata={'safe': True},
        )
    )
    db_session.commit()

    response = client.get('/api/v1/integrations/connections')

    assert response.status_code == 200
    payload = response.json()
    assert payload == [
        {
            'connector_type': 'slack',
            'workspace_id': 'T999',
            'workspace_name': 'ParaWorks',
            'status': 'connected',
            'credential_status': 'missing',
            'masked_bot_token': 'xoxb...oken',
            'scopes': ['channels:history'],
        }
    ]
    assert 'token_ref' not in str(payload)
    assert 'local:slack:T999:bot' not in str(payload)


def test_integration_connections_api_marks_resolvable_vault_token_available(
    client: TestClient,
    db_session: Session,
) -> None:
    token_ref = LOCAL_TOKEN_VAULT.store_bot_token(
        connector_type='slack',
        workspace_id='T777',
        token='xoxb-available',
    )
    db_session.add(
        IntegrationConnection(
            connector_type='slack',
            workspace_id='T777',
            workspace_name='ParaWorks',
            bot_user_id='U777',
            scopes=['channels:history'],
            token_ref=token_ref,
            masked_bot_token='xoxb...able',
            status='connected',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/integrations/connections')

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]['credential_status'] == 'available'
    assert 'xoxb-available' not in str(payload)
    assert token_ref not in str(payload)


def test_slack_sync_endpoint_uses_installed_connection_token_without_exposing_it(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    token_ref = LOCAL_TOKEN_VAULT.store_bot_token(
        connector_type='slack',
        workspace_id='T123',
        token='xoxb-installed',
    )
    db_session.add(
        IntegrationConnection(
            connector_type='slack',
            workspace_id='T123',
            workspace_name='ParaWorks',
            bot_user_id='U999',
            scopes=['channels:history'],
            token_ref=token_ref,
            masked_bot_token='xoxb...lled',
            status='connected',
        )
    )
    db_session.commit()

    def override_settings() -> Settings:
        return Settings(
            paraworks_demo_mode=False,
            slack_bot_token=None,
            slack_channel_ids='C123',
        )

    captured: dict[str, object] = {}

    def fake_sync_connector_events(*, db: Session, connector, job_id=None):
        assert job_id is None  # synchronous route has no preallocated background job
        captured['bot_token'] = connector.config.bot_token
        captured['channel_ids'] = connector.config.channel_ids
        return SimpleNamespace(
            job_id='sync-test',
            status='complete',
            created_review_items=0,
            fetched_events=0,
            skipped_events=0,
        )

    client.app.dependency_overrides[get_settings] = override_settings
    monkeypatch.setattr(integrations_api, 'sync_connector_events', fake_sync_connector_events)

    response = client.post(
        '/api/v1/integrations/slack/sync',
        json={'selected_channel_ids': ['C456']},
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured == {'bot_token': 'xoxb-installed', 'channel_ids': ['C456']}
    assert 'xoxb-installed' not in str(payload)
    assert 'token_ref' not in str(payload)


def test_slack_sync_endpoint_returns_clear_error_for_slack_api_failure(
    client: TestClient,
    monkeypatch,
) -> None:
    def fake_get_sync_connector(
        connector_type: str,
        settings: Settings,
        *,
        db: Session | None = None,
        slack_channel_ids_override: list[str] | None = None,
    ):
        return FailingSlackConnector()

    monkeypatch.setattr(integrations_api, 'get_sync_connector', fake_get_sync_connector)

    response = client.post('/api/v1/integrations/slack/sync')

    assert response.status_code == 502
    assert response.json()['detail'] == 'Slack conversations.history failed: channel_not_found'
