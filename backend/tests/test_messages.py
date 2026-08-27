import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings


@pytest.fixture(autouse=True)
def enable_message_seed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PARAWORKS_SEED_DEMO_DATA', 'true')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_list_message_channels(client: TestClient) -> None:
    response = client.get('/api/v1/messages/channels')

    assert response.status_code == 200
    data = response.json()
    assert [channel['id'] for channel in data['channels']] == [
        'announcements',
        'project-alpha',
        'review-queue',
    ]
    assert data['channels'][0]['name'] == '전체공지'


def test_list_channel_messages(client: TestClient) -> None:
    response = client.get('/api/v1/messages/channels/project-alpha/messages')

    assert response.status_code == 200
    data = response.json()
    assert data['channel']['id'] == 'project-alpha'
    assert len(data['messages']) >= 2
    assert data['messages'][0]['author_name'] == '김민준'


def test_post_channel_message_appends_to_channel(client: TestClient) -> None:
    response = client.post(
        '/api/v1/messages/channels/project-alpha/messages',
        json={'body': 'Redis 작업 상태 공유 감사합니다. 오늘 오후 검토하겠습니다.'},
    )

    assert response.status_code == 200
    created = response.json()
    assert created['body'] == 'Redis 작업 상태 공유 감사합니다. 오늘 오후 검토하겠습니다.'
    assert created['author_name'] == '관리자'

    messages_response = client.get('/api/v1/messages/channels/project-alpha/messages')
    messages = messages_response.json()['messages']
    assert messages[-1]['id'] == created['id']


def test_post_channel_message_persists_in_database(
    client: TestClient,
    db_session: Session,
) -> None:
    response = client.post(
        '/api/v1/messages/channels/review-queue/messages',
        json={'body': 'DB에 남는 메시지인지 확인합니다.'},
    )

    assert response.status_code == 200
    created = response.json()

    stored = db_session.execute(
        text('select body, channel_id from messages where id = :id'),
        {'id': created['id']},
    ).one()
    assert stored.body == 'DB에 남는 메시지인지 확인합니다.'
    assert stored.channel_id == 'review-queue'


def test_unknown_channel_returns_404(client: TestClient) -> None:
    response = client.get('/api/v1/messages/channels/missing/messages')

    assert response.status_code == 404
    assert response.json()['detail'] == 'message channel not found'


def test_send_message_to_review_creates_source_backed_review_item(
    client: TestClient,
) -> None:
    messages_response = client.get('/api/v1/messages/channels/project-alpha/messages')
    message = messages_response.json()['messages'][0]

    response = client.post(f"/api/v1/messages/messages/{message['id']}/send-to-review")

    assert response.status_code == 200
    review_item = response.json()
    assert review_item['item_type'] == 'message_review'
    assert review_item['status'] == 'pending_review'
    assert review_item['payload']['title'] == '메신저 검토 요청'
    assert review_item['source_links'] == [f"paraworks://messages/{message['id']}"]
    assert review_item['source_snippets'] == [message['body']]


def test_send_unknown_message_to_review_returns_404(client: TestClient) -> None:
    response = client.post('/api/v1/messages/messages/missing/send-to-review')

    assert response.status_code == 404
    assert response.json()['detail'] == 'message not found'


def test_message_channels_stay_empty_without_smoke_demo_seed(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    monkeypatch.setenv('PARAWORKS_SEED_DEMO_DATA', 'false')
    get_settings.cache_clear()

    response = client.get('/api/v1/messages/channels')

    assert response.status_code == 200
    assert response.json()['channels'] == []
