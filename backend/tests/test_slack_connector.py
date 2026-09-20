from datetime import UTC, datetime

import httpx
import pytest

from backend.app.connectors.slack import (
    SLACK_REQUIRED_SCOPES,
    SlackApiError,
    SlackConnector,
    SlackConnectorConfig,
    SlackWebApiClient,
)


class FakeSlackClient:
    def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
        assert channel_id == 'C123'
        return [
            {
                'type': 'message',
                'user': 'U123',
                'text': 'Redis 결정을 검토 큐와 연결해주세요.',
                'ts': '1777600800.000100',
            }
        ]

    def conversation_replies(
        self,
        channel_id: str,
        thread_ts: str,
        *,
        oldest: str | None = None,
    ) -> list[dict]:
        return []


def test_slack_required_scopes_cover_channel_types() -> None:
    assert SLACK_REQUIRED_SCOPES == (
        'channels:history',
        'groups:history',
        'im:history',
        'mpim:history',
        'channels:read',
        'groups:read',
        'im:read',
        'mpim:read',
        'users:read',
    )


def test_slack_connector_maps_history_messages_to_source_events() -> None:
    connector = SlackConnector(
        config=SlackConnectorConfig(
            bot_token='xoxb-test',
            channel_ids=['C123'],
            workspace_url='https://example.slack.com',
        ),
        client=FakeSlackClient(),
    )

    events = connector.fetch_events()

    assert len(events) == 1
    event = events[0]
    assert event.source_type == 'slack'
    assert event.source_id == 'C123:1777600800.000100'
    assert event.source_url == 'https://example.slack.com/archives/C123/p1777600800000100'
    assert event.title == 'Slack message in C123'
    assert event.body == 'Redis 결정을 검토 큐와 연결해주세요.'
    assert event.author == 'U123'
    assert event.participants == ['U123']
    assert event.timestamp == datetime.fromtimestamp(1777600800.000100, tz=UTC)
    # This minimal fake supplies no channel visibility metadata.
    assert event.permission_level == 'restricted'
    assert event.raw_metadata['required_scopes'] == list(SLACK_REQUIRED_SCOPES)


def test_slack_connector_fetches_incremental_history_after_channel_cursor() -> None:
    observed_oldest: list[str | None] = []
    recent_cursor = str(datetime.now(UTC).timestamp() - 60)

    class IncrementalFakeSlackClient:
        def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
            assert channel_id == 'C123'
            observed_oldest.append(oldest)
            return [
                {
                    'type': 'message',
                    'user': 'U456',
                    'text': 'newer decision',
                    'ts': '1777600900.000100',
                }
            ]

        def conversation_replies(
            self,
            channel_id: str,
            thread_ts: str,
            *,
            oldest: str | None = None,
        ) -> list[dict]:
            return []

    connector = SlackConnector(
        config=SlackConnectorConfig(
            bot_token='xoxb-test',
            channel_ids=['C123'],
            workspace_url='https://example.slack.com',
        ),
        client=IncrementalFakeSlackClient(),
    )

    events = connector.fetch_events_since({'C123': recent_cursor})

    assert observed_oldest == [recent_cursor]
    assert [event.source_id for event in events] == ['C123:1777600900.000100']


def test_slack_connector_collects_thread_replies_with_parent_context() -> None:
    observed_replies: list[tuple[str, str, str | None]] = []
    recent_cursor = str(datetime.now(UTC).timestamp() - 60)

    class ThreadFakeSlackClient:
        def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
            assert channel_id == 'C123'
            assert oldest == recent_cursor
            return [
                {
                    'type': 'message',
                    'user': 'U123',
                    'text': '결정: 벡터 DB는 pgvector로 갑니다.',
                    'ts': '1777600800.000100',
                    'thread_ts': '1777600800.000100',
                    'reply_count': 2,
                }
            ]

        def conversation_replies(
            self,
            channel_id: str,
            thread_ts: str,
            *,
            oldest: str | None = None,
        ) -> list[dict]:
            observed_replies.append((channel_id, thread_ts, oldest))
            return [
                {
                    'type': 'message',
                    'user': 'U123',
                    'text': '결정: 벡터 DB는 pgvector로 갑니다.',
                    'ts': '1777600800.000100',
                    'thread_ts': '1777600800.000100',
                },
                {
                    'type': 'message',
                    'user': 'U456',
                    'text': '동의합니다. embedding 비용은 hash skip으로 줄이죠.',
                    'ts': '1777600810.000200',
                    'thread_ts': '1777600800.000100',
                },
            ]

    connector = SlackConnector(
        config=SlackConnectorConfig(
            bot_token='xoxb-test',
            channel_ids=['C123'],
            workspace_url='https://example.slack.com',
        ),
        client=ThreadFakeSlackClient(),
    )

    events = connector.fetch_events_since({'C123': recent_cursor})

    assert observed_replies == [('C123', '1777600800.000100', recent_cursor)]
    assert [event.source_id for event in events] == [
        'C123:1777600800.000100',
        'C123:1777600810.000200',
    ]
    assert events[0].raw_metadata['is_thread_parent'] is True
    assert events[0].raw_metadata['reply_count'] == 2
    assert events[1].title == 'Slack thread reply in C123'
    assert events[1].body == (
        'Thread parent: 결정: 벡터 DB는 pgvector로 갑니다.\n'
        'Thread reply: 동의합니다. embedding 비용은 hash skip으로 줄이죠.'
    )
    assert events[1].raw_metadata['is_thread_reply'] is True
    assert events[1].raw_metadata['thread_ts'] == '1777600800.000100'
    assert events[1].raw_metadata['thread_parent_text'] == '결정: 벡터 DB는 pgvector로 갑니다.'
    assert events[1].raw_metadata['thread_reply_index'] == 1
    assert events[1].raw_metadata['thread_context_window'] == 'parent_plus_reply'


def test_slack_connector_discovers_channels_when_empty() -> None:
    class DiscoveryFakeSlackClient:
        def conversations_list(self) -> list[dict]:
            return [
                {'id': 'C1', 'name': 'chan1', 'is_member': True},
                {'id': 'C2', 'name': 'chan2', 'is_member': False},
                {'id': 'C3', 'name': 'chan3', 'is_member': True},
            ]

        def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
            return [{'type': 'message', 'user': 'U1', 'text': f'msg in {channel_id}', 'ts': '1.0'}]

        def conversation_replies(self, *args, **kwargs) -> list[dict]:
            return []

    connector = SlackConnector(
        config=SlackConnectorConfig(
            bot_token='xoxb-test',
            channel_ids=[],  # Empty channel list
            workspace_url='https://example.slack.com',
        ),
        client=DiscoveryFakeSlackClient(),
    )

    events = connector.fetch_events()

    # Should discover C1 and C3 (is_member=True)
    assert len(events) == 2
    assert events[0].source_id.startswith('C1:')
    assert events[1].source_id.startswith('C3:')


def test_slack_connector_fetches_configured_channel_even_when_membership_list_is_empty() -> None:
    observed_history_channels: list[str] = []

    class ConfiguredChannelFakeSlackClient:
        def conversations_list(self) -> list[dict]:
            return []

        def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
            observed_history_channels.append(channel_id)
            return [{'type': 'message', 'user': 'U1', 'text': 'configured channel work item', 'ts': '1.0'}]

        def conversation_replies(self, *args, **kwargs) -> list[dict]:
            return []

    connector = SlackConnector(
        config=SlackConnectorConfig(
            bot_token='xoxb-test',
            channel_ids=['C123'],
            workspace_url='https://example.slack.com',
        ),
        client=ConfiguredChannelFakeSlackClient(),
    )

    events = connector.fetch_events()

    assert observed_history_channels == ['C123']
    assert [event.source_id for event in events] == ['C123:1.0']


def test_slack_connector_prefers_user_client_for_configured_channel_when_available() -> None:
    observed: list[str] = []

    class BotClient:
        def conversations_list(self) -> list[dict]:
            return []

        def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
            raise AssertionError('bot client should not fetch configured channel when user client is available')

        def conversation_replies(self, *args, **kwargs) -> list[dict]:
            return []

    class UserClient:
        def conversations_list(self) -> list[dict]:
            return []

        def users_list(self) -> list[dict]:
            return []

        def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
            observed.append(f'user:{channel_id}')
            return [{'type': 'message', 'user': 'U1', 'text': 'user token accessible work item', 'ts': '2.0'}]

        def conversation_replies(self, *args, **kwargs) -> list[dict]:
            return []

    connector = SlackConnector(
        config=SlackConnectorConfig(
            bot_token='xoxb-test',
            channel_ids=['C123'],
            workspace_url='https://example.slack.com',
        ),
        client=BotClient(),
        user_client=UserClient(),
    )

    events = connector.fetch_events()

    assert observed == ['user:C123']
    assert [event.source_id for event in events] == ['C123:2.0']


def test_slack_web_api_client_fetches_paginated_history_with_bearer_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers['authorization'] == 'Bearer xoxb-test'
        cursor = request.url.params.get('cursor')
        if cursor:
            return httpx.Response(
                200,
                json={
                    'ok': True,
                    'messages': [
                        {'type': 'message', 'user': 'U2', 'text': 'second page', 'ts': '2.000100'}
                    ],
                    'response_metadata': {},
                },
            )
        return httpx.Response(
            200,
            json={
                'ok': True,
                'messages': [
                    {'type': 'message', 'user': 'U1', 'text': 'first page', 'ts': '1.000100'}
                ],
                'response_metadata': {'next_cursor': 'cursor-2'},
            },
        )

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    messages = client.conversation_history('C123')

    assert [message['text'] for message in messages] == ['first page', 'second page']
    assert requests[0].url.path == '/api/conversations.history'
    assert requests[0].url.params['channel'] == 'C123'
    assert requests[0].url.params['limit'] == '200'
    assert requests[1].url.params['cursor'] == 'cursor-2'


def test_slack_web_api_client_sends_oldest_for_incremental_history() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'ok': True,
                'messages': [{'type': 'message', 'user': 'U1', 'text': 'new page', 'ts': '2.000100'}],
                'response_metadata': {},
            },
        )

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.conversation_history('C123', oldest='1.000100')

    assert requests[0].url.params['oldest'] == '1.000100'


def test_slack_web_api_client_fetches_paginated_replies_with_oldest() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        cursor = request.url.params.get('cursor')
        if cursor:
            return httpx.Response(
                200,
                json={
                    'ok': True,
                    'messages': [
                        {'type': 'message', 'user': 'U2', 'text': 'reply second page', 'ts': '2.000200'}
                    ],
                    'response_metadata': {},
                },
            )
        return httpx.Response(
            200,
            json={
                'ok': True,
                'messages': [{'type': 'message', 'user': 'U1', 'text': 'reply first page', 'ts': '1.000100'}],
                'response_metadata': {'next_cursor': 'cursor-2'},
            },
        )

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    replies = client.conversation_replies('C123', '1.000100', oldest='0.000100')

    assert [reply['text'] for reply in replies] == ['reply first page', 'reply second page']
    assert requests[0].url.path == '/api/conversations.replies'
    assert requests[0].url.params['channel'] == 'C123'
    assert requests[0].url.params['ts'] == '1.000100'
    assert requests[0].url.params['oldest'] == '0.000100'
    assert requests[1].url.params['cursor'] == 'cursor-2'


def test_slack_web_api_client_fetches_paginated_channel_list() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        cursor = request.url.params.get('cursor')
        if cursor:
            return httpx.Response(
                200,
                json={
                    'ok': True,
                    'channels': [{'id': 'C456', 'name': 'dev', 'is_member': True}],
                    'response_metadata': {},
                },
            )
        return httpx.Response(
            200,
            json={
                'ok': True,
                'channels': [{'id': 'C123', 'name': 'general', 'is_member': True}],
                'response_metadata': {'next_cursor': 'cursor-2'},
            },
        )

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    channels = client.conversations_list()

    assert [channel['id'] for channel in channels] == ['C123', 'C456']
    assert requests[0].url.path == '/api/conversations.list'
    assert requests[0].url.params['types'] == 'public_channel,private_channel,im,mpim'
    assert requests[0].url.params['exclude_archived'] == 'true'


def test_slack_web_api_client_retries_rate_limited_history_with_retry_after() -> None:
    requests: list[httpx.Request] = []
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={'Retry-After': '2'})
        return httpx.Response(
            200,
            json={
                'ok': True,
                'messages': [{'type': 'message', 'user': 'U1', 'text': 'after retry', 'ts': '2.000100'}],
                'response_metadata': {},
            },
        )

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep_calls.append,
    )

    messages = client.conversation_history('C123')

    assert [message['text'] for message in messages] == ['after retry']
    assert len(requests) == 2
    assert sleep_calls == [2.0]


def test_slack_web_api_client_stops_retrying_rate_limited_history_after_limit() -> None:
    requests: list[httpx.Request] = []
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers={'Retry-After': '1'})

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
        sleep=sleep_calls.append,
    )

    with pytest.raises(SlackApiError, match='rate_limited'):
        client.conversation_history('C123')

    assert len(requests) == 2
    assert sleep_calls == [1.0]


def test_slack_web_api_client_retries_transient_server_history_errors() -> None:
    requests: list[httpx.Request] = []
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                'ok': True,
                'messages': [{'type': 'message', 'user': 'U1', 'text': 'server recovered', 'ts': '3.000100'}],
                'response_metadata': {},
            },
        )

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep_calls.append,
    )

    messages = client.conversation_history('C123')

    assert [message['text'] for message in messages] == ['server recovered']
    assert len(requests) == 2
    assert sleep_calls == [1.0]


def test_slack_web_api_client_raises_clear_error_for_slack_api_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'ok': False, 'error': 'missing_scope'})

    client = SlackWebApiClient(
        bot_token='xoxb-test',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(SlackApiError, match='missing_scope'):
        client.conversation_history('C123')
