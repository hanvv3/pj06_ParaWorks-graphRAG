"""Explicit local-only Slack observations; never selected by an HTTP payload.

The adapter owns an in-memory client and accepts no token or network client.
Source authority is granted by shared sync's adapter gate, not these labels.
"""

from copy import deepcopy
from dataclasses import replace

from backend.app.connectors.registry import get_connector_manifest
from backend.app.connectors.slack import SlackConnector, SlackConnectorConfig


class _LocalSlackClient:
    def __init__(self, channels, messages, replies, users):
        self.channels = deepcopy(channels)
        self.messages = deepcopy(messages)
        self.replies = deepcopy(replies)
        self.users = deepcopy(users)

    def conversations_list(self):
        return deepcopy(self.channels)

    def users_list(self):
        return deepcopy(self.users)

    def conversation_history(self, channel_id, *, oldest=None):
        # An explicit snapshot can deliver edits behind the cursor.
        return deepcopy(self.messages.get(channel_id, []))

    def conversation_replies(self, channel_id, thread_ts, *, oldest=None):
        return deepcopy(self.replies.get((channel_id, thread_ts), []))


class LocalSyntheticSlackConnector(SlackConnector):
    def __init__(
        self,
        *,
        channels,
        messages_by_channel,
        replies_by_thread=None,
        users=(),
        deleted_messages=(),
    ):
        self._deleted_messages = frozenset(deleted_messages)
        messages = deepcopy(messages_by_channel)
        for channel_id, timestamp in self._deleted_messages:
            messages.setdefault(channel_id, []).append(
                {
                    'type': 'message',
                    'ts': timestamp,
                    'text': '[deleted synthetic message]',
                }
            )
        super().__init__(
            SlackConnectorConfig(
                bot_token='local-synthetic-no-network',
                channel_ids=[channel['id'] for channel in channels],
                workspace_url='https://synthetic.invalid',
            ),
            _LocalSlackClient(channels, messages, replies_by_thread or {}, users),
        )

    @property
    def manifest(self):
        return get_connector_manifest('slack', demo_mode=True)

    def _message_to_source_event(self, *args, **kwargs):
        # Connector observations of parent text are not current parent authority.
        # Parent evidence is resolved separately by the Slack evidence service.
        kwargs['parent_text'] = None
        kwargs['parent_user_id'] = None
        event = super()._message_to_source_event(*args, **kwargs)
        return replace(
            event,
            raw_metadata={
                **event.raw_metadata,
                'synthetic': True,
                'fixture_origin': 'local-synthetic-slack:v1',
                'slack_state': (
                    'deleted'
                    if (event.raw_metadata['channel_id'], event.raw_metadata['ts'])
                    in self._deleted_messages
                    else 'active'
                ),
            },
        )
