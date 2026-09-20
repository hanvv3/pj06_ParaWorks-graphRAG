"""Invented local conversation; not recovered company history or source authority."""
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

from backend.app.connectors.slack import SlackConnector, SlackConnectorConfig


class SyntheticSlackConnector(SlackConnector):
    """Fixture provenance label, never an authority/approval bypass."""

    @property
    def manifest(self):
        return replace(super().manifest, mode='mock')

    def _message_to_source_event(self, *args, **kwargs):
        event = super()._message_to_source_event(*args, **kwargs)
        return replace(event, raw_metadata={**event.raw_metadata,
                       'synthetic': True, 'fixture_origin': 'slack-s1-invented-conversation-v1'})


class SyntheticSlackClient:
    def __init__(self):
        now = int(datetime.now(UTC).timestamp())
        self.parent_ts = f'{now - 20 * 86400}.000001'
        self.cursor_ts = f'{now - 10}.000002'
        self.reply_ts = f'{now - 30}.000003'  # behind another thread's cursor
        self.histories = {
            'CPUBLIC': [self.message(self.parent_ts, '결정: pgvector를 사용합니다.', 'UPARENT'),
                        self.message(self.cursor_ts, '다른 작업을 확인했습니다.', 'UOTHER')],
            'CPRIVATE': [self.message(self.cursor_ts, '비공개 예산을 검토합니다.', 'UPRIVATE')],
        }
        self.replies = {}
        self.calls = []
        self.replay = False

    @staticmethod
    def message(ts, text, user):
        return {'type': 'message', 'ts': ts, 'text': text, 'user': user, 'team': 'TSYNTHETIC'}

    def users_list(self):
        return [{'id': 'UPARENT', 'real_name': '가상 기획자'},
                {'id': 'UREPLY', 'real_name': '가상 개발자'}]

    def conversations_list(self):
        return [{'id': 'CPUBLIC', 'name': 'synthetic-public', 'is_member': True},
                {'id': 'CPRIVATE', 'name': 'synthetic-private', 'is_member': True, 'is_private': True}]

    def conversation_history(self, channel_id, *, oldest=None):
        self.calls.append(('history', channel_id, oldest))
        # Initial explicit snapshot includes an old known root. Later history behaves
        # like upstream: an old root is NOT surfaced merely because it got a reply.
        return deepcopy(self.histories[channel_id])

    def conversation_replies(self, channel_id, thread_ts, *, oldest=None):
        self.calls.append(('replies', channel_id, thread_ts, oldest))
        values = self.replies.get((channel_id, thread_ts), [])
        return deepcopy([m for m in values if self.replay or not oldest
                         or Decimal(m['ts']) > Decimal(oldest)])

    def add_late_reply(self):
        self.histories = {'CPUBLIC': [], 'CPRIVATE': []}
        self.replies[('CPUBLIC', self.parent_ts)] = [
            dict(self.message(self.reply_ts, '할 일: <@UPARENT>와 색인을 검증합니다.', 'UREPLY'),
                 thread_ts=self.parent_ts)
        ]

    def connector(self):
        return SyntheticSlackConnector(SlackConnectorConfig(
            bot_token='fixture-only', channel_ids=['CPUBLIC', 'CPRIVATE'],
            workspace_url='https://synthetic.invalid'), self)
