import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import sleep as default_sleep
from typing import Protocol

import httpx

from backend.app.connectors.base import ConnectorManifest, SourceEvent

SLACK_REQUIRED_SCOPES = (
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
SLACK_MAX_CHANNELS = 50
SLACK_KNOWN_MESSAGE_WINDOW = 50


class SlackApiClient(Protocol):
    def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
        raise NotImplementedError

    def conversation_replies(self, channel_id: str, thread_ts: str, *, oldest: str | None = None) -> list[dict]:
        raise NotImplementedError

    def conversations_list(self) -> list[dict]:
        raise NotImplementedError

    def users_list(self) -> list[dict]:
        raise NotImplementedError


class SlackApiError(RuntimeError):
    pass


class SlackSyncLimitError(SlackApiError):
    """A bounded fetch must abort, including optional metadata/fallback paths."""


class SlackWebApiClient:
    def __init__(
        self,
        *,
        bot_token: str,
        http_client: httpx.Client | None = None,
        base_url: str = 'https://slack.com/api',
        page_limit: int = 200,
        max_retries: int = 2,
        sleep: Callable[[float], None] = default_sleep,
    ) -> None:
        self.bot_token = bot_token
        self.http_client = http_client or httpx.Client(timeout=30.0)
        self.base_url = base_url.rstrip('/')
        self.page_limit = page_limit
        self.max_retries = max_retries
        self.sleep = sleep

    def conversation_history(self, channel_id: str, *, oldest: str | None = None) -> list[dict]:
        params = {
            'channel': channel_id,
            'limit': str(self.page_limit),
        }
        if oldest:
            params['oldest'] = oldest

        return self._get_paginated_items('conversations.history', 'messages', params)

    def conversation_replies(self, channel_id: str, thread_ts: str, *, oldest: str | None = None) -> list[dict]:
        params = {
            'channel': channel_id,
            'ts': thread_ts,
            'limit': str(self.page_limit),
        }
        if oldest:
            params['oldest'] = oldest

        return self._get_paginated_items('conversations.replies', 'messages', params)

    def conversations_list(self) -> list[dict]:
        return self._get_paginated_items(
            'conversations.list',
            'channels',
            {
                'types': 'public_channel,private_channel,im,mpim',
                'exclude_archived': 'true',
                'limit': str(self.page_limit),
            },
        )

    def auth_test(self) -> dict:
        response = self._get_with_retries('auth.test', {})
        payload = response.json()
        if not payload.get('ok'):
            raise SlackApiError(f"Slack auth.test failed: {payload.get('error', 'unknown_error')}")
        return payload

    def users_list(self) -> list[dict]:
        return self._get_paginated_items('users.list', 'members', {'limit': str(self.page_limit)})

    def _get_paginated_items(self, method: str, item_key: str, params: dict[str, str]) -> list[dict]:
        items: list[dict] = []
        cursor: str | None = None

        for _ in range(20):
            page_params = dict(params)
            if cursor:
                page_params['cursor'] = cursor

            response = self._get_with_retries(method, page_params)
            payload = response.json()
            if not payload.get('ok'):
                raise SlackApiError(f"Slack {method} failed: {payload.get('error', 'unknown_error')}")
            items.extend(payload.get(item_key, []))
            cursor = str(payload.get('response_metadata', {}).get('next_cursor') or '')
            if not cursor:
                return items
        raise SlackSyncLimitError(f'Slack {method} failed: page_limit_exceeded')

    def _get_with_retries(self, method: str, params: dict[str, str]) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            response = self.http_client.get(
                f'{self.base_url}/{method}',
                headers={'Authorization': f'Bearer {self.bot_token}'},
                params=params,
            )
            if response.status_code == 429:
                if attempt >= self.max_retries:
                    raise SlackApiError(f'Slack {method} failed: rate_limited')
                self.sleep(_retry_after_seconds(response))
                continue
            if response.status_code >= 500:
                if attempt >= self.max_retries:
                    raise SlackApiError(f'Slack {method} failed: http_{response.status_code}')
                self.sleep(_retry_after_seconds(response))
                continue
            if response.status_code >= 400:
                raise SlackApiError(f'Slack {method} failed: http_{response.status_code}')
            return response
        raise SlackApiError(f'Slack {method} failed: retry_exhausted')


@dataclass(frozen=True)
class SlackConnectorConfig:
    bot_token: str
    channel_ids: list[str]
    workspace_url: str = 'https://slack.com'


@dataclass(frozen=True)
class SlackConnector:
    config: SlackConnectorConfig
    client: SlackApiClient
    user_client: SlackApiClient | None = None
    source_type: str = 'slack'

    @property
    def manifest(self) -> ConnectorManifest:
        return ConnectorManifest(
            connector_type='slack',
            display_name='Slack',
            mode='live',
            auth_type='oauth',
            required_scopes=SLACK_REQUIRED_SCOPES,
            sync_strategy='incremental',
            cost_policy='Fetch source deltas first; embed only changed chunks after review approval.',
        )

    def fetch_events(self) -> list[SourceEvent]:
        return self.fetch_events_since({})

    def _fetch_events_since_bot_membership_filtered(self, latest_timestamps_by_partition: dict[str, str]) -> list[SourceEvent]:
        events: list[SourceEvent] = []

        # 1. 사용자 목록을 가져와 ID -> 실명 매핑 생성 (작성자 이름 및 본문 멘션 치환용)
        user_map: dict[str, str] = {}
        try:
            for member in self.client.users_list():
                uid = member.get('id')
                if not uid:
                    continue
                # 실명이 있으면 실명을, 없으면 표시 이름을 사용
                name = member.get('real_name') or member.get('profile', {}).get('real_name') or member.get('name')
                if name:
                    user_map[uid] = name
        except Exception:
            # 사용자 목록을 가져오지 못해도 동기화는 계속 진행
            pass

        # 2. 채널 목록을 가져와 ID -> 채널명 매핑 생성 (메타데이터 보강용)
        channel_map: dict[str, str] = {}
        try:
            for channel in self.client.conversations_list():
                cid = channel.get('id')
                cname = channel.get('name')
                if cid and cname:
                    channel_map[cid] = cname
        except Exception:
            pass

        # 3. 대상 채널 결정
        configured_channel_ids = self.config.channel_ids

        # 4. 봇이 참여 중인 채널 목록 조회 (필터링용)
        all_channels = self.client.conversations_list()
        joined_channel_ids = {
            c['id'] for c in all_channels 
            if c.get('is_member') or c.get('is_im') or c.get('is_mpim')
        }

        # 5. 설정된 채널 중 봇이 참여 중인 채널만 선별
        if configured_channel_ids:
            # .env에 채널이 설정되어 있다면, 그중 봇이 들어있는 채널만 처리
            target_channel_ids = [cid for idx, cid in enumerate(configured_channel_ids) if cid in joined_channel_ids]
        else:
            # .env에 채널 설정이 없다면, 봇이 들어있는 모든 채널 처리
            target_channel_ids = list(joined_channel_ids)

        for channel_id in target_channel_ids:
            # 7일 전 타임스탬프 계산
            seven_days_ago = (datetime.now(UTC) - timedelta(days=7)).timestamp()

            # DB 기록이 없으면 최근 7일치만, 있으면 기록된 시점 이후만 가져옴
            oldest_val = latest_timestamps_by_partition.get(channel_id)
            oldest = str(max(float(oldest_val), seven_days_ago)) if oldest_val else str(seven_days_ago)

            for message in self.client.conversation_history(channel_id, oldest=oldest):
                if message.get('type') != 'message' or not message.get('text'):
                    continue
                events.append(self._message_to_source_event(channel_id, message, user_map=user_map, channel_map=channel_map))
                thread_ts = str(message.get('thread_ts') or message.get('ts') or '')
                if not thread_ts or int(message.get('reply_count') or 0) <= 0:
                    continue
                reply_index = 0
                parent_text = str(message.get('text') or '')
                for reply in self.client.conversation_replies(channel_id, thread_ts, oldest=oldest):
                    if reply.get('ts') == message.get('ts'):
                        continue
                    if reply.get('type') != 'message' or not reply.get('text'):
                        continue
                    reply_index += 1
                    events.append(
                        self._message_to_source_event(
                            channel_id,
                            reply,
                            parent_ts=thread_ts,
                            parent_text=parent_text,
                            reply_index=reply_index,
                            user_map=user_map,
                            channel_map=channel_map,
                        )
                    )
        return events

    def fetch_events_since(
        self, latest_timestamps_by_partition: dict[str, str], *,
        known_messages_by_channel: dict[str, list[dict]] | None = None,
    ) -> list[SourceEvent]:
        """Optionally revisit bounded persisted observations, without granting trust.

        Unknown old roots absent from upstream history cannot be discovered here.
        Each known thread uses its own cursor, not another thread's newest event.
        """
        events: list[SourceEvent] = []
        available_clients = [self.client]
        if self.user_client is not None:
            available_clients.append(self.user_client)

        user_map: dict[str, str] = {}
        for client in available_clients:
            for member in _safe_users_list(client):
                uid = member.get('id')
                if not uid:
                    continue
                name = member.get('real_name') or member.get('profile', {}).get('real_name') or member.get('name')
                if name:
                    user_map[uid] = name

        channel_map: dict[str, str] = {}
        restricted_channels: set[str] = set()
        public_channels: set[str] = set()
        accessible_channels: dict[str, SlackApiClient] = {}
        for client in available_clients:
            for channel in _safe_conversations_list(client):
                cid = channel.get('id')
                cname = channel.get('name')
                if cid and any(channel.get(flag) for flag in ('is_private', 'is_im', 'is_mpim')):
                    restricted_channels.add(cid)
                if cid and channel.get('is_channel') is True and channel.get('is_private') is False:
                    public_channels.add(cid)
                if cid and cname:
                    channel_map[cid] = cname
                if cid and (channel.get('is_member') or channel.get('is_im') or channel.get('is_mpim')):
                    accessible_channels.setdefault(cid, client)

        configured_channel_ids = list(dict.fromkeys(self.config.channel_ids))
        target_channel_ids = configured_channel_ids if configured_channel_ids else list(accessible_channels)
        if len(target_channel_ids) > SLACK_MAX_CHANNELS:
            raise SlackApiError('Slack sync failed: channel_limit_exceeded')

        for channel_id in target_channel_ids:
            history_client = _client_for_channel(
                channel_id=channel_id,
                bot_client=self.client,
                user_client=self.user_client,
                accessible_channels=accessible_channels,
                configured_channel_ids=configured_channel_ids,
            )
            seven_days_ago = (datetime.now(UTC) - timedelta(days=7)).timestamp()
            oldest_val = latest_timestamps_by_partition.get(channel_id)
            oldest = str(max(float(oldest_val), seven_days_ago)) if oldest_val else str(seven_days_ago)

            messages = _conversation_history_with_fallback(
                primary_client=history_client,
                fallback_client=self.client if history_client is self.user_client else None,
                channel_id=channel_id,
                oldest=oldest,
            )
            permission = (
                'internal' if channel_id in public_channels
                and channel_id not in restricted_channels and not channel_id.startswith('D')
                else 'restricted'
            )
            threads: dict[str, dict] = {}
            for metadata in (known_messages_by_channel or {}).get(channel_id, [])[:SLACK_KNOWN_MESSAGE_WINDOW]:
                thread_ts = metadata.get('thread_ts') or metadata.get('ts')
                ts = metadata.get('ts')
                if not thread_ts or not ts:
                    continue
                previous = threads.get(thread_ts)
                thread_permission = (
                    'restricted' if metadata.get('permission_level') == 'restricted'
                    or (previous and previous['permission'] == 'restricted') else permission
                )
                if previous:
                    previous['permission'] = thread_permission
                # Own text/user describes the root only when this observation IS
                # the root. A history-only thread_broadcast is still a reply.
                parent_text = (metadata.get('raw_text') if ts == thread_ts
                               else metadata.get('thread_parent_text'))
                parent_user = (metadata.get('slack_user_id') if ts == thread_ts
                               else metadata.get('thread_parent_user_id'))
                if previous is None or Decimal(ts) > Decimal(previous['cursor']):
                    threads[thread_ts] = {
                        'cursor': ts,
                        'text': parent_text or (previous['text'] if previous else None),
                        'user': parent_user or (previous['user'] if previous else None),
                        'team': metadata.get('workspace_id'),
                        'permission': thread_permission,
                    }
                elif parent_text and (ts == thread_ts or not previous['text']):
                    previous['text'] = parent_text
                    previous['user'] = parent_user
            for message in messages:
                if message.get('type') != 'message' or not message.get('text'):
                    continue
                events.append(self._message_to_source_event(channel_id, message, user_map=user_map, channel_map=channel_map, permission_level=permission))
                thread_ts = str(message.get('thread_ts') or message.get('ts') or '')
                if not thread_ts or thread_ts != str(message.get('ts')) or int(message.get('reply_count') or 0) <= 0:
                    continue
                previous = threads.get(thread_ts)
                threads[thread_ts] = {
                    'cursor': previous['cursor'] if previous else oldest,
                    'text': str(message.get('text') or ''),
                    'user': message.get('user'), 'team': message.get('team'),
                    'permission': previous['permission'] if previous else permission,
                }
            if len(threads) > SLACK_KNOWN_MESSAGE_WINDOW:
                raise SlackApiError('Slack sync failed: thread_limit_exceeded')
            for thread_ts, context in threads.items():
                reply_index = 0
                replies = _conversation_replies_with_fallback(
                    primary_client=history_client,
                    fallback_client=self.client if history_client is self.user_client else None,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    oldest=context['cursor'],
                )
                for reply in replies:
                    if reply.get('ts') == thread_ts:
                        continue
                    if reply.get('type') != 'message' or not reply.get('text'):
                        continue
                    reply_index += 1
                    events.append(
                        self._message_to_source_event(
                            channel_id,
                            reply,
                            parent_ts=thread_ts,
                            parent_text=context['text'],
                            parent_user_id=context['user'],
                            workspace_id=context['team'],
                            permission_level=('restricted' if 'restricted' in
                                              (permission, context['permission']) else permission),
                            reply_index=reply_index,
                            user_map=user_map,
                            channel_map=channel_map,
                        )
                    )
        # A thread-broadcast may occur in both history and replies. Keep context.
        return list({event.source_id: event for event in events}.values())

    def _message_to_source_event(
        self,
        channel_id: str,
        message: dict,
        *,
        user_map: dict[str, str],
        channel_map: dict[str, str],
        parent_ts: str | None = None,
        parent_text: str | None = None,
        reply_index: int | None = None,
        parent_user_id: str | None = None,
        workspace_id: str | None = None,
        permission_level: str = 'internal',
    ) -> SourceEvent:
        timestamp = str(message['ts'])
        user_id = message.get('user') or message.get('username')
        # 사용자 ID를 실명으로 변환
        author = user_map.get(user_id, user_id) if user_id else None

        thread_ts = str(message.get('thread_ts') or parent_ts or timestamp)
        is_thread_reply = timestamp != thread_ts
        reply_count = int(message.get('reply_count') or 0)

        raw_text = str(message['text'])
        # 본문 내 사용자 멘션(<@U...>)을 실명으로 치환
        body_text = _resolve_mentions(raw_text, user_map)
        resolved_parent_text = _resolve_mentions(parent_text, user_map) if parent_text else None

        body = _thread_context_body(message_text=body_text, parent_text=resolved_parent_text)

        # [RAG 태그 고도화] 정적 메타데이터 보강
        event_dt = datetime.fromtimestamp(float(timestamp), tz=UTC)
        channel_name = channel_map.get(channel_id, channel_id)
        if not channel_name.startswith('#') and not channel_id.startswith('D'):
             channel_name = f"#{channel_name}"

        # 중복 임베딩 방지를 위한 콘텐츠 시그니처 (해시)
        content_signature = hashlib.sha256(body.encode('utf-8')).hexdigest()

        return SourceEvent(
            source_type='slack',
            source_id=f'{channel_id}:{timestamp}',
            source_url=_slack_permalink(self.config.workspace_url, channel_id, timestamp),
            title=f'Slack thread reply in {channel_id}' if is_thread_reply else f'Slack message in {channel_id}',
            body=body,
            author=author,
            participants=list(dict.fromkeys(user_map.get(uid, uid) for uid in
                                             (parent_user_id, user_id) if uid)),
            timestamp=event_dt,
            permission_level=permission_level,
            semantic_timestamp_raw=timestamp,
            raw_metadata={
                'workspace_id': message.get('team') or workspace_id,
                'workspace_url': self.config.workspace_url.rstrip('/'),
                'raw_text': raw_text,
                'source_snippet': raw_text,
                'slack_participant_ids': list(dict.fromkeys(uid for uid in (parent_user_id, user_id) if uid)),
                'thread_parent_user_id': parent_user_id,
                'channel_id': channel_id,
                'channel_name': channel_name, # 보강된 태그
                'author_name': author,        # 보강된 태그
                'ts': timestamp,
                'thread_ts': thread_ts,
                'is_thread_reply': is_thread_reply, # 보강된 태그
                'parent_ts': thread_ts if is_thread_reply else None, # 보강된 태그
                'created_at_date': event_dt.strftime('%Y-%m-%d'), # 보강된 태그
                'content_signature': content_signature, # 중복 방지 태그
                'is_thread_parent': reply_count > 0 and thread_ts == timestamp,
                'reply_count': reply_count,
                'thread_parent_text': resolved_parent_text,
                'thread_parent_missing': is_thread_reply and not parent_text,
                'thread_reply_index': reply_index,
                'thread_context_window': ('parent_plus_reply' if parent_text else
                                          'reply_without_parent' if is_thread_reply else 'single_message'),
                'required_scopes': list(SLACK_REQUIRED_SCOPES),
                'slack_user_id': user_id,
            },
        )


def build_slack_permalink(workspace_url: str, channel_id: str, timestamp: str) -> str:
    """슬랙 메시지 이동을 위한 표준 16자리 p-타임스탬프 URL을 생성합니다."""
    normalized_workspace = workspace_url.rstrip('/')
    # 점(.)을 제거하고 16자리가 되도록 뒤를 0으로 채움 (슬랙 표준)
    permalink_ts = timestamp.replace('.', '').ljust(16, '0')
    return f'{normalized_workspace}/archives/{channel_id}/p{permalink_ts}'


def _slack_permalink(workspace_url: str, channel_id: str, timestamp: str) -> str:
    return build_slack_permalink(workspace_url, channel_id, timestamp)


def _resolve_mentions(text: str, user_map: dict[str, str]) -> str:
    """텍스트 내의 <@U...> 멘션을 실명으로 치환합니다."""
    if not text:
        return text
    import re
    def replace_mention(match):
        uid = match.group(1)
        return f"@{user_map.get(uid, uid)}"

    return re.sub(r'<@([A-Z0-9]+)>', replace_mention, text)


def _thread_context_body(*, message_text: str, parent_text: str | None) -> str:
    if not parent_text:
        return message_text
    return f'Thread parent: {parent_text}\nThread reply: {message_text}'


def _safe_users_list(client: SlackApiClient) -> list[dict]:
    try:
        return client.users_list()
    except SlackSyncLimitError:
        raise
    except Exception:
        return []


def _safe_conversations_list(client: SlackApiClient) -> list[dict]:
    try:
        return client.conversations_list()
    except SlackSyncLimitError:
        raise
    except Exception:
        return []


def _client_for_channel(
    *,
    channel_id: str,
    bot_client: SlackApiClient,
    user_client: SlackApiClient | None,
    accessible_channels: dict[str, SlackApiClient],
    configured_channel_ids: list[str],
) -> SlackApiClient:
    if user_client is not None and channel_id in configured_channel_ids:
        return user_client
    return accessible_channels.get(channel_id, bot_client)


def _conversation_history_with_fallback(
    *,
    primary_client: SlackApiClient,
    fallback_client: SlackApiClient | None,
    channel_id: str,
    oldest: str | None,
) -> list[dict]:
    try:
        return primary_client.conversation_history(channel_id, oldest=oldest)
    except SlackSyncLimitError:
        raise
    except SlackApiError:
        if fallback_client is None:
            raise
        return fallback_client.conversation_history(channel_id, oldest=oldest)


def _conversation_replies_with_fallback(
    *,
    primary_client: SlackApiClient,
    fallback_client: SlackApiClient | None,
    channel_id: str,
    thread_ts: str,
    oldest: str | None,
) -> list[dict]:
    try:
        return primary_client.conversation_replies(channel_id, thread_ts, oldest=oldest)
    except SlackSyncLimitError:
        raise
    except SlackApiError:
        if fallback_client is None:
            raise
        return fallback_client.conversation_replies(channel_id, thread_ts, oldest=oldest)


def _retry_after_seconds(response: httpx.Response) -> float:
    try:
        return max(float(response.headers.get('Retry-After', '1')), 0.0)
    except ValueError:
        return 1.0
