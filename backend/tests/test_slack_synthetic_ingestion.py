"""S1 source observations only. Review/source authority belongs to S2."""
import httpx
import pytest
from sqlalchemy import select

from backend.app.connectors.registry import get_connector_manifest
from backend.app.connectors.slack import SlackApiError, SlackWebApiClient
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import DocumentChunk, ReviewItem, Source
from backend.tests.slack_synthetic_fixture import SyntheticSlackClient


def test_synthetic_channel_permission_and_raw_identifiers_are_preserved():
    fake = SyntheticSlackClient()
    events = fake.connector().fetch_events()
    public, _, private = events
    assert private.permission_level == 'restricted'
    assert public.permission_level == 'internal'
    assert public.raw_metadata['workspace_id'] == 'TSYNTHETIC'
    assert public.raw_metadata['synthetic'] is True
    assert public.raw_metadata['fixture_origin'] == 'slack-s1-invented-conversation-v1'
    assert public.raw_metadata['workspace_url'] == 'https://synthetic.invalid'
    assert public.raw_metadata['source_snippet'] == '결정: pgvector를 사용합니다.'
    assert public.raw_metadata['slack_participant_ids'] == ['UPARENT']
    assert public.raw_metadata['ts'] == fake.parent_ts
    assert public.semantic_timestamp_raw == fake.parent_ts
    assert public.source_url.endswith('/CPUBLIC/p' + fake.parent_ts.replace('.', ''))


def test_shared_sync_recovers_known_old_thread_behind_channel_cursor_and_replay(db_session):
    fake = SyntheticSlackClient()
    first = sync_connector_events(db_session, fake.connector())
    assert (first.fetched_events, first.created_review_items, first.skipped_events) == (3, 0, 0)
    assert len(first.changed_source_ids) == 3
    fake.add_late_reply()
    second = sync_connector_events(db_session, fake.connector())
    assert (second.fetched_events, second.created_review_items, second.skipped_events) == (1, 0, 0)
    assert second.changed_source_ids == [f'CPUBLIC:{fake.reply_ts}']
    source = db_session.scalar(select(Source).where(Source.source_id == f'CPUBLIC:{fake.reply_ts}'))
    assert source.raw_metadata['thread_parent_text'] == '결정: pgvector를 사용합니다.'
    assert source.raw_metadata['slack_participant_ids'] == ['UPARENT', 'UREPLY']
    assert set(source.raw_metadata['participants']) == {'가상 기획자', '가상 개발자'}
    assert source.raw_metadata['source_snippet'] == '할 일: <@UPARENT>와 색인을 검증합니다.'
    chunk = db_session.scalar(select(DocumentChunk).where(DocumentChunk.source_id == source.id))
    assert chunk.text == ('Thread parent: 결정: pgvector를 사용합니다.\n'
                          'Thread reply: 할 일: @가상 기획자와 색인을 검증합니다.')
    assert source.server_content_signature is None
    assert db_session.scalar(select(ReviewItem)) is None
    fake.replay = True  # inclusive/cursor-boundary duplicate delivery
    third = sync_connector_events(db_session, fake.connector())
    assert (third.fetched_events, third.created_review_items, third.skipped_events) == (1, 0, 1)
    assert len(db_session.scalars(select(Source)).all()) == 4
    reply_calls = [c for c in fake.calls if c[:3] == ('replies', 'CPUBLIC', fake.parent_ts)]
    assert reply_calls[-1][-1] == fake.reply_ts
    assert not any(c[:3] == ('replies', 'CPRIVATE', fake.parent_ts) for c in fake.calls)


def test_reply_pagination_refuses_partial_batch_at_bound():
    requests = []

    def handle(request):
        requests.append(request)
        assert len(requests) <= 20, 'unbounded pagination'
        return httpx.Response(200, json={'ok': True, 'messages': [],
                                        'response_metadata': {'next_cursor': str(len(requests))}})

    client = SlackWebApiClient(bot_token='fixture-only', http_client=httpx.Client(
        transport=httpx.MockTransport(handle)))
    # Repeating cursor also stops a broken API; no partial success/cursor advance.
    with pytest.raises(SlackApiError, match='page_limit_exceeded'):
        client.conversation_replies('CPUBLIC', '1.000001')
    assert len(requests) == 20


def test_fresh_thread_window_refuses_partial_sync(db_session):
    fake = SyntheticSlackClient()
    fake.histories['CPUBLIC'] = [dict(fake.message(f'{i}.000001', '합성 결정', 'UPARENT'),
                                     reply_count=1) for i in range(1, 52)]
    with pytest.raises(SlackApiError, match='thread_limit_exceeded'):
        sync_connector_events(db_session, fake.connector())
    assert db_session.scalar(select(Source)) is None


def test_same_thread_timestamp_in_other_channel_keeps_context_separate(db_session):
    fake = SyntheticSlackClient()
    fake.histories['CPRIVATE'] = [fake.message(fake.parent_ts, '비공개 결정', 'UPRIVATE')]
    sync_connector_events(db_session, fake.connector())
    fake.add_late_reply()
    fake.replies['CPRIVATE', fake.parent_ts] = [dict(fake.message(
        fake.reply_ts, '비공개 할 일', 'UREPLY'), thread_ts=fake.parent_ts)]
    result = sync_connector_events(db_session, fake.connector())
    assert result.fetched_events == 2
    private = db_session.scalar(select(Source).where(Source.source_id == f'CPRIVATE:{fake.reply_ts}'))
    assert private.permission_level == 'restricted'
    assert private.raw_metadata['thread_parent_text'] == '비공개 결정'
    public = db_session.scalar(select(Source).where(Source.source_id == f'CPUBLIC:{fake.reply_ts}'))
    assert public.raw_metadata['thread_parent_text'] == '결정: pgvector를 사용합니다.'


def test_known_message_window_and_unknown_thread_discovery_limits(db_session):
    fake = SyntheticSlackClient()
    # Recent history snapshot with 51 roots, no initial replies.
    fake.histories['CPUBLIC'] = [fake.message(f'{i}.000001', f'합성 결정 {i}', 'UPARENT')
                                for i in range(1, 52)]
    fake.histories['CPRIVATE'] = []
    sync_connector_events(db_session, fake.connector())
    fake.calls.clear()
    fake.histories['CPUBLIC'] = []
    fake.replies['CPUBLIC', '1.000001'] = [fake.message('100.000001', '창 밖 답글', 'UREPLY')]
    fake.replies['CPUBLIC', '0.000001'] = [fake.message('101.000001', '미발견 답글', 'UREPLY')]
    result = sync_connector_events(db_session, fake.connector())
    assert result.fetched_events == 0
    calls = [c for c in fake.calls if c[0] == 'replies']
    assert len(calls) == 50
    assert {c[2] for c in calls} == {f'{i}.000001' for i in range(2, 52)}


def test_live_registry_exposes_same_scopes_as_actual_slack_adapter():
    assert get_connector_manifest('slack', demo_mode=False).required_scopes == (
        SyntheticSlackClient().connector().manifest.required_scopes)


def test_known_thread_keeps_strictest_permission_across_observations():
    fake = SyntheticSlackClient()
    fake.add_late_reply()
    events = fake.connector().fetch_events_since({}, known_messages_by_channel={
        'CPUBLIC': [
            {'ts': fake.parent_ts, 'thread_ts': fake.parent_ts, 'raw_text': '제한 근거',
             'slack_user_id': 'UPARENT', 'permission_level': 'restricted'},
            {'ts': str(float(fake.parent_ts) + 1), 'thread_ts': fake.parent_ts,
             'thread_parent_text': '제한 근거', 'thread_parent_user_id': 'UPARENT',
             'permission_level': 'internal'},
        ]})
    assert len(events) == 1
    assert events[0].permission_level == 'restricted'
