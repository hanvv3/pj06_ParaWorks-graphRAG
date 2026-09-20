"""New local synthetic inputs cross the real ingestion/authority boundary."""

import pytest
from sqlalchemy import select

from backend.app.ingestion.source_authority import resolve_exact_source_authority
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import Source
from backend.tests.slack_synthetic_fixture import SyntheticSlackClient


def local_connector(fake):
    from backend.app.connectors import slack_synthetic

    return slack_synthetic.LocalSyntheticSlackConnector(
        channels=fake.conversations_list(),
        messages_by_channel=fake.histories,
        replies_by_thread=fake.replies,
        users=fake.users_list(),
    )


def test_only_explicit_local_adapter_grants_new_source_authority(db_session):
    fake = SyntheticSlackClient()
    # External synthetic flags and a mock manifest grant no authority.
    sync_connector_events(db_session, fake.connector())
    assert all(
        s.server_content_signature is None for s in db_session.scalars(select(Source))
    )
    fake.histories = {
        'CPUBLIC': [fake.message('1777600801.000001', '결정: 새 합성 근거', 'UPARENT')]
    }
    fake.replies = {}
    result = sync_connector_events(db_session, local_connector(fake))
    source = db_session.scalar(
        select(Source).where(Source.source_id == 'CPUBLIC:1777600801.000001')
    )
    assert resolve_exact_source_authority(db_session, source=source) is not None
    assert result.changed_source_refs[0].source_id == source.source_id


def test_local_adapter_refuses_unsigned_id_collision(db_session):
    fake = SyntheticSlackClient()
    sync_connector_events(db_session, fake.connector())
    with pytest.raises(ValueError, match='unsigned Slack source collision'):
        sync_connector_events(db_session, local_connector(fake))
    assert all(
        s.server_content_signature is None for s in db_session.scalars(select(Source))
    )


def test_unverifiable_old_signature_cannot_upgrade_legacy_collision(db_session):
    from sqlalchemy.exc import IntegrityError

    fake = SyntheticSlackClient()
    sync_connector_events(db_session, fake.connector())
    # Negative corruption fixture, never a valid signature or approval seed.
    for source in db_session.scalars(select(Source)):
        source.server_content_signature = 'unverifiable-legacy-value'
    with pytest.raises(
        IntegrityError, match='ck_sources_server_content_signature_authority'
    ):
        db_session.commit()
    db_session.rollback()
    with pytest.raises(ValueError, match='unsigned Slack source collision'):
        sync_connector_events(db_session, local_connector(fake))


def test_normal_connector_cannot_overwrite_signed_synthetic_source(db_session):
    fake = SyntheticSlackClient()
    sync_connector_events(db_session, local_connector(fake))
    with pytest.raises(ValueError, match='signed synthetic Slack source'):
        sync_connector_events(db_session, fake.connector())


def test_signed_reply_does_not_launder_cached_unsigned_parent(db_session):
    fake = SyntheticSlackClient()
    sync_connector_events(db_session, fake.connector())
    fake.add_late_reply()
    sync_connector_events(db_session, local_connector(fake))
    source = db_session.scalar(
        select(Source).where(Source.source_id == f'CPUBLIC:{fake.reply_ts}')
    )
    authority = resolve_exact_source_authority(db_session, source=source)
    assert authority is not None
    assert 'pgvector' not in authority.version.body
    assert 'pgvector' not in str(source.raw_metadata)
    assert authority.version.body == '할 일: @가상 기획자와 색인을 검증합니다.'


def test_synthetic_replay_and_edit_use_canonical_server_signature(db_session):
    fake = SyntheticSlackClient()
    first = sync_connector_events(db_session, local_connector(fake))
    replay = sync_connector_events(db_session, local_connector(fake))
    assert (first.fetched_events, replay.skipped_events) == (3, 3)
    old_ref = first.changed_source_refs[0]
    fake.histories['CPUBLIC'][0]['text'] = '결정: 변경된 현재 합성 근거입니다.'
    edited = sync_connector_events(db_session, local_connector(fake))
    assert len(edited.changed_source_refs) == 1
    assert (
        edited.changed_source_refs[0].version_or_signature
        != old_ref.version_or_signature
    )


def test_local_adapter_does_not_accept_network_clients():
    from backend.app.connectors import slack_synthetic

    with pytest.raises(TypeError):
        slack_synthetic.LocalSyntheticSlackConnector(client=object())


class RecordingSlackModel:
    def __init__(self):
        self.packets = []

    def extract(self, packet):
        from backend.app.agents.slack_agent import SlackAgentModelResponse

        self.packets.append(packet)
        return SlackAgentModelResponse(
            title='합성 pgvector 결정',
            summary='합성 검색 색인을 검증합니다.',
            item_type='history_event',
            confidence_score=0.65,
            input_tokens=120,
            output_tokens=35,
            uncertainty_reason='합성 대화이며 실제 회사 결정이 아닙니다.',
            extra_fields={'reason': '합성 대화의 결정과 후속 검증에 근거합니다.'},
        )


def draft(db, model, *, user=None, source_ids=None, settings=None):
    from backend.app.agent_runtime.slack_synthetic_review import (
        create_local_slack_review_items,
    )
    from backend.app.core.demo_auth import USERS

    return create_local_slack_review_items(
        db=db,
        model=model,
        user=user or USERS['admin'],
        source_ids=source_ids,
        settings=settings,
    )


def test_local_review_uses_current_evidence_pending_and_replay_cache(db_session):
    from backend.app.models import HistoryEvent, ReviewItem, ReviewItemEvidenceRef

    fake = SyntheticSlackClient()
    sync_connector_events(db_session, local_connector(fake))
    model = RecordingSlackModel()
    created = draft(db_session, model)
    assert len(created) == 1
    item = created[0]
    assert item.status == 'pending_review'
    assert item.candidate_contract_version == 'c5-v1'
    assert item.payload['agent_name'] == 'slack_agent'
    assert item.payload['token_usage']['total_tokens'] == 155
    assert item.payload['uncertainty_reason']
    assert item.confidence_score == 0.65
    assert item.permission_level == 'restricted'
    assert db_session.scalar(select(ReviewItemEvidenceRef)) is not None
    assert db_session.scalar(select(HistoryEvent)) is None
    assert draft(db_session, model) == []
    assert len(model.packets) == 1
    assert len(db_session.scalars(select(ReviewItem)).all()) == 1


def test_local_review_filters_permissions_before_model_and_ignores_unsigned(db_session):
    from backend.app.core.demo_auth import USERS

    fake = SyntheticSlackClient()
    sync_connector_events(db_session, fake.connector())
    model = RecordingSlackModel()
    assert draft(db_session, model) == []
    assert model.packets == []
    fake.histories['CPUBLIC'] = [
        fake.message('1777600801.000001', '결정: pgvector 사용', 'UPARENT')
    ]
    fake.histories['CPRIVATE'] = [
        fake.message('1777600802.000001', '극비 합성 예산', 'UPRIVATE')
    ]
    sync_connector_events(db_session, local_connector(fake))
    created = draft(db_session, model, user=USERS['viewer'])
    assert created[0].permission_level == 'internal'
    assert len(model.packets) == 1
    assert all(m.permission_level == 'internal' for m in model.packets[0].messages)
    assert '극비' not in str(model.packets)


def test_local_review_reply_context_uses_current_parent_version(db_session):
    fake = SyntheticSlackClient()
    sync_connector_events(db_session, local_connector(fake))
    fake.histories['CPUBLIC'][0]['text'] = '결정: 현재 부모 본문 pgvector 검증'
    sync_connector_events(db_session, local_connector(fake))
    fake.add_late_reply()
    sync_connector_events(db_session, local_connector(fake))
    model = RecordingSlackModel()
    created = draft(db_session, model, source_ids=[f'CPUBLIC:{fake.reply_ts}'])
    assert len(created) == 1
    messages = model.packets[0].messages
    assert {m.source_id for m in messages} == {
        f'CPUBLIC:{fake.parent_ts}',
        f'CPUBLIC:{fake.reply_ts}',
    }
    assert any('현재 부모 본문' in m.text for m in messages)
    assert all('Thread parent:' not in m.text for m in messages)


def approve(db, item, *, user=None, settings=None):
    from backend.app.core.demo_auth import USERS
    from backend.app.review.actors import human_review_actor
    from backend.app.review.transitions import ReviewTransitionService

    result = ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(user or USERS['admin']),
    )
    db.commit()
    return result


def history_documents(db):
    from backend.app.rag.indexing import build_rag_index_documents

    return [
        d
        for d in build_rag_index_documents(db)
        if d.document_id.startswith('history_event:')
    ]


def test_human_approval_then_index_search_and_idempotent_retry(db_session):
    from backend.app.core.demo_auth import USERS
    from backend.app.models import HistoryEvent, TrustedKnowledgeEvidenceLink
    from backend.app.rag.vector_store import InMemoryVectorStore

    fake = SyntheticSlackClient()
    sync_connector_events(db_session, local_connector(fake))
    item = draft(db_session, RecordingSlackModel())[0]
    assert history_documents(db_session) == []
    first = approve(db_session, item)
    second = approve(db_session, item)
    assert not first.replayed and second.replayed
    assert db_session.scalar(select(HistoryEvent)) is not None
    assert db_session.scalar(select(TrustedKnowledgeEvidenceLink)) is not None
    documents = history_documents(db_session)
    assert len(documents) == 1
    store = InMemoryVectorStore()
    store.upsert_many(documents)
    assert store.search(query='pgvector', user=USERS['admin']).matches
    hidden = store.search(query='pgvector', user=USERS['viewer'])
    assert not hidden.matches and hidden.hidden_match_count == 1


@pytest.mark.parametrize('mutation', ['edit', 'delete', 'restrict'])
def test_source_lifecycle_invalidates_bound_knowledge_through_sync(
    db_session, mutation
):
    fake = SyntheticSlackClient()
    fake.histories['CPRIVATE'] = []
    sync_connector_events(db_session, local_connector(fake))
    fake.add_late_reply()
    sync_connector_events(db_session, local_connector(fake))
    item = draft(
        db_session, RecordingSlackModel(), source_ids=[f'CPUBLIC:{fake.reply_ts}']
    )[0]
    approve(db_session, item)
    assert len(history_documents(db_session)) == 1
    from backend.app.connectors.slack_synthetic import LocalSyntheticSlackConnector

    message = fake.message(fake.parent_ts, '결정: pgvector를 사용합니다.', 'UPARENT')
    channels = fake.conversations_list()
    deleted = []
    if mutation == 'edit':
        message['text'] = '결정: 부모 내용 변경, 새 검토 필요'
    elif mutation == 'delete':
        deleted = [('CPUBLIC', fake.parent_ts)]
    else:
        channels[0]['is_private'] = True
    connector = LocalSyntheticSlackConnector(
        channels=channels,
        messages_by_channel={'CPUBLIC': [message]} if not deleted else {},
        users=fake.users_list(),
        deleted_messages=deleted,
    )
    result = sync_connector_events(db_session, connector)
    assert result.changed_source_ids == [f'CPUBLIC:{fake.parent_ts}']
    documents = history_documents(db_session)
    if mutation == 'restrict':
        assert all(d.permission_level == 'restricted' for d in documents)
    else:
        assert documents == []
    if mutation == 'delete':
        parent = db_session.scalar(
            select(Source).where(Source.source_id == f'CPUBLIC:{fake.parent_ts}')
        )
        assert resolve_exact_source_authority(db_session, source=parent) is None


def test_public_review_contract_and_catalog_still_exclude_slack():
    from pydantic import ValidationError

    from backend.app.agent_runtime.review_v2_agents import build_review_agent_catalog
    from backend.app.core.config import Settings
    from backend.app.schemas.review_workflow import (
        ReviewWorkflowSourceRef,
        normalize_agent_names,
    )

    assert 'slack_agent' not in build_review_agent_catalog(Settings()).registry.names
    with pytest.raises(ValueError):
        normalize_agent_names(['slack_agent'])
    with pytest.raises(ValidationError):
        ReviewWorkflowSourceRef(
            source_type='slack',
            source_id='CPUBLIC:1777600801.000001',
            version_or_signature='a' * 64,
        )


def test_signed_slack_can_back_existing_v2_trusted_knowledge(db_session):
    from backend.app.core.config import get_settings
    from backend.app.rag.indexing import build_rag_v2_index_documents

    fake = SyntheticSlackClient()
    sync_connector_events(db_session, local_connector(fake))
    assert not any(
        d.document_id.startswith('history_event:')
        for d in build_rag_v2_index_documents(db_session, settings=get_settings())
    )
    item = draft(db_session, RecordingSlackModel())[0]
    approve(db_session, item)
    documents = build_rag_v2_index_documents(db_session, settings=get_settings())
    assert any(d.document_id.startswith('history_event:') for d in documents)


def test_missing_source_never_calls_model_or_creates_candidate(db_session):
    model = RecordingSlackModel()
    assert draft(db_session, model) == []
    assert model.packets == []


def test_registered_prompt_change_invalidates_local_agent_cache(
    db_session, monkeypatch
):
    from dataclasses import replace

    from backend.app.agent_runtime import slack_synthetic_review as service
    from backend.app.agents.slack_agent import agent as agent_module

    sync_connector_events(db_session, local_connector(SyntheticSlackClient()))
    model = RecordingSlackModel()
    assert len(draft(db_session, model)) == 1
    assert draft(db_session, model) == []
    manifest = replace(
        service.SLACK_AGENT_MANIFEST, prompt_versions=('slack-timeline:v2',)
    )
    monkeypatch.setattr(agent_module, 'SLACK_AGENT_PROMPT_VERSION', 'slack-timeline:v2')
    monkeypatch.setattr(service, 'SLACK_AGENT_MANIFEST', manifest)
    monkeypatch.setattr(
        service._LocalSyntheticSlackReviewCatalog,
        'approved_manifests',
        {'slack_agent': manifest},
    )
    assert len(draft(db_session, model)) == 1
    assert len(model.packets) == 2


def test_current_human_review_denies_unauthorized_and_stale_pending(db_session):
    from fastapi import HTTPException

    from backend.app.core.demo_auth import USERS
    from backend.app.models import HistoryEvent

    fake = SyntheticSlackClient()
    sync_connector_events(db_session, local_connector(fake))
    item = draft(db_session, RecordingSlackModel())[0]
    with pytest.raises(HTTPException) as denied:
        approve(db_session, item, user=USERS['viewer'])
    assert denied.value.status_code == 403
    fake.histories['CPUBLIC'][0]['text'] = '결정: 내용 변경으로 재검토가 필요합니다.'
    sync_connector_events(db_session, local_connector(fake))
    with pytest.raises((ValueError, HTTPException)):
        approve(db_session, item)
    assert db_session.scalar(select(HistoryEvent)) is None


def test_actual_langchain_runnable_and_graph_use_only_visible_evidence(db_session):
    import json

    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from backend.app.agents.slack_agent import LangChainSlackAgentModel
    from backend.app.core.demo_auth import USERS

    sync_connector_events(db_session, local_connector(SyntheticSlackClient()))
    seen = []

    def respond(messages):
        seen.append(messages)
        return AIMessage(
            content=json.dumps(
                {
                    'title': '합성 검색 결정',
                    'summary': '공개 합성 채널에서 pgvector 사용을 결정했습니다.',
                    'item_type': 'history_event',
                    'confidence_score': 0.65,
                    'uncertainty_reason': '합성 대화이며 실제 회사 사실이 아닙니다.',
                },
                ensure_ascii=False,
            ),
            usage_metadata={
                'input_tokens': 200,
                'output_tokens': 40,
                'total_tokens': 240,
            },
        )

    model = LangChainSlackAgentModel(
        provider='fake', model_name='fake-runnable', chat_model=RunnableLambda(respond)
    )
    items = draft(db_session, model, user=USERS['viewer'])
    assert len(items) == 1 and len(seen) == 1
    assert '비공개 예산' not in str(seen)
    assert items[0].payload['token_usage']['total_tokens'] == 240


@pytest.mark.parametrize('signed', [False, True])
@pytest.mark.parametrize(
    'body,expected',
    [
        ('Redis should support queue and job progress workflows.', False),
        ('Redis is used for queue and job progress workflows.', True),
    ],
)
def test_historical_ranked_signal_failure_is_independent_of_signing(
    db_session, signed, body, expected
):
    from backend.app.agent_runtime import PermissionContext
    from backend.app.agents.slack_agent import build_slack_evidence_packet

    fake = SyntheticSlackClient()
    fake.histories = {
        'CPUBLIC': [fake.message(fake.parent_ts, body, 'UPARENT')],
        'CPRIVATE': [],
    }
    sync_connector_events(
        db_session, local_connector(fake) if signed else fake.connector()
    )
    packet = build_slack_evidence_packet(
        db=db_session,
        permission_context=PermissionContext('tester', 'reviewer'),
        source_window='synthetic:signal',
        selection_strategy='ranked',
    )
    assert bool(packet.messages) is expected
