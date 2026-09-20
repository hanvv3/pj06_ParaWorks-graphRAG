from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime import EvidencePacket, PermissionContext
from backend.app.agents.slack_agent import (
    SlackAgent,
    SlackAgentModelResponse,
    build_slack_evidence_packet,
    create_slack_agent_review_items,
)
from backend.app.models import (
    AgentRun,
    Document,
    DocumentChunk,
    DocumentVersion,
    ReviewItem,
    Source,
)


class FakeSlackModel:
    def extract(self, packet: EvidencePacket) -> SlackAgentModelResponse:
        assert packet.source_type == 'slack'
        assert packet.strictest_permission == 'restricted'
        return SlackAgentModelResponse(
            title='Redis decision timeline',
            summary='Redis was selected for job progress updates.',
            item_type='history_event',
            confidence_score=0.88,
            input_tokens=700,
            output_tokens=140,
        )


class InternalSlackModel:
    def extract(self, packet: EvidencePacket) -> SlackAgentModelResponse:
        assert packet.source_type == 'slack'
        assert packet.strictest_permission == 'internal'
        return SlackAgentModelResponse(
            title='Ranked evidence decision',
            summary='Ranked Slack evidence was selected for review.',
            item_type='history_event',
            confidence_score=0.86,
            input_tokens=600,
            output_tokens=120,
        )


def seed_slack_chunk(db: Session, permission_level: str = 'restricted') -> None:
    source = Source(
        source_type='slack',
        source_id='C123:1777600800.000100',
        source_url='https://example.slack.com/archives/C123/p1777600800000100',
        title='Slack message in C123',
        author='U123',
        permission_level=permission_level,
        raw_metadata={'ts': '1777600800.000100', 'channel_id': 'C123'},
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body='Redis로 진행 상태를 관리합니다.')
    db.add(version)
    db.flush()

    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            chunk_index=0,
            text='Redis로 진행 상태를 관리합니다.',
            source_snippet='Redis로 진행 상태를 관리합니다.',
            permission_level=permission_level,
            metadata_={'source_url': source.source_url, 'source_type': 'slack'},
        )
    )
    db.commit()


def seed_slack_chunk_with_ts(db: Session, source_id: str, ts: str, body: str) -> None:
    source = Source(
        source_type='slack',
        source_id=source_id,
        source_url=f'https://example.slack.com/archives/C123/p{ts.replace(".", "")}',
        title='Slack message in C123',
        author='U123',
        permission_level='internal',
        raw_metadata={'ts': ts, 'channel_id': 'C123'},
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body=body)
    db.add(version)
    db.flush()

    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            chunk_index=0,
            text=body,
            source_snippet=body,
            permission_level='internal',
            metadata_={'source_url': source.source_url, 'source_type': 'slack'},
        )
    )


def test_slack_agent_bridge_persists_review_item(db_session: Session) -> None:
    seed_slack_chunk(db_session)
    agent = SlackAgent(model=FakeSlackModel())

    created = create_slack_agent_review_items(
        db=db_session,
        agent=agent,
        permission_context=PermissionContext(
            user_id='demo-admin', role='admin',
            allowed_permission_levels=('public', 'internal', 'restricted'),
        ),
        source_window='C123:2026-05-01',
    )

    assert len(created) == 1
    stored = db_session.scalars(select(ReviewItem)).one()
    assert stored.status == 'pending_review'
    assert stored.item_type == 'history_event'
    assert stored.payload['title'] == 'Redis decision timeline'
    assert stored.payload['summary'] == 'Redis was selected for job progress updates.'
    assert stored.payload['agent_name'] == 'slack_agent'
    assert stored.payload['prompt_version'] == 'slack-timeline:v1'
    assert stored.payload['token_usage']['total_tokens'] == 840
    assert stored.payload['estimated_cost_usd'] > 0
    assert stored.permission_level == 'restricted'
    assert stored.source_links == ['https://example.slack.com/archives/C123/p1777600800000100']


def test_slack_evidence_packet_can_select_recent_bounded_window(db_session: Session) -> None:
    seed_slack_chunk_with_ts(db_session, 'C123:1.000100', '1.000100', 'oldest')
    seed_slack_chunk_with_ts(db_session, 'C123:3.000100', '3.000100', 'newest')
    seed_slack_chunk_with_ts(db_session, 'C123:2.000100', '2.000100', 'middle')
    db_session.commit()

    packet = build_slack_evidence_packet(
        db=db_session,
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
        source_window='slack:live:recent:2',
        max_messages=2,
        newest_first=True,
    )

    assert [message.source_id for message in packet.messages] == ['C123:3.000100', 'C123:2.000100']


def test_slack_evidence_packet_can_select_ranked_deduped_window(db_session: Session) -> None:
    seed_slack_chunk_with_ts(db_session, 'C123:1.000100', '1.000100', '결정: Postgres와 pgvector를 사용합니다. 비용 상한을 둡니다.')
    seed_slack_chunk_with_ts(db_session, 'C123:2.000100', '2.000100', 'TODO: Slack API 권한을 확인하고 배포 전 테스트합니다.')
    seed_slack_chunk_with_ts(db_session, 'C123:3.000100', '3.000100', '결정: Postgres와 pgvector를 사용합니다. 비용 상한을 둡니다.')
    seed_slack_chunk_with_ts(db_session, 'C123:4.000100', '4.000100', 'ㅋㅋㅋ 좋아요')
    db_session.commit()

    packet = build_slack_evidence_packet(
        db=db_session,
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
        source_window='slack:live:ranked:2',
        max_messages=2,
        selection_strategy='ranked',
    )

    assert [message.source_id for message in packet.messages] == ['C123:3.000100', 'C123:2.000100']
    assert packet.messages[0].metadata['selection_strategy'] == 'ranked'
    assert packet.messages[0].metadata['evidence_rank'] == 1
    assert packet.messages[0].metadata['importance_score'] > packet.messages[1].metadata['importance_score']


def test_slack_agent_run_stores_ranked_evidence_summary(db_session: Session) -> None:
    seed_slack_chunk_with_ts(db_session, 'C123:1.000100', '1.000100', '결정: Postgres와 pgvector를 사용합니다. 비용 상한을 둡니다.')
    seed_slack_chunk_with_ts(db_session, 'C123:2.000100', '2.000100', 'TODO: Slack API 권한을 확인하고 배포 전 테스트합니다.')
    db_session.commit()
    agent = SlackAgent(model=InternalSlackModel())

    create_slack_agent_review_items(
        db=db_session,
        agent=agent,
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
        source_window='slack:live:ranked:2',
        max_messages=2,
        selection_strategy='ranked',
    )

    agent_run = db_session.scalars(select(AgentRun)).one()
    evidence_summary = agent_run.metadata_['evidence_summary']
    assert agent_run.metadata_['selection_strategy'] == 'ranked'
    assert evidence_summary[0]['rank'] == 1
    assert evidence_summary[0]['source_id'] == 'C123:1.000100'
    assert evidence_summary[0]['importance_score'] > 0
    assert 'Postgres' in evidence_summary[0]['snippet']
