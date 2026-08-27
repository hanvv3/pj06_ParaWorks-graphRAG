from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime import EvidencePacket, PermissionContext
from backend.app.agents.slack_agent import (
    SlackAgent,
    SlackAgentModelResponse,
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
        return SlackAgentModelResponse(
            title='Redis decision timeline',
            summary='Redis was selected for job progress updates.',
            item_type='history_event',
            confidence_score=0.88,
            input_tokens=700,
            output_tokens=140,
        )


def seed_slack_chunk(db: Session) -> None:
    source = Source(
        source_type='slack',
        source_id='C123:1777600800.000100',
        source_url='https://example.slack.com/archives/C123/p1777600800000100',
        title='Slack message in C123',
        author='U123',
        permission_level='internal',
        raw_metadata={'ts': '1777600800.000100', 'channel_id': 'C123'},
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body='Redis manages job progress state.')
    db.add(version)
    db.flush()

    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            chunk_index=0,
            text='Redis manages job progress state.',
            source_snippet='Redis manages job progress state.',
            permission_level='internal',
            metadata_={'source_url': source.source_url, 'source_type': 'slack'},
        )
    )
    db.commit()


def test_slack_agent_bridge_persists_agent_run(db_session: Session) -> None:
    seed_slack_chunk(db_session)
    agent = SlackAgent(model=FakeSlackModel())

    created = create_slack_agent_review_items(
        db=db_session,
        agent=agent,
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
        source_window='C123:2026-05-01',
    )

    assert len(created) == 1
    agent_run = db_session.scalars(select(AgentRun)).one()
    review_item = db_session.scalars(select(ReviewItem)).one()
    assert agent_run.agent_name == 'slack_agent'
    assert agent_run.prompt_version == 'slack-timeline:v1'
    assert agent_run.status == 'complete'
    assert agent_run.source_window == 'C123:2026-05-01'
    assert agent_run.input_tokens == 700
    assert agent_run.output_tokens == 140
    assert agent_run.total_tokens == 840
    assert agent_run.estimated_cost_usd > 0
    assert agent_run.permission_level == 'internal'
    assert agent_run.cache_key
    assert review_item.payload['agent_run_id'] == agent_run.id
