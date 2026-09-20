"""Delivered permission changes must survive unsigned Slack body deduplication."""

import pytest
from sqlalchemy import select

from backend.app.agent_runtime import PermissionContext
from backend.app.agent_runtime.company_memory import (
    run_company_memory_agent_orchestration,
)
from backend.app.agents.slack_agent.service import build_slack_evidence_packet
from backend.app.core.demo_auth import USERS, get_demo_user
from backend.app.ingestion.source_versions import source_version_ref
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import (
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    ReviewItem,
    Source,
)
from backend.tests.slack_synthetic_fixture import SyntheticSlackClient


@pytest.mark.parametrize('source_already_restricted', [False, True])
def test_delivered_restriction_narrows_unsigned_source_and_existing_chunks(
    client, db_session, source_already_restricted
):
    # Real ordinary Slack adapter + shared sync, not the signed local adapter.
    fake = SyntheticSlackClient()
    fake.histories = {
        'CPUBLIC': [
            fake.message(fake.cursor_ts, 'Use pgvector for search.', 'UPARENT')
        ],
        'CPRIVATE': [],
    }
    channels = fake.conversations_list()
    fake.conversations_list = lambda: channels
    first = sync_connector_events(db_session, fake.connector())
    assert first.fetched_events == 1 and first.skipped_events == 0
    source = db_session.scalar(select(Source))
    chunk = db_session.scalar(select(DocumentChunk))
    assert source.permission_level == chunk.permission_level == 'internal'
    signature = source.connector_content_signature
    version_ids = tuple(db_session.scalars(select(DocumentVersion.id)))
    parser_ids = tuple(db_session.scalars(select(DocumentParserRun.id)))
    chunk_id = chunk.id
    if source_already_restricted:
        # Existing inconsistent storage must not let a Source-only comparison skip.
        source.permission_level = 'restricted'
        db_session.commit()

    channels[0]['is_private'] = True
    second = sync_connector_events(db_session, fake.connector())
    db_session.refresh(source)
    db_session.refresh(chunk)
    assert source.permission_level == chunk.permission_level == 'restricted'
    assert chunk.metadata_['permission_level'] == 'restricted'
    assert (
        second.changed_source_ids == [source.source_id] and second.skipped_events == 0
    )
    assert source.connector_content_signature == signature
    assert (
        source.server_content_signature is None and source_version_ref(source) is None
    )

    assert chunk.id == chunk_id
    assert tuple(db_session.scalars(select(DocumentVersion.id))) == version_ids
    assert tuple(db_session.scalars(select(DocumentParserRun.id))) == parser_ids

    client.app.dependency_overrides[get_demo_user] = lambda: USERS['viewer']
    response = client.post('/api/v1/integrations/slack/agent-review')
    assert response.status_code == 200 and response.json()['created_review_items'] == 0
    memory = run_company_memory_agent_orchestration(
        db=db_session, user=USERS['viewer'], question='pgvector search'
    )
    assert memory.outputs['slack_review_items_created'] == 0
    assert memory.outputs['cost_plan']['slack_agent']['budget_status'] == 'no_input'
    assert db_session.scalar(select(ReviewItem)) is None

    replay = sync_connector_events(db_session, fake.connector())
    assert replay.skipped_events == 1 and replay.changed_source_ids == []
    # An unchanged lower-permission delivery must not broaden stored evidence.
    channels[0]['is_private'] = False
    lower = sync_connector_events(db_session, fake.connector())
    db_session.refresh(source)
    db_session.refresh(chunk)
    assert lower.skipped_events == 1 and lower.changed_source_ids == []
    assert source.permission_level == chunk.permission_level == 'restricted'
    assert tuple(db_session.scalars(select(DocumentVersion.id))) == version_ids
    assert tuple(db_session.scalars(select(DocumentParserRun.id))) == parser_ids
    assert (
        source.server_content_signature is None and source_version_ref(source) is None
    )


@pytest.mark.parametrize(
    'source_permission,chunk_permission',
    [
        ('restricted', 'internal'),
        ('internal', 'restricted'),
        ('restricted', 'restricted'),
    ],
)
def test_slack_packet_filters_both_permissions_before_windowing(
    db_session, source_permission, chunk_permission
):
    fake = SyntheticSlackClient()
    fake.histories = {
        'CPUBLIC': [
            fake.message(fake.parent_ts, 'Use restricted search policy.', 'UPARENT'),
            fake.message(fake.cursor_ts, 'Use visible search policy.', 'UPARENT'),
        ],
        'CPRIVATE': [],
    }
    sync_connector_events(db_session, fake.connector())
    sources = db_session.scalars(select(Source).order_by(Source.id)).all()
    chunks = db_session.scalars(select(DocumentChunk).order_by(DocumentChunk.id)).all()
    sources[0].permission_level = source_permission
    chunks[0].permission_level = chunk_permission
    db_session.commit()
    viewer = PermissionContext('viewer', 'reviewer', ('public', 'internal'))
    packet = build_slack_evidence_packet(
        db=db_session, permission_context=viewer, source_window='test', max_messages=1
    )
    assert [message.text for message in packet.messages] == [
        'Use visible search policy.'
    ]
    admin = PermissionContext('admin', 'admin', ('public', 'internal', 'restricted'))
    authorized = build_slack_evidence_packet(
        db=db_session, permission_context=admin, source_window='test', max_messages=1
    )
    assert [message.text for message in authorized.messages] == [
        'Use restricted search policy.'
    ]
    assert authorized.strictest_permission == 'restricted'
