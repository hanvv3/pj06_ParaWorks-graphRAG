from hashlib import sha256

from backend.app.connectors.mock import get_mock_connector
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import (
    AgentRun,
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    ReviewItem,
    Source,
)


def _establish_c5_source_authority(db_session) -> None:
    for source in db_session.query(Source).all():
        signature = sha256(source.source_id.encode()).hexdigest()
        document = db_session.query(Document).filter_by(source_id=source.id).one()
        version = (
            db_session.query(DocumentVersion)
            .filter_by(document_id=document.id)
            .one()
        )
        parser_run = (
            db_session.query(DocumentParserRun)
            .filter_by(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
            )
            .one()
        )
        source.server_content_signature_schema = 'server-source-content:v1'
        source.server_content_signature = signature
        document.current_document_version_id = version.id
        parser_run.server_content_signature_schema = 'server-source-content:v1'
        parser_run.server_content_signature = signature
        parser_run.parser_policy_version = 'parser-policy:v1'
        parser_run.parser_version = 'source-event:v1'
        parser_run.chunk_policy_version = 'chunk-policy:v1'
        for chunk in db_session.query(DocumentChunk).filter_by(
            source_id=source.id,
            version_id=version.id,
        ):
            chunk.parser_run_id = parser_run.id
    db_session.commit()


def _human_approve_synced_sources(db_session) -> None:
    for source in db_session.query(Source).all():
        db_session.add(
            ReviewItem(
                item_type='source_evidence',
                payload={'source_ids': [source.source_id]},
                source_links=[source.source_url],
                source_snippets=[source.title],
                confidence_score=1.0,
                permission_level=source.permission_level,
                status='approved',
                resolution_source='human',
            )
        )
    db_session.commit()


def test_ask_api_answers_with_visible_sources(client, db_session) -> None:
    sync_connector_events(db=db_session, connector=get_mock_connector('gmail'))
    _establish_c5_source_authority(db_session)
    _human_approve_synced_sources(db_session)

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis job state'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['agent_name'] == 'rag_orchestrator_agent'
    assert payload['answer']
    assert payload['source_links']
    assert payload['source_ids']
    assert payload['estimated_cost_usd'] > 0
    assert payload['token_usage']['total_tokens'] > 0
    agent_run = db_session.query(AgentRun).filter_by(agent_name='rag_orchestrator_agent').one()
    assert payload['agent_run_id'] == agent_run.id


def test_ask_api_respects_viewer_permissions(client, db_session) -> None:
    sync_connector_events(db=db_session, connector=get_mock_connector('drive'))
    _establish_c5_source_authority(db_session)
    _human_approve_synced_sources(db_session)

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'confidential pricing'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['source_links'] == []
    assert payload['source_ids'] == []
    assert payload['hidden_match_count'] == 1
    assert payload['permission_notice'] == 'Some sources may be hidden by permissions.'


def test_ask_api_answers_from_approved_knowledge(client, db_session) -> None:
    db_session.add(
        DecisionRecord(
            title='Use Redis for queues',
            decision_summary='Redis should power queue and job progress updates.',
            source_links=['https://knowledge.mock/redis-decision'],
            source_snippets=['Approved Redis decision snippet'],
            confidence_score=0.91,
            permission_level='internal',
            review_status='approved',
        )
    )
    db_session.commit()

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis queues'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['source_links'] == ['https://knowledge.mock/redis-decision']
    assert payload['source_ids'] == ['decision_record:1']
    assert payload['source_snippets'] == ['Approved Redis decision snippet']
    assert payload['hidden_match_count'] == 0
