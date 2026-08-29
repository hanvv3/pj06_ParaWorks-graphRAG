from hashlib import sha256

from backend.app.connectors.mock import get_mock_connector
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import (
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


def test_viewer_search_cannot_see_restricted_drive_content(
    client, db_session
) -> None:
    sync_connector_events(db=db_session, connector=get_mock_connector('drive'))
    _establish_c5_source_authority(db_session)
    _human_approve_synced_sources(db_session)
    response = client.post('/api/v1/search', headers={'X-Demo-User': 'viewer'}, json={'query': 'confidential pricing'})
    assert response.status_code == 200
    assert response.json()['results'] == []
    assert response.json()['permission_notice'] == 'Some sources may be hidden by permissions.'
    assert response.json()['hidden_match_count'] == 1


def test_admin_search_can_see_restricted_drive_content(
    client, db_session
) -> None:
    sync_connector_events(db=db_session, connector=get_mock_connector('drive'))
    _establish_c5_source_authority(db_session)
    _human_approve_synced_sources(db_session)
    response = client.post('/api/v1/search', headers={'X-Demo-User': 'admin'}, json={'query': 'confidential pricing'})
    assert response.status_code == 200
    assert len(response.json()['results']) == 1
    assert response.json()['results'][0]['source_id'] == 'drive:permission-leakage-case'
    assert response.json()['hidden_match_count'] == 0
