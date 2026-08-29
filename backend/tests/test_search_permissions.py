from backend.app.connectors.mock import get_mock_connector
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import DocumentParserRun, ReviewItem, Source


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
    _human_approve_synced_sources(db_session)
    response = client.post('/api/v1/search', headers={'X-Demo-User': 'admin'}, json={'query': 'confidential pricing'})
    assert response.status_code == 200
    assert len(response.json()['results']) == 1
    assert response.json()['results'][0]['source_id'] == 'drive:permission-leakage-case'
    assert response.json()['hidden_match_count'] == 0


def test_wrong_allowed_source_mime_is_absent_from_search_ask_and_hidden_count(
    client, db_session
) -> None:
    sync_connector_events(db=db_session, connector=get_mock_connector('drive'))
    _human_approve_synced_sources(db_session)
    source = (
        db_session.query(Source)
        .filter_by(source_id='drive:permission-leakage-case')
        .one()
    )
    parser_run = (
        db_session.query(DocumentParserRun)
        .filter_by(source_id=source.id)
        .one()
    )
    assert source.raw_metadata['mime_type'] != 'application/pdf'
    parser_run.mime_type = 'application/pdf'
    db_session.commit()

    admin_search = client.post(
        '/api/v1/search',
        headers={'X-Demo-User': 'admin'},
        json={'query': 'confidential pricing'},
    )
    viewer_search = client.post(
        '/api/v1/search',
        headers={'X-Demo-User': 'viewer'},
        json={'query': 'confidential pricing'},
    )
    viewer_ask = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'confidential pricing'},
    )

    assert admin_search.status_code == 200
    assert admin_search.json()['results'] == []
    assert admin_search.json()['hidden_match_count'] == 0
    assert viewer_search.status_code == 200
    assert viewer_search.json()['results'] == []
    assert viewer_search.json()['hidden_match_count'] == 0
    assert viewer_ask.status_code == 200
    assert viewer_ask.json()['source_ids'] == []
    assert viewer_ask.json()['hidden_match_count'] == 0
