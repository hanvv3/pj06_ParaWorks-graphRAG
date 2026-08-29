from backend.app.models import ReviewItem, Source


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
    client.post('/api/v1/integrations/drive/sync')
    _human_approve_synced_sources(db_session)
    response = client.post('/api/v1/search', headers={'X-Demo-User': 'viewer'}, json={'query': 'confidential pricing'})
    assert response.status_code == 200
    assert response.json()['results'] == []
    assert response.json()['permission_notice'] == 'Some sources may be hidden by permissions.'
    assert response.json()['hidden_match_count'] == 1


def test_admin_search_can_see_restricted_drive_content(
    client, db_session
) -> None:
    client.post('/api/v1/integrations/drive/sync')
    _human_approve_synced_sources(db_session)
    response = client.post('/api/v1/search', headers={'X-Demo-User': 'admin'}, json={'query': 'confidential pricing'})
    assert response.status_code == 200
    assert len(response.json()['results']) == 1
    assert response.json()['results'][0]['source_id'] == 'drive-permission-leakage-case'
    assert response.json()['hidden_match_count'] == 0
