from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.models import ReviewItem


def test_employee_cannot_approve_review_item(client, db_session: Session) -> None:
    item = _add_review_item(db_session, permission_level='internal')

    response = client.post(f'/api/v1/review/{item.id}/approve', headers={'X-Demo-User': 'hanvv3@koreacu.ac.kr'})

    assert response.status_code == 403
    assert response.json()['detail'] == 'Review approval permission required.'


def test_reviewer_can_approve_internal_review_item(client, db_session: Session) -> None:
    item = _add_review_item(db_session, permission_level='internal')

    response = client.post(f'/api/v1/review/{item.id}/approve', headers={'X-Demo-User': 'mina@paraworks.com'})

    assert response.status_code == 200
    assert response.json()['status'] == 'approved'
    assert response.json()['reviewer_id'] == 'employee-mina'


def test_reviewer_cannot_approve_restricted_review_item(client, db_session: Session) -> None:
    item = _add_review_item(db_session, permission_level='restricted')

    response = client.post(f'/api/v1/review/{item.id}/approve', headers={'X-Demo-User': 'mina@paraworks.com'})

    assert response.status_code == 403
    assert response.json()['detail'] == 'Review approval permission required.'


def test_admin_can_approve_restricted_review_item(client, db_session: Session) -> None:
    item = _add_review_item(db_session, permission_level='restricted')

    response = client.post(f'/api/v1/review/{item.id}/approve', headers={'X-Demo-User': 'hanvv-admin'})

    assert response.status_code == 200
    assert response.json()['status'] == 'approved'
    assert response.json()['reviewer_id'] == 'google-hanvv-admin'


def test_approve_route_rechecks_exact_actor_permission_membership(client, db_session: Session) -> None:
    item = _add_review_item(db_session, permission_level='restricted')
    actor = DemoUser(
        id='admin-without-restricted',
        email='limited-admin@example.com',
        role='admin',
        permission_levels={'public', 'internal'},
        name='Limited Admin',
        title='Administrator',
        department='Platform',
    )
    client.app.dependency_overrides[get_demo_user] = lambda: actor

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 403
    assert db_session.get(ReviewItem, item.id).status == 'pending_review'


def test_unauthorized_human_bulk_empty_preserves_http_200_compatibility(client) -> None:
    response = client.post(
        '/api/v1/review/bulk',
        json={'action': 'reject', 'item_ids': []},
        headers={'X-Demo-User': 'hanvv-employee'},
    )

    assert response.status_code == 200
    assert response.json() == {
        'action': 'reject',
        'approved_count': 0,
        'rejected_count': 0,
        'failed_items': [],
        'skipped_items': [],
        'approved_item_ids': [],
        'rejected_item_ids': [],
        'replayed_items': [],
        'cost_policy': {
            'paid_llm_calls': False,
            'embedding_calls': False,
            'requires_human_review_state': True,
        },
    }


def test_unauthorized_human_bulk_reports_per_item_failure_with_http_200(
    client,
    db_session: Session,
) -> None:
    item = _add_review_item(db_session, permission_level='internal')

    response = client.post(
        '/api/v1/review/bulk',
        json={'action': 'reject', 'item_ids': [item.id]},
        headers={'X-Demo-User': 'hanvv-employee'},
    )

    assert response.status_code == 200
    assert response.json()['failed_items'] == [
        {'id': item.id, 'detail': 'Review approval permission required.'}
    ]
    assert response.json()['rejected_count'] == 0
    assert db_session.get(ReviewItem, item.id).status == 'pending_review'


def test_review_list_hides_items_above_user_permission(client, db_session: Session) -> None:
    visible_item = _add_review_item(db_session, permission_level='internal')
    hidden_item = _add_review_item(db_session, permission_level='restricted')

    response = client.get('/api/v1/review?status=pending_review', headers={'X-Demo-User': 'hanvv3@koreacu.ac.kr'})

    assert response.status_code == 200
    item_ids = {item['id'] for item in response.json()['items']}
    assert visible_item.id in item_ids
    assert hidden_item.id not in item_ids


def test_review_preview_hides_items_above_user_permission(client, db_session: Session) -> None:
    item = _add_review_item(db_session, permission_level='restricted')

    response = client.get(f'/api/v1/review/{item.id}/promotion-preview', headers={'X-Demo-User': 'hanvv3@koreacu.ac.kr'})

    assert response.status_code == 404


def test_reviewer_cannot_edit_restricted_review_item(client, db_session: Session) -> None:
    item = _add_review_item(db_session, permission_level='restricted')

    response = client.patch(
        f'/api/v1/review/{item.id}',
        json={'payload': {'title': 'Changed title', 'reason': 'Changed reason'}},
        headers={'X-Demo-User': 'mina@paraworks.com'},
    )

    assert response.status_code == 403
    assert response.json()['detail'] == 'Review approval permission required.'


def _add_review_item(db_session: Session, *, permission_level: str) -> ReviewItem:
    item = ReviewItem(
        item_type='history_event',
        payload={
            'title': 'Confirm project timeline',
            'reason': 'Slack and Gmail evidence agree on the project milestone.',
        },
        source_links=['https://slack.mock/team/123'],
        source_snippets=['The milestone date was confirmed in the project channel.'],
        confidence_score=0.88,
        permission_level=permission_level,
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)
    return item
