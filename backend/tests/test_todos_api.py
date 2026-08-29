from backend.app.models import Project, TimelineEvent, Todo


def test_complete_todo_persists_completion_and_hides_from_dashboard(client, db_session) -> None:
    todo = Todo(
        project_key='project-alpha',
        title='고객사 공유본 발송',
        assignee='김하나',
        due_date='2099-01-01',
        priority='high',
        priority_reason='고객사에게 오늘 공유본을 보내야 합니다.',
        source_links=['https://slack.mock/project-alpha/p123'],
        source_snippets=['오늘 공유본을 보내주세요.'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    db_session.add(todo)
    db_session.commit()

    response = client.post(f'/api/v1/todos/{todo.id}/complete', headers={'X-Demo-User': 'yonghee199702'})

    assert response.status_code == 200
    body = response.json()
    assert body['id'] == todo.id
    assert body['status'] == 'completed'
    assert body['completed_by'] == 'yonghee199702'
    assert body['completed_at']

    db_session.refresh(todo)
    assert todo.completed_at is not None
    assert todo.completed_by == 'yonghee199702'

    dashboard = client.get('/api/v1/dashboard').json()
    assert dashboard['today_todos'] == []


def test_completed_todo_is_visible_as_completed_project_activity(client, db_session) -> None:
    project = Project(
        project_key='project-alpha',
        name='Project Alpha',
        summary='고객사 공유본 프로젝트입니다.',
    )
    todo = Todo(
        project_key='project-alpha',
        title='고객사 공유본 발송',
        assignee='김하나',
        due_date='2099-01-01',
        priority='high',
        priority_reason='고객사에게 오늘 공유본을 보내야 합니다.',
        source_links=['https://slack.mock/project-alpha/p123'],
        source_snippets=['오늘 공유본을 보내주세요.'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    db_session.add_all([project, todo])
    db_session.commit()

    client.post(f'/api/v1/todos/{todo.id}/complete', headers={'X-Demo-User': 'yonghee199702'})

    response = client.get('/api/v1/projects')

    assert response.status_code == 200
    project_payload = response.json()['projects'][0]
    todo_items = [item for item in project_payload['activity_items'] if item['item_type'] == 'todo']
    assert todo_items[0]['title'] == '고객사 공유본 발송'
    assert todo_items[0]['completed_at']
    assert todo_items[0]['completed_by'] == 'yonghee199702'


def test_completed_todo_updates_existing_timeline_item_without_adding_todo_timeline(client, db_session) -> None:
    project = Project(
        project_key='project-alpha',
        name='Project Alpha',
        summary='고객사 공유본 프로젝트입니다.',
    )
    source_links = ['https://slack.mock/project-alpha/p123']
    todo = Todo(
        project_key='project-alpha',
        title='고객사 공유본 발송',
        assignee='김하나',
        due_date='2099-01-01',
        priority='high',
        priority_reason='고객사에게 오늘 공유본을 보내야 합니다.',
        source_links=source_links,
        source_snippets=['오늘 공유본을 보내주세요.'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    timeline = TimelineEvent(
        project_key='project-alpha',
        title='[할 일] 고객사 공유본 발송',
        result_summary='담당자: 김하나, 기한: 2099-01-01',
        source_links=source_links,
        source_snippets=['오늘 공유본을 보내주세요.'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    db_session.add_all([project, todo, timeline])
    db_session.commit()

    client.post(f'/api/v1/todos/{todo.id}/complete', headers={'X-Demo-User': 'yonghee199702'})

    response = client.get('/api/v1/projects')

    assert response.status_code == 200
    timeline_items = response.json()['projects'][0]['timeline_items']
    assert timeline_items == [
        {
            'id': f'timeline_event:{timeline.id}',
            'item_type': 'timeline_event',
            'title': '[할 일] 고객사 공유본 발송',
            'summary': '담당자: 김하나, 기한: 2099-01-01',
            'source_links': source_links,
            'source_snippets': ['오늘 공유본을 보내주세요.'],
            'confidence_score': 0.91,
            'permission_level': 'internal',
            'review_status': 'approved',
            'created_at': timeline_items[0]['created_at'],
            'occurred_at': timeline_items[0]['occurred_at'],
            'evidence_reason': '승인된 타임라인 항목이 이 프로젝트와 연결되어 있습니다.',
            'project_key': 'project-alpha',
            'completed_at': timeline_items[0]['completed_at'],
            'completed_by': 'yonghee199702',
        }
    ]
    assert timeline_items[0]['completed_at']


def test_complete_todo_rejects_inaccessible_permission(client, db_session) -> None:
    todo = Todo(
        title='제한된 할 일',
        assignee='김하나',
        due_date='2099-01-01',
        priority='high',
        priority_reason='restricted evidence에서 나온 할 일입니다.',
        source_links=['https://slack.mock/restricted/p123'],
        source_snippets=['restricted task'],
        confidence_score=0.91,
        permission_level='restricted',
        review_status='approved',
    )
    db_session.add(todo)
    db_session.commit()

    response = client.post(f'/api/v1/todos/{todo.id}/complete', headers={'X-Demo-User': 'viewer'})

    assert response.status_code == 404
    db_session.refresh(todo)
    assert todo.completed_at is None
    assert todo.completed_by is None
