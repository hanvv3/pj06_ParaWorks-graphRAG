from sqlalchemy import select

from backend.app.models import (
    DecisionRecord,
    HistoryEvent,
    ReviewItem,
    TimelineEvent,
    Todo,
)


def seed_review_item(db_session, *, item_type: str, payload: dict) -> ReviewItem:
    item = ReviewItem(
        item_type=item_type,
        payload=payload,
        source_links=['https://slack.mock/source-1'],
        source_snippets=['source snippet'],
        confidence_score=0.87,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)
    return item


def test_approve_decision_record_promotes_to_knowledge_table(client, db_session) -> None:
    item = seed_review_item(
        db_session,
        item_type='decision_record',
        payload={
            'title': 'Use Redis for queues',
            'decision_summary': 'Redis should power queue and job progress updates.',
        },
    )

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 200
    decision = db_session.scalars(select(DecisionRecord)).one()
    assert decision.title == 'Use Redis for queues'
    assert decision.decision_summary == 'Redis should power queue and job progress updates.'
    assert decision.source_links == ['https://slack.mock/source-1']
    assert decision.source_snippets == ['source snippet']
    assert decision.confidence_score == 0.87
    assert decision.permission_level == 'internal'
    assert decision.review_status == 'approved'


def test_approve_history_event_promotes_to_knowledge_table(client, db_session) -> None:
    item = seed_review_item(
        db_session,
        item_type='history_event',
        payload={
            'title': 'Project Beta scope changed',
            'reason': 'Advanced diff UI moved out of MVP.',
        },
    )

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 200
    history_event = db_session.scalars(select(HistoryEvent)).one()
    assert history_event.title == 'Project Beta scope changed'
    assert history_event.reason == 'Advanced diff UI moved out of MVP.'
    assert history_event.source_links == ['https://slack.mock/source-1']
    assert history_event.source_snippets == ['source snippet']
    assert history_event.review_status == 'approved'


def test_approve_todo_promotes_to_knowledge_table(client, db_session) -> None:
    item = seed_review_item(
        db_session,
        item_type='todo',
        payload={
            'title': 'Verify evidence inspection before launch',
            'priority': 'high',
            'priority_reason': 'Evidence must be checked before launch readiness review.',
            'assignee': '김하나',
            'due_date': '2026-05-18',
        },
    )

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 200
    todo = db_session.scalars(select(Todo)).one()
    assert todo.title == 'Verify evidence inspection before launch'
    assert todo.priority == 'high'
    assert todo.priority_reason == 'Evidence must be checked before launch readiness review.'
    assert todo.assignee == '김하나'
    assert todo.due_date == '2026-05-18'
    assert todo.source_links == ['https://slack.mock/source-1']
    assert todo.source_snippets == ['source snippet']
    assert todo.review_status == 'approved'


def test_approve_mail_document_todo_returns_promotion_next_routes(client, db_session) -> None:
    item = seed_review_item(
        db_session,
        item_type='todo',
        payload={
            'title': 'K테크 1개월 파일럿 제안 검토 및 회신',
            'task_summary': '1개월 파일럿 제안의 범위와 성공 기준을 검토합니다.',
            'recommended_next_step': '파일럿 범위, 성공 지표, 일정 초안을 정리해 회신합니다.',
            'assignee': '용희',
            'due_date': '2026-05-15',
            'project_key': 'k-tech-pilot',
        },
    )

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'approved'
    assert body['promotion_result']['target_type'] == 'todo'
    assert body['promotion_result']['project_key'] == 'k-tech-pilot'
    assert body['promotion_result']['next_routes'] == ['/projects', '/timeline']
    todo = db_session.scalars(select(Todo)).one()
    timeline = db_session.scalars(select(TimelineEvent)).one()
    assert body['promotion_result']['created_record_ids'] == [todo.id]
    assert body['promotion_result']['created_timeline_event_ids'] == [timeline.id]
    assert todo.assignee == '용희'
    assert todo.due_date == '2026-05-15'
    assert todo.priority_reason == '파일럿 범위, 성공 지표, 일정 초안을 정리해 회신합니다.'
    assert timeline.title == '[할 일] K테크 1개월 파일럿 제안 검토 및 회신'
    assert timeline.result_summary == '담당자: 용희, 기한: 2026-05-15'


def test_approve_timeline_event_promotes_to_timeline_table(client, db_session) -> None:
    item = seed_review_item(
        db_session,
        item_type='timeline_event',
        payload={
            'title': 'Customer launch date confirmed',
            'result_summary': 'Slack evidence confirmed the customer launch date and owner.',
        },
    )

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 200
    timeline_event = db_session.scalars(select(TimelineEvent)).one()
    assert timeline_event.title == 'Customer launch date confirmed'
    assert timeline_event.result_summary == 'Slack evidence confirmed the customer launch date and owner.'
    assert timeline_event.source_links == ['https://slack.mock/source-1']
    assert timeline_event.source_snippets == ['source snippet']
    assert timeline_event.review_status == 'approved'


def test_bulk_approve_agent_candidates_promotes_only_agent_items(client, db_session) -> None:
    agent_history = seed_review_item(
        db_session,
        item_type='history_event',
        payload={
            'title': 'Redis queue decision captured',
            'summary': 'Slack evidence indicates Redis should support queue progress.',
            'agent_name': 'slack_agent',
        },
    )
    agent_decision = seed_review_item(
        db_session,
        item_type='decision_record',
        payload={
            'title': 'PostgreSQL remains durable store',
            'decision_summary': 'Mail evidence keeps PostgreSQL as source of record.',
            'agent_name': 'mail_document_agent',
        },
    )
    manual_item = seed_review_item(
        db_session,
        item_type='todo',
        payload={
            'title': 'Manual follow-up',
            'priority': 'medium',
            'priority_reason': 'This item was written by a person.',
        },
    )

    response = client.post('/api/v1/review/approve-agent-candidates')

    assert response.status_code == 200
    payload = response.json()
    assert payload['approved_count'] == 2
    assert payload['skipped_count'] == 1
    assert payload['approved_item_ids'] == [agent_history.id, agent_decision.id]
    assert payload['cost_policy'] == {
        'paid_llm_calls': False,
        'embedding_calls': False,
        'requires_human_review_state': True,
    }

    db_session.refresh(agent_history)
    db_session.refresh(agent_decision)
    db_session.refresh(manual_item)
    assert agent_history.status == 'approved'
    assert agent_decision.status == 'approved'
    assert manual_item.status == 'pending_review'

    assert db_session.scalars(select(HistoryEvent)).one().title == 'Redis queue decision captured'
    assert db_session.scalars(select(DecisionRecord)).one().title == 'PostgreSQL remains durable store'


def test_single_approve_replay_returns_canonical_promotion_ids(client, db_session) -> None:
    item = seed_review_item(
        db_session,
        item_type='decision_record',
        payload={
            'title': 'Use one transition boundary',
            'decision_summary': 'Every approval shares exactly-once promotion.',
        },
    )

    first = client.post(f'/api/v1/review/{item.id}/approve')
    replay = client.post(f'/api/v1/review/{item.id}/approve')

    assert first.status_code == 200
    assert replay.status_code == 200
    assert first.json()['replayed'] is False
    assert replay.json()['replayed'] is True
    assert replay.json()['promotion'] == first.json()['promotion']
    assert replay.json()['promotion_result'] == first.json()['promotion_result']
    assert len(db_session.scalars(select(DecisionRecord)).all()) == 1
    assert len(db_session.scalars(select(TimelineEvent)).all()) == 1


def test_approve_agent_candidates_replays_canonical_effects(client, db_session) -> None:
    item = seed_review_item(
        db_session,
        item_type='history_event',
        payload={
            'title': 'Candidate replay is idempotent',
            'reason': 'Response loss must not duplicate company memory.',
            'agent_name': 'mail_document_agent',
        },
    )

    first = client.post('/api/v1/review/approve-agent-candidates')
    replay = client.post('/api/v1/review/approve-agent-candidates')

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json()['approved_count'] == 1
    assert replay.json()['approved_item_ids'] == [item.id]
    history = db_session.scalars(select(HistoryEvent)).one()
    timeline = db_session.scalars(select(TimelineEvent)).one()
    assert replay.json()['replayed_items'] == [
        {
            'item_id': item.id,
            'status': 'approved',
            'replayed': True,
            'promotion': {
                'target_type': 'history_event',
                'created_record_ids': [history.id],
                'created_timeline_event_ids': [timeline.id],
            },
        }
    ]
    assert len(db_session.scalars(select(HistoryEvent)).all()) == 1
    assert len(db_session.scalars(select(TimelineEvent)).all()) == 1
