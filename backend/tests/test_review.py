from sqlalchemy import select

from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    AuditLog,
    Document,
    DocumentChunk,
    DocumentVersion,
    Project,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
)
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
)


def _seed_filtered_review_workflow(
    db_session,
    thread_id: str,
    *,
    security_scope_id: str = 'default',
    permission_level: str = 'internal',
    statuses: tuple[str, ...] = ('pending_review',),
) -> AgentWorkflowThread:
    source = Source(
        source_type='gmail',
        source_id=f'gmail:{thread_id}',
        source_url=f'https://mail.example.test/{thread_id}',
        title=f'Workflow source {thread_id}',
        permission_level=permission_level,
        raw_metadata={'content_signature': f'signature-{thread_id}'},
    )
    db_session.add(source)
    db_session.flush()
    thread = AgentWorkflowThread(
        thread_id=thread_id,
        workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
        graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        checkpoint_thread_id=f'review-v2:{thread_id}',
        checkpoint_store='memory',
        owner_subject_id='demo-admin',
        security_scope_id=security_scope_id,
        input_hash='a' * 64,
        evidence_version_hash='b' * 64,
        status='awaiting_human_review',
    )
    db_session.add(thread)
    db_session.add(
        AgentWorkflowEvidenceRef(
            workflow_thread_id=thread_id,
            ordinal=0,
            canonical_source_type='gmail',
            canonical_table='sources',
            canonical_row_id=source.id,
            document_version_id=None,
            external_revision=None,
            content_signature=source.raw_metadata['content_signature'],
            permission_level_snapshot=permission_level,
            content_fingerprint='c' * 64,
        )
    )
    for index, status in enumerate(statuses):
        db_session.add(
            ReviewItem(
                item_type='history_event',
                payload={'title': f'{thread_id} candidate {index + 1}'},
                source_links=[source.source_url],
                source_snippets=[f'{thread_id} snippet {index + 1}'],
                confidence_score=0.8 + (index / 100),
                permission_level=permission_level,
                status=status,
                workflow_thread_id=thread_id,
                candidate_key=f'{thread_id}-candidate-{index + 1}',
            )
        )
    db_session.commit()
    return thread


def test_approve_review_item_changes_status(client) -> None:
    client.post('/api/v1/integrations/slack/sync')
    item = client.get('/api/v1/review?status=pending_review').json()['items'][0]
    response = client.post(f"/api/v1/review/{item['id']}/approve")
    assert response.status_code == 200
    assert response.json()['status'] == 'approved'


def test_visible_workflow_filter_returns_only_bound_items(client, db_session) -> None:
    visible = _seed_filtered_review_workflow(
        db_session,
        'workflow-visible-filter',
        statuses=('pending_review', 'pending_review'),
    )
    db_session.add(
        ReviewItem(
            item_type='todo',
            payload={'title': 'Unrelated queue item'},
            source_links=['https://mail.example.test/unrelated'],
            source_snippets=['Unrelated visible review item'],
            confidence_score=0.7,
            permission_level='internal',
            status='pending_review',
        )
    )
    db_session.commit()

    response = client.get(
        '/api/v1/review',
        params={
            'status': 'pending_review',
            'workflow_thread_id': visible.thread_id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['total_count'] == 2
    assert len(body['items']) == 2
    assert all(
        item['payload']['title'].startswith('workflow-visible-filter')
        for item in body['items']
    )


def test_hidden_missing_and_zero_visible_workflow_filters_are_identical_empty_results(
    client,
    db_session,
) -> None:
    foreign = _seed_filtered_review_workflow(
        db_session,
        'workflow-foreign-filter',
        security_scope_id='another-security-scope',
    )
    hidden = _seed_filtered_review_workflow(
        db_session,
        'workflow-hidden-filter',
        permission_level='restricted',
    )
    zero = _seed_filtered_review_workflow(
        db_session,
        'workflow-zero-filter',
        statuses=(),
    )

    responses = [
        client.get(
            '/api/v1/review',
            params={'workflow_thread_id': foreign.thread_id},
        ),
        client.get(
            '/api/v1/review',
            params={'workflow_thread_id': hidden.thread_id},
            headers={'X-Demo-User': 'viewer'},
        ),
        client.get(
            '/api/v1/review',
            params={'workflow_thread_id': 'workflow-missing-filter'},
        ),
        client.get(
            '/api/v1/review',
            params={'workflow_thread_id': zero.thread_id},
        ),
    ]

    projections = [
        {
            'items': response.json()['items'],
            'groups': response.json()['groups'],
            'total_count': response.json()['total_count'],
        }
        for response in responses
    ]
    assert all(response.status_code == 200 for response in responses)
    assert projections == [
        {'items': [], 'groups': [], 'total_count': 0},
    ] * 4


def test_workflow_filter_runs_before_pagination_grouping_and_respects_status(
    client,
    db_session,
) -> None:
    workflow = _seed_filtered_review_workflow(
        db_session,
        'workflow-order-filter',
        statuses=('pending_review', 'approved', 'approved'),
    )
    for index in range(3):
        db_session.add(
            ReviewItem(
                item_type='history_event',
                payload={'title': f'Unrelated approved {index}'},
                source_links=[f'https://mail.example.test/unrelated-{index}'],
                source_snippets=[f'Unrelated {index}'],
                confidence_score=0.7,
                permission_level='internal',
                status='approved',
            )
        )
    db_session.commit()

    response = client.get(
        '/api/v1/review',
        params={
            'workflow_thread_id': workflow.thread_id,
            'status': 'approved',
            'limit': 1,
            'offset': 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['total_count'] == 2
    assert len(body['items']) == 1
    assert body['has_more'] is False
    assert sum(group['total_count'] for group in body['groups']) == 1
    assert body['items'][0]['status'] == 'approved'
    assert body['items'][0]['payload']['title'].startswith('workflow-order-filter')


def test_reject_review_item_changes_status(client) -> None:
    client.post('/api/v1/integrations/slack/sync')
    item = client.get('/api/v1/review?status=pending_review').json()['items'][0]
    response = client.post(f"/api/v1/review/{item['id']}/reject")
    assert response.status_code == 200
    assert response.json()['status'] == 'rejected'


def test_reject_mail_document_review_item_preserves_source_evidence(client, db_session) -> None:
    source = Source(
        source_type='gmail',
        source_id='gmail:reject-preserve',
        source_url='https://gmail.mock/reject-preserve',
        title='Reject preserve source',
        author='owner@example.com',
        permission_level='internal',
        raw_metadata={},
    )
    db_session.add(source)
    db_session.flush()
    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version='v1', body='Rejecting the AI candidate must not delete this source.')
    db_session.add(version)
    db_session.flush()
    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        chunk_index=0,
        text='Rejecting the AI candidate must not delete this source.',
        source_snippet='Rejecting the AI candidate must not delete this source.',
        permission_level='internal',
        metadata_={'source_type': 'gmail'},
    )
    item = ReviewItem(
        item_type='history_event',
        payload={
            'title': 'Reject preserve candidate',
            'summary': 'The AI candidate is not trusted yet.',
            'source_ids': ['gmail:reject-preserve'],
        },
        source_links=[source.source_url],
        source_snippets=[chunk.source_snippet],
        confidence_score=0.72,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add_all([chunk, item])
    db_session.commit()
    db_session.refresh(item)

    response = client.post(f'/api/v1/review/{item.id}/reject')

    assert response.status_code == 200
    assert response.json()['status'] == 'rejected'
    assert db_session.scalar(select(Source).where(Source.source_id == 'gmail:reject-preserve')) is not None
    assert db_session.scalar(select(DocumentChunk).where(DocumentChunk.source_id == source.id)) is not None
    audit_log = db_session.scalar(select(AuditLog).where(AuditLog.action == 'review.reject'))
    assert audit_log is not None
    assert audit_log.metadata_ == {
        'actor_type': 'human',
        'review_item_id': item.id,
        'outcome': 'rejected',
        'item_type': 'history_event',
    }
    assert 'gmail:reject-preserve' not in str(audit_log.metadata_)


def test_patch_review_item_updates_payload(client) -> None:
    client.post('/api/v1/integrations/slack/sync')
    item = client.get('/api/v1/review?status=pending_review').json()['items'][0]

    response = client.patch(f"/api/v1/review/{item['id']}", json={'payload': {'title': 'Updated title'}})

    assert response.status_code == 200
    body = response.json()
    assert body['payload']['title'] == 'Updated title'
    assert body['status'] == 'pending_review'


def test_terminal_workflow_bound_review_item_cannot_be_patched(client, db_session) -> None:
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Bound candidate', 'reason': 'The workflow owns its terminal resolution.'},
        source_links=['https://mail.mock/thread/bound'],
        source_snippets=['The candidate has already been resolved.'],
        confidence_score=0.88,
        permission_level='internal',
        status='approved',
        workflow_thread_id='workflow-bound-patch',
    )
    db_session.add(item)
    db_session.commit()

    response = client.patch(f'/api/v1/review/{item.id}', json={'payload': {'title': 'Rewritten'}})

    assert response.status_code == 409
    assert response.json()['detail'] == {'code': 'invalid_state_transition'}
    assert db_session.get(ReviewItem, item.id).payload['title'] == 'Bound candidate'


def test_terminal_legacy_unbound_review_item_remains_patch_compatible(client, db_session) -> None:
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Legacy candidate', 'reason': 'Legacy editing remains supported.'},
        source_links=['https://mail.mock/thread/legacy'],
        source_snippets=['The old row is not workflow-bound.'],
        confidence_score=0.88,
        permission_level='internal',
        status='approved',
    )
    db_session.add(item)
    db_session.commit()

    response = client.patch(f'/api/v1/review/{item.id}', json={'payload': {'title': 'Corrected legacy title'}})

    assert response.status_code == 200
    assert response.json()['payload']['title'] == 'Corrected legacy title'


def test_patch_review_item_requires_registered_project_key(client, db_session) -> None:
    db_session.add(
        Project(
            project_key='project-client-portal',
            name='고객 포털 개편',
            summary='고객 포털 개편 프로젝트',
        )
    )
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Project candidate', 'summary': 'Needs a project.'},
        source_links=['https://slack.mock/team/123'],
        source_snippets=['고객 포털 화면 검토가 필요합니다.'],
        confidence_score=0.8,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    missing = client.patch(f"/api/v1/review/{item.id}", json={'payload': {'project_key': 'project-missing'}})
    valid = client.patch(f"/api/v1/review/{item.id}", json={'payload': {'project_key': 'project-client-portal'}})

    assert missing.status_code == 400
    assert missing.json()['detail'] == 'Project key is not registered'
    assert valid.status_code == 200
    assert valid.json()['payload']['project_key'] == 'project-client-portal'
    assert valid.json()['payload']['project_name'] == '고객 포털 개편'


def test_project_assignment_group_title_sanitizes_calendar_metadata(client, db_session) -> None:
    item = ReviewItem(
        item_type='project_assignment',
        payload={
            'agent_name': 'project_classifier',
            'title': '케크 source 연결',
            'summary': (
                '자전거 정비 예약 Description: <p>더미 데이터: 타이어 점검, 브레이크 조정. '
                'Marker: DUMMY-DATA-FUTURE-14D-20</p> Location: 동네 정비소 '
                'Start: 2026-05-16T14:00:00+09:00 End: 2026-05-16T15:00:00+09:00'
            ),
            'task_summary': (
                '자전거 정비 예약 Description: <p>더미 데이터: 타이어 점검, 브레이크 조정. '
                'Marker: DUMMY-DATA-FUTURE-14D-20</p> Location: 동네 정비소 '
                'Start: 2026-05-16T14:00:00+09:00 End: 2026-05-16T15:00:00+09:00'
            ),
            'source_title': '자전거 정비 예약',
            'source_type': 'calendar',
        },
        source_links=['https://calendar.google.com/event?eid=bike'],
        source_snippets=['자전거 정비 예약'],
        confidence_score=0.88,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()

    response = client.get('/api/v1/review?status=pending_review')

    assert response.status_code == 200
    group = response.json()['groups'][0]
    assert group['title'] == '자전거 정비 예약'
    assert 'Description:' not in group['title']
    assert '<p>' not in group['title']


def test_project_assignment_group_title_sanitizes_mail_and_drive_metadata(client, db_session) -> None:
    items = [
        ReviewItem(
            item_type='project_assignment',
            payload={
                'agent_name': 'project_classifier',
                'title': 'ParaWorks source 연결',
                'summary': (
                    'ParaWorks 회사 소개서 전달드립니다 From: í•œìŠ¹í—Œ Date: Fri, 15 May 2026 01:58:19 -0700 '
                    'ParaWorks 소개서를 공유합니다.'
                ),
                'task_summary': (
                    'ParaWorks 회사 소개서 전달드립니다 From: í•œìŠ¹í—Œ Date: Fri, 15 May 2026 01:58:19 -0700 '
                    'ParaWorks 소개서를 공유합니다.'
                ),
                'source_title': 'ParaWorks 회사 소개서 전달드립니다',
                'source_type': 'gmail',
            },
            source_links=['https://mail.google.com/mail/u/0/#inbox/gmail-paraworks-intro'],
            source_snippets=['ParaWorks 회사 소개서 전달드립니다'],
            confidence_score=0.88,
            permission_level='internal',
            status='pending_review',
        ),
        ReviewItem(
            item_type='project_assignment',
            payload={
                'agent_name': 'project_classifier',
                'title': 'Project Alpha source 연결',
                'summary': (
                    'Google Drive file changed: Project Alpha rollout plan Mime type: application/pdf '
                    'Owner: owner@example.com Modified: 2026-05-15T02:00:00Z'
                ),
                'task_summary': (
                    'Google Drive file changed: Project Alpha rollout plan Mime type: application/pdf '
                    'Owner: owner@example.com Modified: 2026-05-15T02:00:00Z'
                ),
                'source_title': 'Project Alpha rollout plan',
                'source_type': 'drive',
            },
            source_links=['https://drive.google.com/file/d/project-alpha-plan/view'],
            source_snippets=['Project Alpha rollout plan'],
            confidence_score=0.88,
            permission_level='internal',
            status='pending_review',
        ),
    ]
    db_session.add_all(items)
    db_session.commit()

    response = client.get('/api/v1/review?status=pending_review')

    assert response.status_code == 200
    titles = {group['title'] for group in response.json()['groups']}
    assert 'ParaWorks 회사 소개서 전달드립니다' in titles
    assert 'Project Alpha rollout plan' in titles
    assert all('From:' not in title for title in titles)
    assert all('Date:' not in title for title in titles)
    assert all('Mime type:' not in title for title in titles)
    assert all('Google Drive file changed:' not in title for title in titles)


def test_slack_agent_review_item_requires_project_before_approval(client, db_session) -> None:
    db_session.add(Project(project_key='project-alpha', name='Project Alpha', summary='Redis work'))
    item = ReviewItem(
        item_type='history_event',
        payload={
            'title': '등록 프로젝트 확인 필요',
            'summary': 'Slack Agent가 업무 후보로 판단했지만 프로젝트 선택이 필요합니다.',
            'agent_name': 'slack_agent',
            'project_assignment_method': 'llm_tool',
            'project_needs_user_selection': True,
        },
        source_links=['https://example.slack.com/archives/C123/p1777600800000100'],
        source_snippets=['새 프로젝트로 보이는 업무 논의입니다.'],
        confidence_score=0.82,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    preview = client.get(f'/api/v1/review/{item.id}/promotion-preview')
    blocked = client.post(f'/api/v1/review/{item.id}/approve')
    response = client.patch(f"/api/v1/review/{item.id}", json={'payload': {'project_key': 'project-alpha'}})
    approved = client.post(f'/api/v1/review/{item.id}/approve')

    assert preview.status_code == 200
    assert preview.json()['can_approve'] is False
    assert 'project_key' in preview.json()['missing_required_fields']
    assert blocked.status_code == 400
    assert response.status_code == 200
    body = response.json()
    assert body['payload']['project_key'] == 'project-alpha'
    assert body['payload']['project_name'] == 'Project Alpha'
    assert body['payload']['project_needs_user_selection'] is False
    assert approved.status_code == 200
    assert approved.json()['promotion_result']['project_key'] == 'project-alpha'


def test_review_list_supports_limit_offset_metadata(client, db_session) -> None:
    for index in range(3):
        db_session.add(
            ReviewItem(
                item_type='history_event',
                payload={'title': f'Paged item {index}', 'summary': 'Pagination target.'},
                source_links=[f'https://slack.mock/team/{index}'],
                source_snippets=[f'Snippet {index}'],
                confidence_score=0.7,
                permission_level='internal',
                status='pending_review',
            )
        )
    db_session.commit()

    response = client.get('/api/v1/review?status=pending_review&limit=1&offset=1')

    assert response.status_code == 200
    body = response.json()
    assert body['total_count'] == 3
    assert body['limit'] == 1
    assert body['offset'] == 1
    assert body['has_more'] is True
    assert len(body['items']) == 1


def test_review_list_prioritizes_knowledge_candidates_before_project_assignments(
    client,
    db_session,
) -> None:
    knowledge_item = ReviewItem(
        item_type='decision_record',
        payload={
            'title': 'Slack decision candidate',
            'decision_summary': 'Slack Agent extracted this decision.',
        },
        source_links=['https://slack.mock/archives/C123/p1'],
        source_snippets=['Redis queue policy was decided.'],
        confidence_score=0.91,
        permission_level='internal',
        status='pending_review',
    )
    project_assignment = ReviewItem(
        item_type='project_assignment',
        payload={
            'title': 'Project source assignment',
            'summary': 'Project classifier linked this source.',
            'agent_name': 'project_classifier',
        },
        source_links=['https://slack.mock/archives/C123/p2'],
        source_snippets=['Project evidence snippet.'],
        confidence_score=0.88,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add_all([knowledge_item, project_assignment])
    db_session.commit()

    response = client.get('/api/v1/review?status=pending_review&limit=2')

    assert response.status_code == 200
    assert [item['item_type'] for item in response.json()['items']] == [
        'decision_record',
        'project_assignment',
    ]


def test_review_list_uses_display_title_when_payload_title_is_low_signal(client, db_session) -> None:
    item = ReviewItem(
        item_type='decision_record',
        payload={'title': 'ParaWorks source 연결', 'summary': '실제 검토 큐 표시 제목'},
        source_links=['https://slack.mock/archives/C123/p1'],
        source_snippets=['Evidence snippet.'],
        confidence_score=0.91,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()

    response = client.get('/api/v1/review?status=pending_review&limit=1')

    assert response.status_code == 200
    body = response.json()
    assert body['groups'][0]['title'] == '실제 검토 큐 표시 제목'
    assert body['groups'][0]['group_id'] == 'decision_record:실제 검토 큐 표시 제목'


def test_request_more_evidence_changes_status(client) -> None:
    client.post('/api/v1/integrations/slack/sync')
    item = client.get('/api/v1/review?status=pending_review').json()['items'][0]

    response = client.post(f"/api/v1/review/{item['id']}/request-more-evidence")

    assert response.status_code == 200
    assert response.json()['status'] == 'needs_more_evidence'


def test_bulk_reject_pending_review_items(client, db_session) -> None:
    items = [
        ReviewItem(
            item_type='history_event',
            payload={'title': f'Reject {index}', 'summary': 'Reject in bulk.'},
            source_links=[f'https://slack.mock/reject/{index}'],
            source_snippets=[f'Reject snippet {index}'],
            confidence_score=0.7,
            permission_level='internal',
            status='pending_review',
        )
        for index in range(3)
    ]
    db_session.add_all(items)
    db_session.commit()
    item_ids = [item.id for item in items]

    response = client.post('/api/v1/review/bulk', json={'action': 'reject', 'item_ids': item_ids})

    assert response.status_code == 200
    assert response.json()['rejected_count'] == 3
    assert response.json()['failed_items'] == []
    statuses = db_session.scalars(select(ReviewItem.status).where(ReviewItem.id.in_(item_ids))).all()
    assert statuses == ['rejected', 'rejected', 'rejected']


def test_bulk_approve_reports_items_that_cannot_be_promoted(client, db_session) -> None:
    valid = ReviewItem(
        item_type='todo',
        payload={
            'title': '고객사 공유본 준비',
            'priority': 'high',
            'priority_reason': '금요일까지 고객사 공유본 준비가 필요합니다.',
        },
        source_links=['https://drive.mock/project-alpha/plan'],
        source_snippets=['고객사 공유본을 준비해주세요.'],
        confidence_score=0.88,
        permission_level='internal',
        status='pending_review',
    )
    invalid = ReviewItem(
        item_type='decision_record',
        payload={'title': 'Use Redis'},
        source_links=['https://slack.mock/team/456'],
        source_snippets=['Redis decision needs a summary.'],
        confidence_score=0.9,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add_all([valid, invalid])
    db_session.commit()

    response = client.post('/api/v1/review/bulk', json={'action': 'approve', 'item_ids': [valid.id, invalid.id]})

    assert response.status_code == 200
    body = response.json()
    assert body['approved_count'] == 1
    assert body['failed_items'] == [{'id': invalid.id, 'detail': 'Review item is missing required fields'}]
    assert db_session.get(ReviewItem, valid.id).status == 'approved'
    assert db_session.get(ReviewItem, invalid.id).status == 'pending_review'


def test_bulk_approve_replays_single_approval_without_duplicate_effects(client, db_session) -> None:
    item = ReviewItem(
        item_type='timeline_event',
        payload={
            'title': 'Promotion replay verified',
            'result_summary': 'Single and bulk approval return the same canonical effect.',
        },
        source_links=['https://drive.mock/replay'],
        source_snippets=['The exact timeline event is supported by Drive evidence.'],
        confidence_score=0.9,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()

    single = client.post(f'/api/v1/review/{item.id}/approve')
    bulk = client.post('/api/v1/review/bulk', json={'action': 'approve', 'item_ids': [item.id]})

    assert single.status_code == 200
    assert bulk.status_code == 200
    assert bulk.json()['approved_count'] == 1
    assert bulk.json()['approved_item_ids'] == [item.id]
    assert bulk.json()['replayed_items'] == [
        {
            'item_id': item.id,
            'status': 'approved',
            'replayed': True,
            'promotion': single.json()['promotion'],
        }
    ]
    assert len(db_session.scalars(select(TimelineEvent)).all()) == 1


def test_terminal_route_action_returns_typed_conflict(client, db_session) -> None:
    item = ReviewItem(
        item_type='timeline_event',
        payload={'title': 'Resolved item', 'result_summary': 'Already rejected.'},
        source_links=['https://drive.mock/resolved'],
        source_snippets=['The prior review was terminal.'],
        confidence_score=0.8,
        permission_level='internal',
        status='rejected',
        workflow_thread_id='workflow-terminal-route',
    )
    db_session.add(item)
    db_session.commit()

    response = client.post(f'/api/v1/review/{item.id}/request-more-evidence')

    assert response.status_code == 409
    assert response.json()['detail'] == {'code': 'invalid_state_transition'}


def test_workflow_bound_review_audit_is_bounded_and_contains_no_source_ids(client, db_session) -> None:
    item = ReviewItem(
        item_type='history_event',
        payload={
            'title': 'Sensitive workflow candidate',
            'reason': 'Bound audit data is minimized.',
            'source_ids': ['gmail:secret-message-id'],
        },
        source_links=['https://mail.google.com/mail/u/0/#inbox/secret-message-id'],
        source_snippets=['Sensitive source content must not enter AuditLog.'],
        confidence_score=0.84,
        permission_level='internal',
        status='pending_review',
        workflow_thread_id='workflow-audit-safe',
    )
    db_session.add(item)
    db_session.commit()

    response = client.post(f'/api/v1/review/{item.id}/reject')

    assert response.status_code == 200
    audit = db_session.scalar(select(AuditLog).where(AuditLog.action == 'review.reject'))
    assert audit.target_type == 'review_workflow'
    assert audit.target_id == 'workflow-audit-safe'
    assert audit.metadata_ == {
        'actor_type': 'human',
        'review_item_id': item.id,
        'outcome': 'rejected',
        'replayed': False,
        'effect_count': 0,
    }
    serialized = str(audit.metadata_)
    assert 'secret-message-id' not in serialized
    assert 'Sensitive source content' not in serialized


def test_request_more_evidence_preserves_reviewer_note(client, db_session) -> None:
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Need more context', 'summary': 'Missing source detail.'},
        source_links=['https://slack.mock/team/789'],
        source_snippets=['We need to confirm this with the owner.'],
        confidence_score=0.62,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    response = client.post(
        f'/api/v1/review/{item.id}/request-more-evidence',
        json={'note': '담당자 발언과 결정 근거를 하나 더 찾아주세요.'},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'needs_more_evidence'
    assert body['payload']['needs_more_evidence']['note'] == '담당자 발언과 결정 근거를 하나 더 찾아주세요.'
    assert body['payload']['needs_more_evidence']['requested_by'] == 'demo-admin'
    assert 'source_count' not in body['payload']['needs_more_evidence']
    assert body['evidence_status'] == 'evidence_unavailable'
    assert body['source_links'] == []
    assert body['source_snippets'] == []


def test_review_item_preview_returns_promotion_shape(client, db_session) -> None:
    item = ReviewItem(
        item_type='todo',
        payload={
            'title': 'Follow up on Redis rollout',
            'priority': 'high',
            'priority_reason': 'The queue migration needs owner confirmation.',
        },
        source_links=['https://slack.mock/team/123'],
        source_snippets=['Redis rollout needs a clear owner.'],
        confidence_score=0.87,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    response = client.get(f'/api/v1/review/{item.id}/promotion-preview')

    assert response.status_code == 200
    payload = response.json()
    assert payload['can_approve'] is True
    assert payload['target_type'] == 'todo'
    assert payload['missing_required_fields'] == []
    assert payload['normalized_payload'] == {
        'title': 'Follow up on Redis rollout',
        'priority': 'high',
        'priority_reason': 'The queue migration needs owner confirmation.',
    }


def test_review_item_response_includes_structured_source_evidence(client, db_session) -> None:
    agent_run = AgentRun(
        agent_name='slack_agent',
        prompt_version='slack-summary:v1',
        status='complete',
        source_window='slack:live:ranked:2',
        cache_key='cache-source-evidence',
        model_name='fake-model',
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.00001,
        permission_level='internal',
        metadata_={
            'evidence_summary': [
                {
                    'rank': 1,
                    'source_id': 'slack:C123:1710000000.000100',
                    'source_url': 'https://slack.mock/archives/C123/p1710000000000100',
                    'source_type': 'slack',
                    'timestamp': '1710000000.000100',
                    'author': 'U123',
                    'permission_level': 'internal',
                    'importance_score': 95,
                    'parser_status': 'parsed',
                    'section_path': '결정 사항',
                    'evidence_reason': 'Redis rollout owner를 직접 언급합니다.',
                    'snippet': 'Redis rollout decision needs owner confirmation.',
                }
            ]
        },
    )
    db_session.add(agent_run)
    db_session.flush()
    item = ReviewItem(
        item_type='decision_record',
        payload={
            'title': 'Confirm Redis rollout owner',
            'decision_summary': 'Redis rollout needs a named owner.',
            'agent_name': 'slack_agent',
            'agent_run_id': agent_run.id,
        },
        source_links=['https://slack.mock/archives/C123/p1710000000000100'],
        source_snippets=['Redis rollout decision needs owner confirmation.'],
        confidence_score=0.91,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()

    response = client.get('/api/v1/review?status=pending_review')

    assert response.status_code == 200
    body = response.json()['items'][0]
    assert body['evidence_status'] == 'evidence_unavailable'
    assert body['agent_run_id'] is None
    assert body['source_evidence'] == []
    assert body['source_links'] == []
    assert body['source_snippets'] == []


def test_mail_document_review_source_evidence_uses_indexed_source_type_fallback(client, db_session) -> None:
    item = ReviewItem(
        item_type='history_event',
        payload={
            'title': 'Mail docs calendar source labels',
            'summary': 'Review evidence should keep each Google source family visible.',
            'agent_name': 'mail_document_agent',
            'source_ids': ['gmail:message-1', 'drive:file-1', 'calendar:primary:event-1'],
            'source_types': ['gmail', 'drive', 'calendar'],
        },
        source_links=[],
        source_snippets=['Mail thread evidence.', 'Drive document evidence.', 'Calendar event evidence.'],
        confidence_score=0.84,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()

    response = client.get('/api/v1/review?status=pending_review')

    assert response.status_code == 200
    body = response.json()['items'][0]
    assert body['evidence_status'] == 'evidence_unavailable'
    assert body['source_evidence'] == []
    assert body['source_links'] == []
    assert body['source_snippets'] == []


def test_approve_review_item_rejects_missing_required_fields(client, db_session) -> None:
    item = ReviewItem(
        item_type='decision_record',
        payload={'title': 'Use Redis'},
        source_links=['https://slack.mock/team/456'],
        source_snippets=['Redis decision needs a summary.'],
        confidence_score=0.9,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 400
    assert response.json()['detail'] == 'Review item is missing required fields'


def test_approve_mail_document_llm_tool_item_requires_project_selection(client, db_session) -> None:
    item = ReviewItem(
        item_type='todo',
        payload={
            'title': 'Reply to customer with Drive plan',
            'priority': 'high',
            'priority_reason': 'The customer asked for the reviewed plan.',
            'agent_name': 'mail_document_agent',
            'project_assignment_method': 'llm_tool',
            'project_assignment_summary': 'No registered project was selected automatically.',
            'project_assignment_reason': 'The Gmail and Drive evidence did not clearly match a project.',
            'project_needs_user_selection': True,
            'source_ids': ['gmail:message-1', 'drive:file-1'],
        },
        source_links=['https://mail.google.com/mail/u/0/#inbox/message-1', 'https://drive.mock/file-1'],
        source_snippets=['Customer asked for the reviewed plan.', 'The Drive plan needs confirmation.'],
        confidence_score=0.82,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    preview = client.get(f'/api/v1/review/{item.id}/promotion-preview')
    approve = client.post(f'/api/v1/review/{item.id}/approve')

    assert preview.status_code == 200
    assert preview.json()['can_approve'] is False
    assert preview.json()['missing_required_fields'] == ['project_key']
    assert approve.status_code == 400
    assert approve.json()['detail'] == '프로젝트를 선택해야 승인할 수 있습니다.'


def test_approve_todo_promotes_clean_korean_timeline_without_mojibake(client, db_session) -> None:
    item = ReviewItem(
        item_type='todo',
        payload={
            'title': '고객사 공유본 준비',
            'priority': 'high',
            'priority_reason': '금요일까지 고객사 공유본 준비가 필요합니다.',
            'assignee': '김하나',
            'due_date': '2026-05-15',
        },
        source_links=['https://drive.mock/project-alpha/plan'],
        source_snippets=['김하나님은 금요일까지 고객사 공유본을 준비해주세요.'],
        confidence_score=0.88,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 200
    todo = db_session.query(Todo).one()
    timeline = db_session.query(TimelineEvent).one()
    assert todo.title == '고객사 공유본 준비'
    assert timeline.title == '[할 일] 고객사 공유본 준비'
    assert timeline.result_summary == '담당자: 김하나, 기한: 2026-05-15'
    assert '?' not in timeline.title
    assert '?' not in timeline.result_summary
