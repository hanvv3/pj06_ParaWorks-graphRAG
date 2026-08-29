from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.connectors.base import SourceEvent
from backend.app.ingestion.service import ingest_events
from backend.app.models import HistoryEvent, Project, ReviewItem, Source, TimelineEvent
from backend.app.projects.classifier import build_project_assignment_candidates


def _event(
    *,
    source_type: str,
    source_id: str,
    title: str,
    body: str,
    permission_level: str = 'internal',
    source_url: str | None = None,
) -> SourceEvent:
    raw_metadata: dict[str, object] = {
        'sync_partition': source_type,
        'sync_cursor': '2026-05-01T09:00:00Z',
    }
    semantic_timestamp_raw: str | None = None
    if source_type == 'gmail':
        message_id = source_id.removeprefix('gmail:')
        raw_metadata.update(
            message_id=message_id,
            thread_id=f'thread-{message_id}',
            content_signature=f'{source_id}:1777600800000',
        )
        semantic_timestamp_raw = '1777600800000'
    elif source_type == 'gmail_attachment':
        _, message_id, attachment_id = source_id.split(':', maxsplit=2)
        raw_metadata.update(
            parent_source_id=f'gmail:{message_id}',
            message_id=message_id,
            attachment_id=attachment_id,
            filename=title,
            mime_type='application/pdf',
            content_signature=f'{source_id}:2048',
        )
        semantic_timestamp_raw = '1777600800000'
    elif source_type == 'drive':
        raw_metadata.update(
            mime_type='text/plain',
            modified_time='2026-05-01T09:00:00Z',
            content_signature=f'{source_id}:2026-05-01T09:00:00Z',
        )
        semantic_timestamp_raw = '2026-05-01T09:00:00Z'
    elif source_type == 'calendar':
        raw_metadata.update(
            start='2026-05-01T09:00:00Z',
            end='2026-05-01T10:00:00Z',
            attendee_domains=['example.com'],
            event_status='confirmed',
            location=None,
            organizer_email='owner@example.com',
            content_signature=f'{source_id}:2026-05-01T09:00:00Z',
        )
        semantic_timestamp_raw = '2026-05-01T09:00:00Z'
    return SourceEvent(
        source_type=source_type,
        source_id=source_id,
        source_url=source_url or f'https://{source_type}.mock/{source_id}',
        title=title,
        body=body,
        author='owner@example.com',
        participants=['owner@example.com'],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level=permission_level,
        raw_metadata=raw_metadata,
        semantic_timestamp_raw=semantic_timestamp_raw,
    )


def test_project_classifier_finds_ktech_and_ir_but_excludes_company_rules(
    db_session: Session,
) -> None:
    db_session.add_all(
        [
            Project(
                project_key='k-tech-pilot',
                name='K테크 파일럿',
                summary='K테크 솔루션즈 파일럿과 온보딩 프로젝트',
            ),
            Project(
                project_key='seed-ir',
                name='시드 투자 IR',
                summary='Series Seed IR, VC 미팅, 피치덱 준비 프로젝트',
            ),
        ]
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='drive',
                source_id='drive:ktech-plan',
                title='02_파일럿_프로젝트/K테크 온보딩 계획',
                body='K테크 솔루션즈 파일럿은 3개월 일정으로 진행한다.',
            ),
            _event(
                source_type='gmail',
                source_id='gmail:ir-followup',
                title='Series Seed IR 후속 요청',
                body='VC 미팅 전 피치덱과 재무 프로젝션을 검토한다.',
            ),
            _event(
                source_type='drive',
                source_id='drive:company-rule',
                title='00_회사규정/휴가 정책',
                body='회사 공통 휴가 정책 문서입니다.',
            ),
        ],
    )

    candidates = build_project_assignment_candidates(db_session)

    assert {candidate.project_key for candidate in candidates} == {'k-tech-pilot', 'seed-ir'}
    assert all('00_회사규정' not in candidate.source_title for candidate in candidates)


def test_project_classifier_matches_bracketed_ir_gmail_subject(db_session: Session) -> None:
    db_session.add(
        Project(
            project_key='seed-ir',
            name='시드 투자 IR',
            summary='Series Seed IR, VC 미팅, 피치덱 준비 프로젝트',
        )
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='gmail',
                source_id='gmail:bracket-ir',
                title='[IR] A벤처스 미팅 결과 공유 및 액션 아이템',
                body='시드 투자 IR 후속 미팅 전까지 피치덱 수정안을 준비한다.',
            )
        ],
    )

    candidates = build_project_assignment_candidates(db_session)

    assert len(candidates) == 1
    assert candidates[0].project_key == 'seed-ir'
    assert candidates[0].source_type == 'gmail'


def test_project_classifier_does_not_match_phrase_inside_korean_word(
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-investment-fundraise',
            name='투자 유치',
            summary='투자자 미팅과 자금 유치 활동을 관리하는 프로젝트',
        )
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='slack',
                source_id='slack-kindergarten-dropoff',
                title='투자 유치원 등교 일정',
                body='오늘 유치원 등교 시간이 변경되었습니다.',
                source_url='https://slack.mock/archives/C123/p1',
            )
        ],
    )

    candidates = build_project_assignment_candidates(db_session)

    assert candidates == []


def test_project_classifier_excludes_slack_from_deterministic_assignments(
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-investment-fundraise',
            name='투자 유치',
            summary='투자자 미팅과 자금 유치 활동을 관리하는 프로젝트',
        )
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='slack',
                source_id='slack-fundraise-meeting',
                title='투자 유치 전략 회의',
                body='투자 유치 전략 회의 자료를 금요일까지 검토합니다.',
                source_url='https://slack.mock/archives/C123/p2',
            )
        ],
    )

    candidates = build_project_assignment_candidates(db_session)

    assert candidates == []


def test_project_classifier_ignores_generic_connector_terms_and_low_signal_slack_reply(
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-paraworks-mvp',
            name='Paraworks MVP',
            summary='Summarizes Slack, Gmail, and Google Drive data into a timeline.',
        )
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='slack',
                source_id='slack-good-good-reply',
                title='Slack thread reply in C123',
                body=(
                    'Thread parent: 공유폴더 하나 만들죠: '
                    '<https://drive.google.com/drive/folders/demo>\n'
                    'Thread reply: 굿굿'
                ),
                source_url='https://slack.mock/archives/C123/p3',
            )
        ],
    )

    candidates = build_project_assignment_candidates(db_session)

    assert candidates == []


def test_projects_reclassify_creates_pending_review_without_tokens(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='k-tech-pilot',
            name='K테크 파일럿',
            summary='K테크 파일럿 제안서와 온보딩 프로젝트',
        )
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='drive',
                source_id='drive:ktech-deadline',
                title='K테크 파일럿 제안서',
                body='K테크 파일럿 제안서는 금요일까지 업데이트해 주세요.',
                source_url='https://drive.mock/ktech/proposal',
            )
        ],
    )

    dry_run = client.post('/api/v1/projects/reclassify?dry_run=true', headers={'X-Demo-User': 'demo-admin'})
    execute = client.post('/api/v1/projects/reclassify?dry_run=false', headers={'X-Demo-User': 'demo-admin'})

    assert dry_run.status_code == 200
    assert dry_run.json()['candidate_count'] == 1
    assert dry_run.json()['created_review_items'] == 0
    assert dry_run.json()['cost_policy']['estimated_input_tokens'] == 0
    assert execute.status_code == 200
    assert execute.json()['created_review_items'] == 1
    item = db_session.query(ReviewItem).filter_by(item_type='project_assignment').one()
    assert item.status == 'pending_review'
    assert item.payload['project_key'] == 'k-tech-pilot'
    assert item.source_snippets


def test_calendar_project_assignment_summary_uses_event_title_not_raw_metadata(
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='demo-data',
            name='더미 데이터',
            summary='데모용 더미 데이터 일정',
        )
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='calendar',
                source_id='calendar:primary:bike-maintenance',
                title='자전거 정비 예약',
                body=(
                    '자전거 정비 예약\n\n'
                    'Description: <p>더미 데이터: 타이어 점검, 브레이크 조정, 체인 오일링. '
                    'Marker: DUMMY-DATA-FUTURE-14D-20</p>\n'
                    'Location: 동네 정비소\n'
                    'Start: 2026-05-16T14:00:00+09:00\n'
                    'End: 2026-05-16T15:00:00+09:00'
                ),
                source_url='https://calendar.google.com/event?eid=bike',
            )
        ],
    )

    [candidate] = build_project_assignment_candidates(db_session)

    assert candidate.task_summary == '자전거 정비 예약'
    assert 'Description:' not in candidate.task_summary
    assert '<p>' not in candidate.task_summary
    assert 'Start:' not in candidate.task_summary


def test_project_assignment_summary_strips_mail_and_drive_metadata(
    db_session: Session,
) -> None:
    db_session.add_all(
        [
            Project(project_key='paraworks', name='ParaWorks', summary='ParaWorks 소개서와 데모 운영'),
            Project(project_key='project-alpha', name='Project Alpha', summary='Project Alpha Redis work'),
        ]
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='gmail',
                source_id='gmail:paraworks-intro',
                title='ParaWorks 회사 소개서 전달드립니다',
                body=(
                    'ParaWorks 회사 소개서 전달드립니다\n\n'
                    'From: í•œìŠ¹í—Œ <hanvv3@gmail.com>\n'
                    'Date: Fri, 15 May 2026 01:58:19 -0700\n\n'
                    'ParaWorks 소개서를 공유합니다.'
                ),
                source_url='https://mail.google.com/mail/u/0/#inbox/gmail-paraworks-intro',
            ),
            _event(
                source_type='drive',
                source_id='drive:project-alpha-plan',
                title='Project Alpha rollout plan',
                body=(
                    'Google Drive file changed: Project Alpha rollout plan\n'
                    'Mime type: application/vnd.google-apps.document\n'
                    'Owner: owner@example.com\n'
                    'Modified: 2026-05-15T02:00:00Z'
                ),
                source_url='https://drive.google.com/file/d/project-alpha-plan/view',
            ),
            _event(
                source_type='gmail_attachment',
                source_id='gmail_attachment:paraworks-intro:paraworks-deck',
                title='ParaWorks 소개서.pdf',
                body=(
                    'Gmail attachment: ParaWorks 소개서.pdf\n'
                    'Parent subject: ParaWorks 회사 소개서 전달드립니다\n'
                    'Mime type: application/pdf\n'
                    'Attachment size: 2048'
                ),
                source_url='https://mail.google.com/mail/u/0/#inbox/gmail-paraworks-intro',
            ),
        ],
    )

    summaries = {
        candidate.source_id: candidate.task_summary
        for candidate in build_project_assignment_candidates(db_session)
    }

    assert summaries['gmail:paraworks-intro'] == 'ParaWorks 회사 소개서 전달드립니다'
    assert summaries['drive:project-alpha-plan'] == 'Project Alpha rollout plan'
    assert summaries['gmail_attachment:paraworks-intro:paraworks-deck'] == 'ParaWorks 소개서.pdf'
    for summary in summaries.values():
        assert 'From:' not in summary
        assert 'Date:' not in summary
        assert 'Mime type:' not in summary
        assert 'Parent subject:' not in summary
        assert 'Attachment size:' not in summary


def test_project_define_does_not_backfill_slack_project_assignments(
    client: TestClient,
    db_session: Session,
) -> None:
    ingest_events(
        db_session,
        [
            _event(
                source_type='slack',
                source_id='slack-project-alpha',
                title='Project Alpha Slack message',
                body='Project Alpha Redis sync 작업을 확인했습니다.',
                source_url='https://slack.mock/archives/C123/p1',
            )
        ],
    )

    response = client.post(
        '/api/v1/projects/define',
        json={'name': 'Project Alpha', 'summary': 'Slack Redis sync'},
        headers={'X-Demo-User': 'demo-admin'},
    )

    assert response.status_code == 200
    assert response.json()['created_review_items'] == 0
    assert db_session.query(ReviewItem).filter_by(item_type='project_assignment').count() == 0


def test_slack_tool_routed_approved_items_appear_in_project_activity_and_timeline(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(Project(project_key='project-alpha', name='Project Alpha', summary='Redis work'))
    item = ReviewItem(
        item_type='todo',
        payload={
            'title': 'Redis 큐 점검',
            'priority': 'high',
            'priority_reason': 'Redis 큐 상태를 확인해야 합니다.',
            'agent_name': 'slack_agent',
            'project_assignment_method': 'llm_tool',
            'project_key': 'project-alpha',
            'project_name': 'Project Alpha',
        },
        source_links=['https://example.slack.com/archives/C123/p1777600800000100'],
        source_snippets=['Redis 큐 상태 확인 필요'],
        confidence_score=0.9,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    approve = client.post(f'/api/v1/review/{item.id}/approve')
    projects = client.get('/api/v1/projects').json()['projects']
    project = next(project for project in projects if project['project_key'] == 'project-alpha')

    assert approve.status_code == 200
    assert any(activity['item_type'] == 'todo' for activity in project['activity_items'])
    assert any(timeline['item_type'] == 'timeline_event' for timeline in project['timeline_items'])


def test_created_project_is_visible_without_approved_evidence(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-empty-client-portal',
            name='고객 포털 개편',
            summary='고객이 계약 문서와 처리 상태를 확인하는 포털을 개편한다.',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    project = next(project for project in response.json()['projects'] if project['project_key'] == 'project-empty-client-portal')
    assert project['name'] == '고객 포털 개편'
    assert project['summary']
    assert project['evidence_count'] == 0
    assert project['timeline_items'] == []


def test_defined_projects_excludes_hardcoded_projects(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-client-portal',
            name='고객 포털 개편',
            summary='고객 포털 프로젝트',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects/defined', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    assert response.json()['projects'] == [
        {
            'project_key': 'project-client-portal',
            'name': '고객 포털 개편',
            'summary': '고객 포털 프로젝트',
        }
    ]


def test_define_project_then_projects_api_returns_empty_project(
    client: TestClient,
) -> None:
    create_response = client.post(
        '/api/v1/projects/define',
        headers={'X-Demo-User': 'demo-admin'},
        json={
            'name': '정산 자동화',
            'summary': '정산 파일 검토와 승인 흐름을 자동화하는 프로젝트',
        },
    )

    assert create_response.status_code == 200
    project_key = create_response.json()['project']['project_key']

    list_response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert list_response.status_code == 200
    project = next(project for project in list_response.json()['projects'] if project['project_key'] == project_key)
    assert project['name'] == '정산 자동화'
    assert project['evidence_count'] == 0
    assert project['timeline_items'] == []


def test_define_project_returns_readable_empty_summary(client: TestClient) -> None:
    create_response = client.post(
        '/api/v1/projects/define',
        headers={'X-Demo-User': 'demo-admin'},
        json={
            'name': '정산 자동화',
            'summary': '정산 파일 검토와 승인 흐름을 자동화하는 프로젝트',
        },
    )

    assert create_response.status_code == 200
    project_key = create_response.json()['project']['project_key']

    list_response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert list_response.status_code == 200
    project = next(project for project in list_response.json()['projects'] if project['project_key'] == project_key)
    assert project['summary'] == '정산 파일 검토와 승인 흐름을 자동화하는 프로젝트 아직 승인된 프로젝트 근거가 없습니다.'
    assert '?' not in project['summary']
    assert 'evidence' not in project['summary']


def test_define_project_creates_pending_assignment_candidates_from_existing_sources(
    client: TestClient,
    db_session: Session,
) -> None:
    ingest_events(
        db_session,
        [
            _event(
                source_type='drive',
                source_id='drive:settlement-automation',
                title='정산 자동화 일정',
                body='정산 자동화 프로젝트는 이번 주에 거래처 파일 검토 화면부터 진행합니다.',
                source_url='https://drive.mock/settlement/plan',
            )
        ],
    )

    create_response = client.post(
        '/api/v1/projects/define',
        headers={'X-Demo-User': 'demo-admin'},
        json={
            'name': '정산 자동화',
            'summary': '거래처 파일 검토와 승인 흐름을 자동화하는 프로젝트',
        },
    )

    assert create_response.status_code == 200
    assert create_response.json()['created_review_items'] == 1
    item = db_session.query(ReviewItem).filter_by(item_type='project_assignment').one()
    assert item.status == 'pending_review'
    assert item.payload['project_name'] == '정산 자동화'
    assert item.payload['project_key'] == create_response.json()['project']['project_key']


def test_project_classifier_uses_user_defined_projects(
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-client-portal',
            name='고객 포털 개편',
            summary='계약 문서와 진행 상태를 고객에게 보여주는 포털',
        )
    )
    ingest_events(
        db_session,
        [
            _event(
                source_type='drive',
                source_id='drive:client-portal',
                title='고객 포털 개편 일정',
                body='고객 포털 개편은 이번 주에 계약 문서 화면부터 진행합니다.',
                source_url='https://drive.mock/client-portal/plan',
            )
        ],
    )

    candidates = build_project_assignment_candidates(db_session)

    assert len(candidates) == 1
    assert candidates[0].project_key == 'project-client-portal'
    assert candidates[0].project_name == '고객 포털 개편'


def test_projects_api_returns_approved_project_evidence_only(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='k-tech-pilot',
            name='K테크 파일럿',
            summary='K테크 파일럿 프로젝트',
        )
    )
    db_session.add(
        ReviewItem(
            item_type='project_assignment',
            payload={
                'title': 'K테크 파일럿 source 연결',
                'summary': 'K테크 파일럿 제안서 업데이트',
                'project_key': 'k-tech-pilot',
                'project_name': 'K테크 파일럿',
                'source_id': 'slack-ktech-deadline',
                'source_type': 'slack',
                'source_title': 'Slack message in C0AUJDZUKA8',
                'task_summary': 'K테크 파일럿 제안서 업데이트',
                'evidence_reason': '"K테크" 단서가 발견되었습니다.',
                'timestamp': '2026-05-01T09:00:00Z',
            },
            source_links=['https://slack.mock/archives/C0AUJDZUKA8/p1'],
            source_snippets=['K테크 파일럿 제안서는 금요일까지 업데이트해 주세요.'],
            confidence_score=0.88,
            permission_level='internal',
            status='approved',
        )
    )
    db_session.add(
        ReviewItem(
            item_type='project_assignment',
            payload={
                'title': '프로젝트 결과 source 연결',
                'summary': '잘못된 더미 프로젝트 후보',
                'project_key': 'project-newbiegenie',
                'project_name': 'Project Newbiegenie',
                'source_id': 'dummy',
                'evidence_reason': 'legacy dummy',
            },
            source_links=['https://example.com/dummy'],
            source_snippets=['dummy'],
            confidence_score=0.1,
            permission_level='internal',
            status='approved',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    payload = response.json()
    assert payload['project_count'] == 1
    assert [project['project_key'] for project in payload['projects']] == ['k-tech-pilot']
    ktech = payload['projects'][0]
    assert ktech['evidence_count'] == 1
    assert ktech['evidence'][0]['title'] == 'K테크 파일럿 제안서 업데이트'
    assert ktech['evidence'][0]['evidence_reason']



def test_projects_api_links_timeline_items_to_approved_project_assignments(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='seed-ir',
            name='시드 투자 IR',
            summary='시드 투자 IR 프로젝트',
        )
    )
    db_session.add(
        ReviewItem(
            item_type='project_assignment',
            payload={
                'title': 'IR source 연결',
                'summary': 'IR 피치덱 검토',
                'project_key': 'seed-ir',
                'project_name': '시드 투자 IR',
                'source_id': 'drive-ir-deck',
                'source_type': 'drive',
                'source_title': '03_IR_투자/피치덱',
                'task_summary': 'IR 피치덱 검토',
                'evidence_reason': '"IR" 단서가 발견되었습니다.',
                'timestamp': '2026-05-01T09:00:00Z',
            },
            source_links=['https://drive.mock/drive-ir-deck'],
            source_snippets=['IR 피치덱 검토 일정은 다음 주입니다.'],
            confidence_score=0.88,
            permission_level='internal',
            status='approved',
        )
    )
    db_session.add(
        TimelineEvent(
            title='IR 피치덱 검토 완료',
            result_summary='투자자 미팅 전 피치덱 검토를 완료했다.',
            source_links=['https://drive.mock/drive-ir-deck'],
            source_snippets=['IR 피치덱 검토 일정은 다음 주입니다.'],
            confidence_score=0.9,
            permission_level='internal',
            review_status='approved',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    seed_ir = next(project for project in response.json()['projects'] if project['project_key'] == 'seed-ir')
    assert len(seed_ir['timeline_items']) == 1
    assert seed_ir['timeline_items'][0]['title'] == 'IR 피치덱 검토 완료'


def test_projects_api_links_approved_timeline_by_project_key_without_assignment(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='seed-ir',
            name='시드 투자 IR',
            summary='시드 투자 IR 프로젝트',
        )
    )
    db_session.add(
        TimelineEvent(
            project_key='seed-ir',
            title='IR pitch deck review completed',
            result_summary='Investor meeting pitch deck review was completed.',
            source_links=['https://drive.mock/drive-ir-deck'],
            source_snippets=['IR pitch deck review is scheduled for next week.'],
            confidence_score=0.9,
            permission_level='internal',
            review_status='approved',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    payload = response.json()
    seed_ir = next(project for project in payload['projects'] if project['project_key'] == 'seed-ir')
    assert seed_ir['evidence_count'] == 1
    assert seed_ir['evidence'][0]['source_url'] == 'https://drive.mock/drive-ir-deck'
    assert len(seed_ir['timeline_items']) == 1
    assert seed_ir['timeline_items'][0]['title'] == 'IR pitch deck review completed'
    assert seed_ir['timeline_items'][0]['project_key'] == 'seed-ir'


def test_project_timeline_items_use_slack_source_timestamp_for_occurred_at(
    client: TestClient,
    db_session: Session,
) -> None:
    project = Project(
        project_key='project-alpha',
        name='Project Alpha',
        summary='Slack source time test',
    )
    source = Source(
        source_type='slack',
        source_id='C123:1777600800.000100',
        source_url='https://example.slack.com/archives/C123/p1777600800000100',
        title='Slack 원문',
        author='김하나',
        permission_level='internal',
        raw_metadata={'ts': '1777600800.000100', 'channel_id': 'C123'},
    )
    timeline = TimelineEvent(
        project_key='project-alpha',
        title='실제 Slack 대화 기준 타임라인',
        result_summary='승인 시각이 아니라 Slack 대화 시각으로 보여야 합니다.',
        source_links=['https://example.slack.com/archives/C123/p1777600800000100'],
        source_snippets=['Slack에서 오전에 논의했습니다.'],
        confidence_score=0.9,
        permission_level='internal',
        review_status='approved',
        created_at=datetime(2026, 5, 15, 9, 0, tzinfo=UTC),
    )
    db_session.add_all([project, source, timeline])
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    project_payload = next(project for project in response.json()['projects'] if project['project_key'] == 'project-alpha')
    item = project_payload['timeline_items'][0]
    assert item['created_at'].startswith('2026-05-15T09:00:00')
    assert item['occurred_at'].startswith('2026-05-01T02:00:00')


def test_project_timeline_prefers_slack_permalink_timestamp_when_source_metadata_is_missing(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(Project(project_key='project-alpha', name='Project Alpha', summary='Slack permalink fallback'))
    source_url = 'https://example.slack.com/archives/C123/p1777600800000100'
    db_session.add(
        Source(
            source_type='slack',
            source_id='C123:1777600800.000100',
            source_url=source_url,
            title='Slack 원문',
            author='김하나',
            permission_level='internal',
            raw_metadata={},
            created_at=datetime(2026, 5, 15, 9, 0, tzinfo=UTC),
        )
    )
    db_session.add(
        TimelineEvent(
            project_key='project-alpha',
            title='Slack permalink 기준 타임라인',
            result_summary='Source metadata가 부족해도 permalink 시간을 사용해야 합니다.',
            source_links=[source_url],
            source_snippets=['Slack permalink에 실제 원문 시간이 남아 있습니다.'],
            confidence_score=0.9,
            permission_level='internal',
            review_status='approved',
            created_at=datetime(2026, 5, 15, 9, 30, tzinfo=UTC),
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    project_payload = next(project for project in response.json()['projects'] if project['project_key'] == 'project-alpha')
    item = project_payload['timeline_items'][0]
    assert item['created_at'].startswith('2026-05-15T09:30:00')
    assert item['occurred_at'].startswith('2026-05-01T02:00:00')


def test_project_timeline_items_use_calendar_event_start_for_occurred_at(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(Project(project_key='project-alpha', name='Project Alpha', summary='Calendar time test'))
    source_url = 'https://calendar.google.com/event?eid=event-alpha'
    db_session.add(
        Source(
            source_type='calendar',
            source_id='calendar:team@example.com:event-alpha',
            source_url=source_url,
            title='Project Alpha milestone meeting',
            author='lead@example.com',
            permission_level='internal',
            raw_metadata={
                'calendar_id': 'team@example.com',
                'calendar_summary': 'Team Calendar',
                'event_start': '2026-06-10T10:00:00+09:00',
                'event_end': '2026-06-10T11:00:00+09:00',
            },
            created_at=datetime(2026, 5, 15, 9, 0, tzinfo=UTC),
        )
    )
    db_session.add(
        TimelineEvent(
            project_key='project-alpha',
            title='Project Alpha milestone confirmed',
            result_summary='Calendar evidence confirmed the launch milestone meeting.',
            source_links=[source_url],
            source_snippets=['Project Alpha launch milestone meeting'],
            confidence_score=0.9,
            permission_level='internal',
            review_status='approved',
            created_at=datetime(2026, 5, 15, 9, 30, tzinfo=UTC),
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    project_payload = next(project for project in response.json()['projects'] if project['project_key'] == 'project-alpha')
    item = project_payload['timeline_items'][0]
    assert item['created_at'].startswith('2026-05-15T09:30:00')
    assert item['occurred_at'].startswith('2026-06-10T01:00:00')


def test_approved_review_item_with_project_key_appears_in_project_timeline(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='k-tech-pilot',
            name='K-Tech pilot',
            summary='K-Tech pilot project',
        )
    )
    review_item = ReviewItem(
        item_type='timeline_event',
        payload={
            'title': 'K-Tech proposal deadline confirmed',
            'result_summary': 'K-Tech pilot proposal deadline was confirmed for Friday.',
            'project_key': 'k-tech-pilot',
        },
        source_links=['https://slack.mock/archives/C0AUJDZUKA8/p1'],
        source_snippets=['Please update the K-Tech pilot proposal by Friday.'],
        confidence_score=0.88,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(review_item)
    db_session.commit()
    db_session.refresh(review_item)

    approve_response = client.post(
        f'/api/v1/review/{review_item.id}/approve',
        headers={'X-Demo-User': 'demo-admin'},
    )
    assert approve_response.status_code == 200

    projects_response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert projects_response.status_code == 200
    ktech = next(project for project in projects_response.json()['projects'] if project['project_key'] == 'k-tech-pilot')
    assert [item['title'] for item in ktech['timeline_items']] == ['K-Tech proposal deadline confirmed']
    assert ktech['timeline_items'][0]['project_key'] == 'k-tech-pilot'


def test_projects_api_builds_evidence_from_approved_project_activity_sources(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-alpha',
            name='Project Alpha',
            summary='Approved activity evidence test',
        )
    )
    history = HistoryEvent(
        project_key='project-alpha',
        title='Redis 큐 상태 확인',
        reason='Slack에서 Redis 큐 상태를 확인했습니다.',
        source_links=['https://example.slack.com/archives/C123/p1777600800000100'],
        source_snippets=['Redis 큐 상태 확인 완료'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
        created_at=datetime(2026, 5, 15, 9, 0, tzinfo=UTC),
    )
    db_session.add(history)
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    project_payload = next(project for project in response.json()['projects'] if project['project_key'] == 'project-alpha')
    assert project_payload['evidence_count'] == 1
    assert project_payload['source_types'] == ['slack']
    assert project_payload['evidence'][0]['title'] == 'Redis 큐 상태 확인'
    assert project_payload['evidence'][0]['source_url'] == 'https://example.slack.com/archives/C123/p1777600800000100'
    assert project_payload['evidence'][0]['source_snippet'] == 'Redis 큐 상태 확인 완료'


def test_approved_slack_llm_project_routing_item_appears_in_project_timeline(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        Project(
            project_key='project-alpha',
            name='Project Alpha',
            summary='Redis queue status and sync job reliability work',
        )
    )
    review_item = ReviewItem(
        item_type='history_event',
        payload={
            'title': 'Redis 큐 상태 확인',
            'summary': 'Redis 큐와 동기화 작업 상태를 확인했습니다.',
            'project_key': 'project-alpha',
            'project_name': 'Project Alpha',
            'project_assignment_method': 'llm_tool',
            'project_assignment_summary': 'Redis 큐 상태와 동기화 안정성 개선 논의입니다.',
            'project_assignment_reason': 'Redis와 sync job 근거가 Project Alpha와 일치합니다.',
        },
        source_links=['https://example.slack.com/archives/C123/p1777600800000100'],
        source_snippets=['Redis queue 상태를 확인하고 sync job을 복구합니다.'],
        confidence_score=0.91,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(review_item)
    db_session.commit()
    db_session.refresh(review_item)

    approve_response = client.post(
        f'/api/v1/review/{review_item.id}/approve',
        headers={'X-Demo-User': 'demo-admin'},
    )
    assert approve_response.status_code == 200

    projects_response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert projects_response.status_code == 200
    project_payload = next(
        project
        for project in projects_response.json()['projects']
        if project['project_key'] == 'project-alpha'
    )
    assert any(
        timeline_item['title'] == 'Redis 큐 상태 확인'
        for timeline_item in project_payload['timeline_items']
    )

def test_pending_mail_document_project_routed_item_counts_on_project_card(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(Project(project_key='project-alpha', name='Project Alpha', summary='Gmail and Drive work'))
    db_session.add(
        ReviewItem(
            item_type='todo',
            payload={
                'title': 'Reply with Drive plan',
                'priority': 'high',
                'priority_reason': 'Customer requested the plan.',
                'agent_name': 'mail_document_agent',
                'project_assignment_method': 'llm_tool',
                'project_key': 'project-alpha',
                'project_name': 'Project Alpha',
                'source_ids': ['gmail:message-1', 'drive:file-1'],
            },
            source_links=['https://mail.google.com/mail/u/0/#inbox/message-1', 'https://drive.mock/file-1'],
            source_snippets=['Customer requested the plan.', 'Drive plan evidence.'],
            confidence_score=0.86,
            permission_level='internal',
            status='pending_review',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    project = next(project for project in response.json()['projects'] if project['project_key'] == 'project-alpha')
    assert project['pending_review_count'] == 1


def test_projects_api_hides_unregistered_approved_project_keys(
    client: TestClient,
    db_session: Session,
) -> None:
    db_session.add(
        ReviewItem(
            item_type='project_assignment',
            payload={
                'title': 'Unregistered project source',
                'summary': 'This project was never registered.',
                'project_key': 'project-unregistered',
                'project_name': 'Unregistered',
                'source_id': 'slack-unregistered',
                'source_type': 'slack',
                'source_title': 'Slack message in C123',
                'task_summary': 'Unregistered evidence',
                'evidence_reason': 'legacy dummy',
                'timestamp': '2026-05-01T09:00:00Z',
            },
            source_links=['https://example.com/unregistered'],
            source_snippets=['unregistered'],
            confidence_score=0.88,
            permission_level='internal',
            status='approved',
        )
    )
    db_session.commit()

    response = client.get('/api/v1/projects', headers={'X-Demo-User': 'demo-admin'})

    assert response.status_code == 200
    assert response.json()['projects'] == []
