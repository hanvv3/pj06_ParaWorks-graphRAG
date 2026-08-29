import logging
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.api.v1 import assistant as assistant_api
from backend.app.assistant.service import (
    append_assistant_message,
    append_user_message,
    build_contextual_question,
    create_conversation,
    list_messages,
    update_summary,
)
from backend.app.core.demo_auth import USERS
from backend.app.models import AgentRun, AssistantMessage
from backend.tests.test_rag_orchestrator_service import seed_chunk


def _email_intent(
    *,
    email_intent: bool,
    confidence_score: float = 1.0,
    requires_rag_result: bool = False,
    intent_type: str = 'send',
) -> object:
    return assistant_api.EmailIntentDecision(
        email_intent=email_intent,
        intent_type=intent_type if email_intent else 'none',
        confidence_score=confidence_score,
        requires_rag_result=requires_rag_result,
        model_name='gpt-4.1-nano',
    )


def _patch_email_flow(monkeypatch, *, intent_decision, draft_decision=None) -> None:
    class FakeEmailIntentGate:
        def decide(self, **kwargs):
            return intent_decision

    class FakeEmailDraftComposer:
        def compose(self, **kwargs):
            if callable(draft_decision):
                return draft_decision(**kwargs)
            return draft_decision

    monkeypatch.setattr(assistant_api, 'build_email_intent_gate', lambda settings: FakeEmailIntentGate())
    monkeypatch.setattr(assistant_api, 'build_email_draft_composer', lambda settings: FakeEmailDraftComposer())


def test_assistant_conversation_api_is_user_scoped(client: TestClient) -> None:
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Viewer Redis 질문'},
        headers={'X-Demo-User': 'viewer'},
    )
    assert create_response.status_code == 200
    conversation_id = create_response.json()['conversation']['id']

    other_user_response = client.get(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        headers={'X-Demo-User': 'hanvv-employee'},
    )

    assert other_user_response.status_code == 404
    assert other_user_response.json()['detail'] == 'assistant conversation not found'


def test_assistant_message_flow_stores_rag_answer_without_cost_fields(
    client: TestClient,
    db_session: Session,
) -> None:
    seed_chunk(
        db_session,
        'gmail',
        'gmail-redis-assistant',
        'Redis should be used for transient job state while PostgreSQL stores durable records.',
        'internal',
    )
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Redis 질문'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'Redis job state'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    payload = turn_response.json()
    assert payload['user_message']['role'] == 'user'
    assert payload['assistant_message']['role'] == 'assistant'
    assert payload['assistant_message']['source_links'] == ['https://gmail.mock/gmail-redis-assistant']
    assert payload['assistant_message']['hidden_match_count'] == 0
    assert payload['assistant_message']['agent_run_id'] == db_session.query(AgentRun).one().id
    assert 'estimated_cost_usd' not in payload['assistant_message']
    assert 'token_usage' not in payload['assistant_message']
    assert 'cache_key' not in payload['assistant_message']


def test_assistant_tool_middleware_logs_email_and_rag_tools_in_english(
    client: TestClient,
    caplog,
    monkeypatch,
) -> None:
    caplog.set_level(logging.INFO, logger='AssistantTool')

    def fake_rag_answer(**kwargs):
        tool_logger = kwargs['tool_logger']
        tool_logger.log('rag_retrieval', 'result backend=keyword source_count=1 hidden_count=0')
        tool_logger.log('rag_answer', 'start model=gpt-5.4')
        tool_logger.log('rag_answer', 'result model=gpt-5.4 source_count=1')
        return SimpleNamespace(
            answer='RAG answer',
            citations=[],
            source_ids=['source-1'],
            source_links=['https://source.example/1'],
            source_snippets=['source snippet'],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=123,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v1',
            question=kwargs['question'],
        )

    def override_settings():
        return assistant_api.Settings(
            paraworks_demo_mode=False,
            openai_api_key='test-key',
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=False, confidence_score=0.2, intent_type='none'),
    )
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fake_rag_answer)
    client.app.dependency_overrides[assistant_api.get_settings] = override_settings

    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Tool log'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'Redis job state'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    projected = turn_response.json()['assistant_message']
    assert projected['source_links'] == []
    assert projected['metadata']['status'] == 'evidence_unavailable'
    log_text = caplog.text
    assert '[Tool: email_intent_gate] start' in log_text
    assert '[Tool: email_intent_gate] result email_intent=False confidence=0.2 requires_rag_result=False model=gpt-4.1-nano' in log_text
    assert '[Tool: rag_retrieval] result backend=keyword source_count=1 hidden_count=0' in log_text
    assert '[Tool: rag_answer] start model=gpt-5.4' in log_text
    assert '[Tool: rag_answer] result model=gpt-5.4 source_count=1' in log_text


def test_assistant_context_deduplicates_repeated_assistant_answers(
    db_session: Session,
) -> None:
    user = USERS['viewer']
    conversation = create_conversation(db_session, user, title='반복 답변')
    repeated_answer = 'Redis는 일시적인 작업 상태와 큐 진행 상황을 빠르게 공유하는 데 사용됩니다.'
    conversation.summary = update_summary(None, repeated_answer)
    for _ in range(4):
        append_assistant_message(
            db_session,
            user,
            conversation,
            content=repeated_answer,
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={},
        )

    messages = list_messages(db_session, user, conversation.id)
    contextual_question = build_contextual_question(
        conversation=conversation,
        messages=messages,
        new_message='K테크 일정을 알려줘',
    )

    assert contextual_question.count(repeated_answer) == 1


def test_assistant_email_request_creates_approval_draft_without_rag(
    client: TestClient,
    monkeypatch,
) -> None:
    class FakeEmailActionAgent:
        def decide(self, **kwargs):
            return assistant_api.EmailActionDecision(
                action_type='email_draft',
                to=['partner@example.com'],
                subject='회의 취소 안내',
                body='안녕하세요.\n\n오늘 회의가 취소되었습니다.\n\n감사합니다.',
            )

    def fail_if_rag_runs(**kwargs):
        raise AssertionError('메일 작성 요청은 RAG 근거 확인 없이 액션 초안으로 라우팅되어야 합니다.')

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=True),
        draft_decision=FakeEmailActionAgent().decide,
    )
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_rag_runs)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '메일 작성'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'partner@example.com에 오늘 회의 취소됐다고 메일 보내줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['role'] == 'assistant'
    assert '메일 초안을 작성했습니다' in assistant_message['content']
    assert '회의 취소 안내' in assistant_message['content']
    assert assistant_message['citations'] == []
    assert assistant_message['metadata']['action_type'] == 'email_draft'
    assert assistant_message['metadata']['status'] == 'pending_approval'
    assert assistant_message['metadata']['email_draft']['to'] == ['partner@example.com']
    assert assistant_message['metadata']['email_draft']['subject'] == '회의 취소 안내'


def test_assistant_low_confidence_email_decision_falls_back_to_rag(
    client: TestClient,
    monkeypatch,
) -> None:
    class LowConfidenceEmailActionAgent:
        def decide(self, **kwargs):
            return assistant_api.EmailActionDecision(
                action_type='email_draft',
                to=['partner@example.com'],
                subject='모호한 요청',
                body='모호한 요청입니다.',
                confidence_score=0.4,
            )

    def fake_rag_answer(**kwargs):
        return SimpleNamespace(
            answer='안녕하세요. 무엇을 도와드릴까요?',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=321,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v1',
            question=kwargs['question'],
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=True, confidence_score=0.4),
        draft_decision=LowConfidenceEmailActionAgent().decide,
    )
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fake_rag_answer)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '일반 대화'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': '간단히 인사해줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['content'] == '안녕하세요. 무엇을 도와드릴까요?'
    assert assistant_message['metadata']['agent_name'] == 'rag_orchestrator_agent'
    assert assistant_message['metadata'].get('action_type') != 'email_draft'


def test_assistant_non_email_intent_goes_to_rag(
    client: TestClient,
    monkeypatch,
) -> None:
    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=False, confidence_score=0.91, intent_type='none'),
    )
    monkeypatch.setattr(
        assistant_api,
        'answer_question_with_rag',
        lambda **kwargs: SimpleNamespace(
            answer='RAG answer',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=654,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v1',
            question=kwargs['question'],
        ),
    )
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '일반 대화'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': '간단히 인사해줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['content'] == 'RAG answer'
    assert assistant_message['metadata']['agent_name'] == 'rag_orchestrator_agent'
    assert assistant_message['metadata'].get('action_type') != 'email_draft'


def test_assistant_email_agent_uses_conversation_context_without_rag(
    client: TestClient,
    monkeypatch,
) -> None:
    class FakeEmailActionAgent:
        def decide(self, **kwargs):
            if kwargs['latest_message'] != '그 분에게 오늘 회의 취소됐다고 메일 보내줘':
                return assistant_api.EmailActionDecision(action_type='not_email')
            assert kwargs['latest_message'] == '그 분에게 오늘 회의 취소됐다고 메일 보내줘'
            assert 'partner@example.com' in kwargs['conversation_context']
            return assistant_api.EmailActionDecision(
                action_type='email_draft',
                to=['partner@example.com'],
                subject='회의 취소 안내',
                body='안녕하세요.\n\n오늘 회의가 취소되었습니다.\n\n감사합니다.',
            )

    def fail_if_rag_runs(**kwargs):
        raise AssertionError('메일 의도를 sub-agent가 판단하면 RAG로 가지 않아야 합니다.')

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=True),
        draft_decision=FakeEmailActionAgent().decide,
    )
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '메일 맥락'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']
    client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'partner@example.com은 협력사 담당자입니다.'},
        headers={'X-Demo-User': 'viewer'},
    )
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_rag_runs)

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': '그 분에게 오늘 회의 취소됐다고 메일 보내줘'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['metadata']['action_type'] == 'email_draft'
    assert assistant_message['metadata']['prompt_version'] == assistant_api.EMAIL_ACTION_PROMPT_VERSION
    assert assistant_message['metadata']['email_draft']['to'] == ['partner@example.com']
    assert assistant_message['metadata']['email_draft']['subject'] == '회의 취소 안내'


def test_assistant_can_draft_email_from_rag_answer(
    client: TestClient,
    monkeypatch,
) -> None:
    def compose_from_rag(**kwargs):
        assert kwargs['intent'].requires_rag_result is True
        assert 'Project Alpha launch is Friday.' in kwargs['rag_context']
        assert 'https://source.example/project-alpha' in kwargs['rag_context']
        return assistant_api.EmailActionDecision(
            action_type='email_draft',
            to=['lead@example.com'],
            subject='Project Alpha launch summary',
            body='Project Alpha launch is Friday.',
        )

    def fake_rag_answer(**kwargs):
        return SimpleNamespace(
            answer='Project Alpha launch is Friday.',
            citations=[],
            source_ids=['source-1'],
            source_links=['https://source.example/project-alpha'],
            source_snippets=['Launch decision snippet'],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=987,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v1',
            question=kwargs['question'],
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=True, requires_rag_result=True),
        draft_decision=compose_from_rag,
    )
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fake_rag_answer)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Email RAG result'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'Find the Project Alpha launch date and email it to lead@example.com.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['metadata']['action_type'] == 'email_draft'
    assert assistant_message['metadata']['email_draft']['to'] == ['lead@example.com']
    assert assistant_message['metadata']['email_draft']['body'] == 'Project Alpha launch is Friday.'


def test_assistant_generates_requested_content_before_email_draft(
    client: TestClient,
    monkeypatch,
) -> None:
    generated_intro = (
        'ParaWorks 회사 소개\n'
        'ParaWorks는 흩어진 업무 대화와 문서를 통합해 조직의 기억을 구축하는 AI 플랫폼입니다.\n'
        'Slack, Gmail, Google Drive 데이터를 연결하고 Review Queue로 공식 지식을 승인합니다.'
    )

    def fail_if_email_gate_runs(*args, **kwargs):
        raise AssertionError('작성해서 보내는 요청은 email_intent_gate에 맡기기 전에 RAG 산출물을 먼저 생성해야 합니다.')

    def fake_rag_answer(**kwargs):
        assert 'ParaWorks 회사 소개서 작성' in kwargs['question']
        assert '메일' not in kwargs['question']
        return SimpleNamespace(
            answer=generated_intro,
            citations=[],
            source_ids=['source-1'],
            source_links=['https://source.example/paraworks-intro'],
            source_snippets=['ParaWorks intro source'],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=456,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v1',
            question=kwargs['question'],
        )

    def compose_from_generated_source(**kwargs):
        assert generated_intro in kwargs['rag_context']
        assert kwargs['resolved_recipients'][0]['email'] == 'yonghee199702@gmail.com'
        return assistant_api.EmailActionDecision(
            action_type='email_draft',
            to=['yonghee199702@gmail.com'],
            subject='ParaWorks 회사 소개서 공유드립니다',
            body='회사 소개서 초안을 공유드립니다.',
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=False),
        draft_decision=compose_from_generated_source,
    )
    monkeypatch.setattr(assistant_api, 'build_email_intent_gate', fail_if_email_gate_runs)
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fake_rag_answer)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Generate and email'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'ParaWorks 회사 소개서 작성해서 용희님한테 메일 보내줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    draft = assistant_message['metadata']['email_draft']
    assert assistant_message['metadata']['action_type'] == 'email_draft'
    assert assistant_message['metadata']['source_context']['kind'] == 'generated_rag_answer'
    assert draft['to'] == ['yonghee199702@gmail.com']
    assert '조직의 기억' in draft['body']
    assert 'Review Queue' in draft['body']


def test_assistant_generate_email_unknown_recipient_does_not_reuse_pending_draft(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    def fail_if_costly_email_work_runs(*args, **kwargs):
        raise AssertionError('수신자가 확정되지 않은 생성-메일 요청은 RAG나 초안 작성으로 진행하면 안 됩니다.')

    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_costly_email_work_runs)
    monkeypatch.setattr(assistant_api, 'build_email_draft_composer', fail_if_costly_email_work_runs)
    conversation = create_conversation(db_session, USERS['viewer'], title='Unknown recipient guard')
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='기존 종우님 초안입니다.',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={
            'action_type': 'email_draft',
            'status': 'pending_approval',
            'email_draft': {
                'to': ['kjw4work@gmail.com'],
                'subject': 'ParaWorks 회사 소개서 전달드립니다',
                'body': '기존 초안 본문',
            },
        },
    )

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'ParaWorks 회사 소개서 작성해서 한승혁님한테 메일 보내줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['metadata']['action_type'] == 'email_clarification'
    assert assistant_message['metadata']['reason'] == 'recipient_not_resolved'
    assert 'kjw4work@gmail.com' not in assistant_message['content']


def test_assistant_recipient_only_revision_preserves_pending_draft_body(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    draft_body = (
        'ParaWorks 회사 소개\n'
        'ParaWorks는 조직의 기억을 구축하는 AI 플랫폼입니다.\n'
        'Review Queue와 Permission-Aware RAG를 제공합니다.'
    )

    def fail_if_rag_runs(*args, **kwargs):
        raise AssertionError('수신자만 바꾸는 요청은 기존 초안 본문을 재사용해야 하며 RAG를 다시 호출하면 안 됩니다.')

    def compose_revision(**kwargs):
        assert draft_body in kwargs['rag_context']
        assert kwargs['resolved_recipients'][0]['email'] == 'hanvv3@gmail.com'
        return assistant_api.EmailActionDecision(
            action_type='email_draft',
            to=['hanvv3@gmail.com'],
            subject='ParaWorks 회사 소개서 전달드립니다',
            body='수정된 수신자로 다시 전달드립니다.',
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=False),
        draft_decision=compose_revision,
    )
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_rag_runs)
    conversation = create_conversation(db_session, USERS['viewer'], title='Recipient correction')
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='메일 초안을 작성했습니다.',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={
            'action_type': 'email_draft',
            'status': 'pending_approval',
            'email_draft': {
                'to': ['kjw4work@gmail.com'],
                'subject': 'ParaWorks 회사 소개서 전달드립니다',
                'body': draft_body,
            },
        },
    )

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'SeungHun Han님한테 보내줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    draft = assistant_message['metadata']['email_draft']
    assert draft['to'] == ['hanvv3@gmail.com']
    assert 'Review Queue' in draft['body']
    assert 'Permission-Aware RAG' in draft['body']


def test_assistant_wrong_email_address_asks_for_correct_recipient(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    def fail_if_email_or_rag_runs(*args, **kwargs):
        raise AssertionError('메일 주소 오류 지적은 새 초안이나 RAG가 아니라 수신자 수정 질문으로 처리해야 합니다.')

    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_email_or_rag_runs)
    monkeypatch.setattr(assistant_api, 'build_email_draft_composer', fail_if_email_or_rag_runs)
    conversation = create_conversation(db_session, USERS['viewer'], title='Wrong recipient')
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='메일 초안을 작성했습니다.',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={
            'action_type': 'email_draft',
            'status': 'pending_approval',
            'email_draft': {
                'to': ['kjw4work@gmail.com'],
                'subject': 'ParaWorks 회사 소개서 전달드립니다',
                'body': '기존 초안 본문',
            },
        },
    )

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': '메일 주소가 잘못됐어.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['metadata']['action_type'] == 'email_clarification'
    assert assistant_message['metadata']['reason'] == 'recipient_correction_requested'
    assert '수신자' in assistant_message['content']


def test_assistant_referenced_answer_email_keeps_selected_content(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    source_content = (
        'ParaWorks 회사 소개\n'
        'ParaWorks는 조직의 기억을 하나로 모으는 AI 기반 회사 기억 플랫폼입니다.\n'
        '핵심 가치: 흩어진 지식의 통합, 신뢰 가능한 지식화, 권한 기반 보안.\n'
        '주요 기능: Slack, Gmail, Google Drive 데이터 동기화와 Review Queue.'
    )

    def fail_if_email_gate_or_rag_runs(*args, **kwargs):
        raise AssertionError('참조 메일 요청은 이전 답변을 메일 본문 후보로 고정한 뒤 초안 작성으로 바로 라우팅해야 합니다.')

    def compose_generic_draft(**kwargs):
        assert source_content in kwargs['rag_context']
        assert kwargs['resolved_recipients'][0]['email'] == 'yonghee199702@gmail.com'
        return assistant_api.EmailActionDecision(
            action_type='email_draft',
            to=['yonghee199702@gmail.com'],
            subject='ParaWorks 회사 소개서 공유드립니다',
            body='회사 소개서 초안을 공유드립니다. 검토 부탁드립니다.',
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=False),
        draft_decision=compose_generic_draft,
    )
    monkeypatch.setattr(assistant_api, 'build_email_intent_gate', fail_if_email_gate_or_rag_runs)
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_email_gate_or_rag_runs)
    conversation = create_conversation(db_session, USERS['viewer'], title='Referenced answer email')
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content=source_content,
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={'agent_name': 'rag_orchestrator_agent'},
    )

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': '이 내용을 용희님한테 메일로 보내줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    draft = assistant_message['metadata']['email_draft']
    assert assistant_message['metadata']['action_type'] == 'email_draft'
    assert assistant_message['metadata']['source_context']['kind'] == 'assistant_answer'
    assert draft['to'] == ['yonghee199702@gmail.com']
    assert '핵심 가치' in draft['body']
    assert '주요 기능' in draft['body']


def test_assistant_revises_pending_draft_when_user_says_body_is_missing(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    source_content = (
        'ParaWorks 회사 소개\n'
        'ParaWorks는 Slack, Gmail, Google Drive의 업무 데이터를 연결합니다.\n'
        'Review Queue를 통해 AI가 추출한 후보를 사람이 승인합니다.'
    )

    def fail_if_email_gate_or_rag_runs(*args, **kwargs):
        raise AssertionError('초안 수정 요청은 기존 승인 대기 초안과 이전 산출물을 사용해야 합니다.')

    def compose_revision(**kwargs):
        assert source_content in kwargs['rag_context']
        assert kwargs['resolved_recipients'][0]['email'] == 'yonghee199702@gmail.com'
        return assistant_api.EmailActionDecision(
            action_type='email_draft',
            to=['yonghee199702@gmail.com'],
            subject='ParaWorks 회사 소개서 초안 공유드립니다',
            body='회사 소개서 초안을 공유드립니다.',
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=False),
        draft_decision=compose_revision,
    )
    monkeypatch.setattr(assistant_api, 'build_email_intent_gate', fail_if_email_gate_or_rag_runs)
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_email_gate_or_rag_runs)
    conversation = create_conversation(db_session, USERS['viewer'], title='Draft revision')
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content=source_content,
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={'agent_name': 'rag_orchestrator_agent'},
    )
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='메일 초안을 작성했습니다.',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={
            'action_type': 'email_draft',
            'status': 'pending_approval',
            'email_draft': {
                'to': ['yonghee199702@gmail.com'],
                'subject': 'ParaWorks 회사 소개서 공유드립니다',
                'body': '회사 소개서 초안을 공유드립니다.',
            },
        },
    )

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': '내용이 하나도 안 들어가 있잖아.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    draft = assistant_message['metadata']['email_draft']
    assert assistant_message['metadata']['action_type'] == 'email_draft'
    assert assistant_message['metadata']['source_context']['kind'] == 'assistant_answer'
    assert draft['to'] == ['yonghee199702@gmail.com']
    assert 'Google Drive' in draft['body']
    assert 'Review Queue' in draft['body']


def test_assistant_contact_lookup_returns_known_email_without_email_draft(
    client: TestClient,
    monkeypatch,
) -> None:
    def fail_if_email_or_rag_runs(*args, **kwargs):
        raise AssertionError('연락처 조회는 이메일 작성이나 RAG 흐름으로 라우팅되면 안 됩니다.')

    monkeypatch.setattr(assistant_api, 'build_email_intent_gate', fail_if_email_or_rag_runs)
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_email_or_rag_runs)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Contact lookup'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': '김종우님 이메일 알려줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert 'kjw4work@gmail.com' in assistant_message['content']
    assert assistant_message['metadata']['action_type'] == 'contact_lookup'
    assert assistant_message['metadata']['status'] == 'resolved'


def test_assistant_contact_lookup_followup_uses_recent_lookup_request(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    def fail_if_email_or_rag_runs(*args, **kwargs):
        raise AssertionError('연락처 조회 후속 답변은 이메일 작성이나 RAG 흐름으로 라우팅되면 안 됩니다.')

    monkeypatch.setattr(assistant_api, 'build_email_intent_gate', fail_if_email_or_rag_runs)
    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', fail_if_email_or_rag_runs)
    conversation = create_conversation(db_session, USERS['viewer'], title='Contact lookup followup')
    append_user_message(db_session, USERS['viewer'], conversation, '김종우님 이메일 알려줘.')
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='김종우님의 이메일 주소를 알려주시겠어요?',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
    )

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': '너가 알려줘야지.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert 'kjw4work@gmail.com' in assistant_message['content']
    assert assistant_message['metadata']['action_type'] == 'contact_lookup'


def test_assistant_passes_resolved_recipient_to_email_draft_composer(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    def compose_with_resolved_recipient(**kwargs):
        assert kwargs['resolved_recipients'] == [
            {
                'email': 'yonghee199702@gmail.com',
                'display_name': '김용희',
                'title': 'CTO',
                'department': 'platform',
                'source_type': 'conversation',
                'confidence_score': 1.0,
            }
        ]
        return assistant_api.EmailActionDecision(
            action_type='email_draft',
            to=['yonghee199702@gmail.com'],
            subject='회의 일정 안내',
            body='오늘 회의는 오후 3시에 진행됩니다.',
        )

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=True),
        draft_decision=compose_with_resolved_recipient,
    )
    conversation = create_conversation(db_session, USERS['viewer'], title='Recipient resolver')
    append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='문의처: 김용희 (yonghee199702@gmail.com)',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
    )

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': '김용희님한테 오늘 회의 3시에 있다고 메일 보내줘.'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 200
    assistant_message = turn_response.json()['assistant_message']
    assert assistant_message['metadata']['email_draft']['to'] == ['yonghee199702@gmail.com']


def test_assistant_email_draft_requires_approval_endpoint_before_send(
    client: TestClient,
    monkeypatch,
) -> None:
    class FakeEmailActionAgent:
        def decide(self, **kwargs):
            return assistant_api.EmailActionDecision(
                action_type='email_draft',
                to=['partner@example.com'],
                subject='회의 취소 안내',
                body='안녕하세요.\n\n오늘 회의가 취소되었습니다.\n\n감사합니다.',
            )

    class FakeGmailDraftSender:
        def __init__(self, **kwargs):
            pass

        def send(self, **kwargs):
            return SimpleNamespace(message_id='gmail-sent-1')

    _patch_email_flow(
        monkeypatch,
        intent_decision=_email_intent(email_intent=True),
        draft_decision=FakeEmailActionAgent().decide,
    )
    monkeypatch.setattr(assistant_api, 'GmailDraftSender', FakeGmailDraftSender)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '메일 승인'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']
    draft_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'partner@example.com에 오늘 회의 취소됐다고 메일 보내줘.'},
        headers={'X-Demo-User': 'viewer'},
    )
    draft_message = draft_response.json()['assistant_message']

    send_response = client.post(
        f"/api/v1/assistant/messages/{draft_message['id']}/email/send",
        headers={'X-Demo-User': 'viewer'},
    )

    assert send_response.status_code == 200
    payload = send_response.json()
    assert payload['status'] == 'sent'
    assert payload['gmail_message_id'] == 'gmail-sent-1'
    assert payload['message']['metadata']['status'] == 'sent'
    assert payload['message']['metadata']['gmail_message_id'] == 'gmail-sent-1'


def test_assistant_lists_persisted_messages(client: TestClient, db_session: Session) -> None:
    seed_chunk(
        db_session,
        'drive',
        'drive-redis-assistant',
        'Redis powers short-lived orchestration state.',
        'internal',
    )
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Persisted conversation'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']
    client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'Redis orchestration state'},
        headers={'X-Demo-User': 'viewer'},
    )

    messages_response = client.get(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )

    assert messages_response.status_code == 200
    messages = messages_response.json()['messages']
    assert [message['role'] for message in messages] == ['user', 'assistant']


def test_assistant_rejects_whitespace_message_without_storing(
    client: TestClient,
    db_session: Session,
) -> None:
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Blank content'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': '   '},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 422
    assert turn_response.json()['detail'] == 'assistant message content is required'
    assert db_session.query(AssistantMessage).count() == 0


def test_assistant_records_failed_message_when_rag_answer_is_blank(
    client: TestClient,
    monkeypatch,
) -> None:
    def blank_answer(**kwargs):
        return SimpleNamespace(
            answer='   ',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=987,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v1',
            question=kwargs['question'],
        )

    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', blank_answer)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Blank assistant content'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'Redis job state'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 502
    assert turn_response.json()['detail'] == 'assistant answer generation failed'

    messages_response = client.get(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )

    assert messages_response.status_code == 200
    messages = messages_response.json()['messages']
    assert [message['role'] for message in messages] == ['user', 'assistant']
    assert messages[0]['content'] == 'Redis job state'
    failed_message = messages[1]
    assert failed_message['content'] == assistant_api.ASSISTANT_FAILURE_CONTENT
    assert failed_message['citations'] == []
    assert failed_message['source_ids'] == []
    assert failed_message['source_links'] == []
    assert failed_message['source_snippets'] == []
    assert failed_message['agent_run_id'] == 987
    assert failed_message['metadata']['status'] == 'failed'
    assert failed_message['metadata']['failure_reason'] == 'blank_answer'
    assert failed_message['metadata']['failure_class'] == 'ValueError'
    assert 'estimated_cost_usd' not in failed_message
    assert 'token_usage' not in failed_message
    assert 'cache_key' not in failed_message


def test_assistant_records_failed_message_when_rag_answer_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    def failing_answer(**kwargs):
        raise RuntimeError('rag unavailable')

    monkeypatch.setattr(assistant_api, 'answer_question_with_rag', failing_answer)
    create_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'RAG failure'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = create_response.json()['conversation']['id']

    turn_response = client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'Redis job state'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert turn_response.status_code == 502
    assert turn_response.json()['detail'] == 'assistant answer generation failed'

    messages_response = client.get(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )

    assert messages_response.status_code == 200
    messages = messages_response.json()['messages']
    assert [message['role'] for message in messages] == ['user', 'assistant']
    assert messages[0]['content'] == 'Redis job state'
    failed_message = messages[1]
    assert failed_message['content'] == assistant_api.ASSISTANT_FAILURE_CONTENT
    assert failed_message['citations'] == []
    assert failed_message['source_ids'] == []
    assert failed_message['source_links'] == []
    assert failed_message['source_snippets'] == []
    assert failed_message['permission_level'] is None
    assert failed_message['hidden_match_count'] == 0
    assert failed_message['permission_notice'] is None
    assert failed_message['agent_run_id'] is None
    assert failed_message['metadata']['status'] == 'failed'
    assert failed_message['metadata']['failure_reason'] == 'rag_exception'
    assert failed_message['metadata']['failure_class'] == 'RuntimeError'
    assert 'estimated_cost_usd' not in failed_message
    assert 'token_usage' not in failed_message
    assert 'cache_key' not in failed_message


def test_assistant_conversations_list_is_user_scoped(client: TestClient) -> None:
    client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Viewer conversation'},
        headers={'X-Demo-User': 'viewer'},
    )
    client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'Employee conversation'},
        headers={'X-Demo-User': 'hanvv-employee'},
    )

    list_response = client.get(
        '/api/v1/assistant/conversations',
        headers={'X-Demo-User': 'viewer'},
    )

    assert list_response.status_code == 200
    conversations = list_response.json()['conversations']
    assert [conversation['title'] for conversation in conversations] == ['Viewer conversation']


def test_assistant_create_reuses_existing_empty_new_conversation(client: TestClient) -> None:
    first_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '새 대화'},
        headers={'X-Demo-User': 'viewer'},
    )
    second_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '새 대화'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()['conversation']['id'] == first_response.json()['conversation']['id']


def test_assistant_create_makes_new_conversation_after_empty_one_is_used(client: TestClient) -> None:
    first_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '새 대화'},
        headers={'X-Demo-User': 'viewer'},
    )
    conversation_id = first_response.json()['conversation']['id']
    client.post(
        f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'Redis job state'},
        headers={'X-Demo-User': 'viewer'},
    )

    second_response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': '새 대화'},
        headers={'X-Demo-User': 'viewer'},
    )

    assert second_response.status_code == 200
    assert second_response.json()['conversation']['id'] != conversation_id
