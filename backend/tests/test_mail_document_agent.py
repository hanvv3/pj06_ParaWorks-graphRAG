from backend.app.agent_runtime import EvidenceMessage, EvidencePacket, PermissionContext
from backend.app.agents.mail_document_agent import (
    MAIL_DOCUMENT_AGENT_MANIFEST,
    MAIL_DOCUMENT_AGENT_PROMPT_VERSION,
    DeterministicMailDocumentAgentModel,
    MailDocumentAgent,
    MailDocumentAgentModelResponse,
)
from backend.app.agents.mail_document_agent.llm import (
    FallbackMailDocumentAgentModel,
    LangChainMailDocumentAgentModel,
    MailDocumentLlmProviderError,
    render_mail_docs_llm_prompt,
)
from backend.app.agents.mail_document_agent.service import MailDocumentProjectOption


class FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {'input_tokens': 100, 'output_tokens': 40}


class FakeChatModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, messages):
        return FakeChatResponse(self.content)


class FailingMailProvider:
    provider = 'openai'
    max_input_chars = 12000

    def extract(self, packet):
        raise MailDocumentLlmProviderError('bounded_failure')


class FakeMailDocumentModel:
    def extract(self, packet: EvidencePacket) -> MailDocumentAgentModelResponse:
        assert packet.source_type == 'mail_document'
        return MailDocumentAgentModelResponse(
            title='Redis architecture decision',
            summary='Gmail and Drive evidence confirm Redis is used for transient job state.',
            item_type='decision',
            confidence_score=0.86,
            input_tokens=900,
            output_tokens=160,
        )


def test_mail_fallback_records_the_provider_that_actually_succeeded() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:fallback',
        messages=[
            EvidenceMessage(
                source_id='gmail-fallback-1',
                source_url='https://gmail.mock/fallback-1',
                text='고객 계약 검토가 금요일까지 필요합니다.',
                author='owner@example.com',
                timestamp='2026-08-28T09:00:00+09:00',
                permission_level='internal',
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )
    gemini = LangChainMailDocumentAgentModel(
        provider='gemini',
        model_name='gemini-2.5-flash',
        chat_model=FakeChatModel(
            '{"title":"계약 검토","summary":"금요일까지 검토합니다.",'
            '"item_type":"todo","confidence_score":0.9,'
            '"is_business_related":true,"structured_data":{}}'
        ),
        reasoning_effort='none',
        route_version='review-model-route:v1:fallback',
        output_contract_version='mail-document-output:v1',
    )

    result = MailDocumentAgent(
        model=FallbackMailDocumentAgentModel([FailingMailProvider(), gemini])
    ).run(packet)

    assert result.model_provider == 'gemini'
    assert result.cost.model_name == 'gemini-2.5-flash'
    assert result.model_reasoning_effort == 'none'
    assert result.route_version == 'review-model-route:v1:fallback'
    assert result.output_contract_version == 'mail-document-output:v1'


class FakeProjectRouter:
    def route(self, *, candidates, projects):
        return {
            'decisions': [
                {
                    'source_id': candidates[0].source_id,
                    'item_index': 0,
                    'project_key': projects[0].project_key,
                    'project_name': projects[0].name,
                    'confidence_score': 0.91,
                    'assignment_summary': '메일/문서 후보가 Project Alpha 업무와 연결됩니다.',
                    'assignment_reason': '증거 URL과 요약이 Project Alpha Redis 작업과 일치합니다.',
                    'alternatives': [],
                    'needs_user_selection': False,
                }
            ],
            'input_tokens': 20,
            'output_tokens': 8,
            'model_name': 'fake-project-router',
        }


class SpyProjectRouter:
    model_name = 'spy-project-router'

    def __init__(self) -> None:
        self.called = False

    def route(self, *, candidates, projects):
        self.called = True
        return {'decisions': [], 'model_name': self.model_name}


def test_mail_document_agent_manifest_declares_shared_contracts() -> None:
    assert MAIL_DOCUMENT_AGENT_MANIFEST.name == 'mail_document_agent'
    assert MAIL_DOCUMENT_AGENT_MANIFEST.input_contract == 'EvidencePacket'
    assert MAIL_DOCUMENT_AGENT_MANIFEST.output_contract == 'AgentRunResult'
    assert (
        MAIL_DOCUMENT_AGENT_PROMPT_VERSION
        in MAIL_DOCUMENT_AGENT_MANIFEST.prompt_versions
    )
    assert 'history_generation' in MAIL_DOCUMENT_AGENT_MANIFEST.capabilities


def test_mail_document_agent_creates_evidence_backed_candidate() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:all',
        messages=[
            EvidenceMessage(
                source_id='gmail-1',
                source_url='https://gmail.mock/project-alpha/redis-summary',
                text='Redis should be used for transient job state.',
                author='noah@example.com',
                timestamp='2026-04-30T10:15:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            ),
            EvidenceMessage(
                source_id='drive-1',
                source_url='https://drive.mock/project-alpha/architecture-note',
                text='Architecture note: Redis-backed updates replace polling.',
                author='lee@example.com',
                timestamp='2026-04-30T11:00:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'drive'},
            ),
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )
    result = MailDocumentAgent(model=FakeMailDocumentModel()).run(packet)

    assert result.agent_name == 'mail_document_agent'
    assert result.prompt_version == 'mail-document-history:v1'
    assert result.candidates[0].source_links == [
        'https://gmail.mock/project-alpha/redis-summary',
        'https://drive.mock/project-alpha/architecture-note',
    ]
    assert result.cost.token_usage.total_tokens == 1060
    assert result.cost.estimated_cost_usd > 0
    assert result.cache_key


def test_mail_document_agent_run_records_node_workflow_trace() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:workflow',
        messages=[
            EvidenceMessage(
                source_id='gmail-workflow',
                source_url='https://gmail.mock/project-alpha/redis-summary',
                text='Project Alpha Redis 작업 결과를 공유합니다.',
                author='noah@example.com',
                timestamp='2026-04-30T10:15:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=FakeMailDocumentModel()).run(packet)

    assert result.candidates
    assert packet.context['mail_document_workflow'] == {
        'nodes': [
            'preprocess',
            'classify_reviewability',
            'extract_candidate',
            'project_route',
            'build_result',
        ],
        'reviewability_decision': 'reviewable',
        'is_business_related': True,
        'candidate_count': 1,
    }


def test_mail_document_agent_workflow_skips_project_route_for_unreviewable_llm_output() -> (
    None
):
    class UnreviewableModel:
        def extract(self, packet: EvidencePacket) -> MailDocumentAgentModelResponse:
            return MailDocumentAgentModelResponse(
                title='개인 메일',
                summary='업무 관련 없음',
                item_type='history_event',
                confidence_score=0.2,
                input_tokens=10,
                output_tokens=5,
                model_name='fake-mail-llm',
                is_business_related=False,
                structured_data={
                    'reviewability_decision': 'not_reviewable',
                    'summary_quality': 'non_business',
                },
            )

    router = SpyProjectRouter()
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:workflow-unreviewable',
        messages=[
            EvidenceMessage(
                source_id='gmail-personal',
                source_url='https://gmail.mock/personal',
                text='Subject: Lunch\n\nNo work talk.',
                author='friend@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
        context={
            'project_options': [
                MailDocumentProjectOption(
                    project_key='project-alpha',
                    name='Project Alpha',
                    summary='Redis worker status work',
                )
            ],
            'project_router': router,
        },
    )

    result = MailDocumentAgent(model=UnreviewableModel()).run(packet)

    assert result.candidates == []
    assert router.called is False
    assert packet.context['mail_document_workflow']['nodes'] == [
        'preprocess',
        'classify_reviewability',
        'extract_candidate',
        'project_route',
        'build_result',
    ]
    assert (
        packet.context['mail_document_workflow']['reviewability_decision']
        == 'not_reviewable'
    )


def test_mail_document_agent_run_routes_projects_inside_workflow() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:project-routing',
        messages=[
            EvidenceMessage(
                source_id='gmail-project-alpha',
                source_url='https://gmail.mock/project-alpha/redis-summary',
                text='Project Alpha Redis 작업 결과를 공유합니다.',
                author='noah@example.com',
                timestamp='2026-04-30T10:15:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
        context={
            'project_options': [
                MailDocumentProjectOption(
                    project_key='project-alpha',
                    name='Project Alpha',
                    summary='Redis worker status work',
                )
            ],
            'project_router': FakeProjectRouter(),
        },
    )

    result = MailDocumentAgent(model=FakeMailDocumentModel()).run(packet)

    candidate = result.candidates[0]
    assert candidate.payload_fields['project_assignment_method'] == 'llm_tool'
    assert candidate.payload_fields['project_key'] == 'project-alpha'
    assert candidate.payload_fields['project_name'] == 'Project Alpha'
    assert candidate.payload_fields['project_assignment_confidence'] == 0.91
    assert result.cost.token_usage.input_tokens == 920
    assert result.cost.token_usage.output_tokens == 168
    assert packet.context['project_routing'] == {
        'enabled': True,
        'method': 'langchain_tools',
        'project_count': 1,
        'model_name': 'fake-project-router',
        'input_tokens': 20,
        'output_tokens': 8,
    }


def test_mail_document_llm_treats_string_false_as_not_business_related() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:test',
        messages=[
            EvidenceMessage(
                source_id='gmail-personal-1',
                source_url='https://gmail.mock/personal',
                text='Subject: Lunch\n\nThis is a personal lunch message.',
                author='friend@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )
    model = LangChainMailDocumentAgentModel(
        provider='openai',
        model_name='gpt-5.4-mini',
        chat_model=FakeChatModel(
            '{"title":"개인 메일","summary":"업무 관련 없음","item_type":"history_event",'
            '"confidence_score":0.91,"is_business_related":"false","structured_data":{}}'
        ),
    )

    result = MailDocumentAgent(model=model).run(packet)

    assert result.candidates == []


def test_mail_document_llm_treats_personal_label_as_not_business_related() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:test',
        messages=[
            EvidenceMessage(
                source_id='gmail-personal-2',
                source_url='https://gmail.mock/personal-2',
                text='Subject: Weekend\n\nPersonal plans only.',
                author='friend@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )
    model = LangChainMailDocumentAgentModel(
        provider='openai',
        model_name='gpt-5.4-mini',
        chat_model=FakeChatModel(
            '{"title":"개인 메일","summary":"업무 관련 없음","item_type":"history_event",'
            '"confidence_score":0.91,"is_business_related":"personal","structured_data":{}}'
        ),
    )

    result = MailDocumentAgent(model=model).run(packet)

    assert result.candidates == []


def test_deterministic_mail_document_agent_skips_personal_email() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:personal',
        messages=[
            EvidenceMessage(
                source_id='gmail-personal-lunch',
                source_url='https://gmail.mock/personal-lunch',
                text='Subject: Lunch\n\nAre you free for lunch this weekend? No work talk, just catching up.',
                author='friend@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)

    assert result.candidates == []


def test_deterministic_mail_document_agent_extracts_calendar_meeting_as_timeline_event() -> (
    None
):
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs-calendar:meeting',
        messages=[
            EvidenceMessage(
                source_id='calendar:team@example.com:event-launch',
                source_url='https://calendar.google.com/event?eid=event-launch',
                text=(
                    'Project Alpha launch milestone meeting\n\n'
                    'Description: Customer launch date and milestone scope are confirmed.\n'
                    'Start: 2026-06-10T10:00:00+09:00\n'
                    'End: 2026-06-10T11:00:00+09:00'
                ),
                author='lead@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={
                    'source_type': 'calendar',
                    'calendar_id': 'team@example.com',
                    'calendar_summary': 'Team Calendar',
                    'event_status': 'confirmed',
                    'event_start': '2026-06-10T10:00:00+09:00',
                    'event_end': '2026-06-10T11:00:00+09:00',
                    'location': 'Meet',
                    'organizer_email': 'lead@example.com',
                    'attendee_domains': ['example.com', 'customer.co.kr'],
                    'event_context_key': 'event-launch:2026-05-13T09:00:00Z',
                },
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)
    candidate = result.candidates[0]

    assert candidate.item_type == 'timeline_event'
    assert candidate.payload_fields['calendar_id'] == 'team@example.com'
    assert candidate.payload_fields['calendar_name'] == 'Team Calendar'
    assert candidate.payload_fields['calendar_start'] == '2026-06-10T10:00:00+09:00'
    assert candidate.payload_fields['calendar_end'] == '2026-06-10T11:00:00+09:00'
    assert candidate.payload_fields['calendar_location'] == 'Meet'
    assert candidate.payload_fields['calendar_organizer'] == 'lead@example.com'
    assert (
        candidate.payload_fields['calendar_attendee_summary']
        == 'example.com, customer.co.kr'
    )
    assert (
        candidate.payload_fields['event_context_key']
        == 'event-launch:2026-05-13T09:00:00Z'
    )


def test_deterministic_mail_document_agent_extracts_calendar_preparation_as_todo() -> (
    None
):
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs-calendar:todo',
        messages=[
            EvidenceMessage(
                source_id='calendar:primary:event-prep',
                source_url='https://calendar.google.com/event?eid=event-prep',
                text=(
                    'Customer proposal preparation deadline\n\n'
                    'Description: Please prepare the proposal deck before the customer meeting.\n'
                    'Start: 2026-06-03T09:00:00+09:00\n'
                    'End: 2026-06-03T09:30:00+09:00'
                ),
                author='lead@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={
                    'source_type': 'calendar',
                    'calendar_id': 'primary',
                    'calendar_summary': 'Primary Calendar',
                    'event_status': 'confirmed',
                    'event_start': '2026-06-03T09:00:00+09:00',
                    'event_end': '2026-06-03T09:30:00+09:00',
                    'organizer_email': 'lead@example.com',
                    'attendee_domains': ['example.com'],
                    'event_context_key': 'event-prep:2026-05-13T09:00:00Z',
                },
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)
    candidate = result.candidates[0]

    assert candidate.item_type == 'todo'
    assert candidate.payload_fields['calendar_id'] == 'primary'
    assert candidate.payload_fields['calendar_start'] == '2026-06-03T09:00:00+09:00'


def test_deterministic_mail_document_agent_skips_low_signal_personal_calendar_event() -> (
    None
):
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs-calendar:personal',
        messages=[
            EvidenceMessage(
                source_id='calendar:primary:event-dentist',
                source_url='https://calendar.google.com/event?eid=event-dentist',
                text='Dentist appointment\n\nStart: 2026-06-03T09:00:00+09:00',
                author='me@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={
                    'source_type': 'calendar',
                    'calendar_id': 'primary',
                    'calendar_summary': 'Primary Calendar',
                    'event_status': 'confirmed',
                    'event_start': '2026-06-03T09:00:00+09:00',
                },
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)

    assert result.candidates == []


def test_mail_document_llm_prompt_requires_reviewable_business_decision() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:prompt-quality',
        messages=[
            EvidenceMessage(
                source_id='gmail-newsletter',
                source_url='https://gmail.mock/newsletter',
                text='Subject: Weekly digest\n\nHere are general product updates and promotional announcements.',
                author='newsletter@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    prompt = render_mail_docs_llm_prompt(packet)

    assert 'reviewability_decision' in prompt
    assert 'set is_business_related=false' in prompt
    assert 'personal mail, newsletters, promotions' in prompt


def test_deterministic_mail_document_agent_marks_metadata_only_evidence_uncertain() -> (
    None
):
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:drive',
        messages=[
            EvidenceMessage(
                source_id='drive-pdf-1',
                source_url='https://drive.mock/policy.pdf',
                text='Google Drive file changed: 휴가 정책 PDF',
                author='owner@example.com',
                timestamp='2026-05-01T09:00:00+00:00',
                permission_level='restricted',
                metadata={
                    'source_type': 'drive',
                    'parser_status': 'metadata_only',
                    'parser_status_reason': 'pdf_parser_not_enabled',
                },
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)
    candidate = result.candidates[0]

    assert candidate.confidence_score == 0.42
    assert candidate.uncertainty_reason == (
        'Some document evidence is not body-parsed: drive-pdf-1=metadata_only(pdf_parser_not_enabled)'
    )


def test_deterministic_mail_document_agent_marks_unsupported_evidence_uncertain() -> (
    None
):
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:drive',
        messages=[
            EvidenceMessage(
                source_id='drive-hwp-1',
                source_url='https://drive.mock/policy.hwp',
                text='Google Drive file changed: 휴가 정책 HWP',
                author='owner@example.com',
                timestamp='2026-05-01T09:00:00+00:00',
                permission_level='restricted',
                metadata={
                    'source_type': 'drive',
                    'parser_status': 'unsupported',
                    'parser_status_reason': 'hwp_parser_not_decided',
                },
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)
    candidate = result.candidates[0]

    assert candidate.confidence_score == 0.3
    assert candidate.uncertainty_reason == (
        'Some document evidence is not body-parsed: drive-hwp-1=unsupported(hwp_parser_not_decided)'
    )


def test_deterministic_mail_document_agent_extracts_generic_korean_work_assignment() -> (
    None
):
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:gmail',
        messages=[
            EvidenceMessage(
                source_id='gmail-task-1',
                source_url='https://gmail.mock/task-1',
                text='Subject: 고객사 공유본 요청\n\n김하나님, 금요일까지 고객사 공유본을 준비해주세요.',
                author='lead@example.com',
                timestamp='2026-05-13T09:00:00+09:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
                source_snippet_override='김하나님, 금요일까지 고객사 공유본을 준비해주세요.',
            ),
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)
    candidate = result.candidates[0]

    assert candidate.item_type == 'todo'
    assert candidate.title == '고객사 공유본 요청'
    assert candidate.summary == '김하나님, 금요일까지 고객사 공유본을 준비해주세요.'
    assert candidate.source_snippets == [
        '김하나님, 금요일까지 고객사 공유본을 준비해주세요.'
    ]
    assert candidate.payload_fields['assignee'] == '김하나'
    assert candidate.payload_fields['due_date'] == '금요일'
    assert (
        candidate.payload_fields['task_summary']
        == '김하나님, 금요일까지 고객사 공유본을 준비해주세요.'
    )
    assert candidate.payload_fields['evidence_reason']


def test_deterministic_mail_document_agent_summarizes_korean_business_request_without_raw_email_header() -> (
    None
):
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:k-tech',
        messages=[
            EvidenceMessage(
                source_id='gmail-k-tech',
                source_url='https://gmail.mock/k-tech',
                text=(
                    'Subject: RE: [논의] K테크 솔루션즈 파일럿 제안 검토 요청\n'
                    'From: "김종우" <kjw4work@gmail.com>\n'
                    'Date: Wed, 13 May 2026 16:48:53 +0900\n\n'
                    'K테크 솔루션즈 측에서 ParaWorks 파일럿 도입에 큰 관심을 보이고 있습니다. '
                    '1개월 파일럿이 유의미한 결과를 낼 수 있을지 검토하고 회신이 필요합니다.'
                ),
                author='kjw4work@gmail.com',
                timestamp='2026-05-13T16:48:53+09:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
                source_snippet_override=(
                    'From: "김종우" <kjw4work@gmail.com> Date: Wed, 13 May 2026 '
                    'K테크 솔루션즈 측에서 ParaWorks 파일럿 도입에 큰 관심을 보이고 있습니다.'
                ),
            ),
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)
    candidate = result.candidates[0]

    assert candidate.item_type == 'todo'
    assert not candidate.title.startswith('RE:')
    assert 'From:' not in candidate.summary
    assert 'Date:' not in candidate.summary
    assert candidate.payload_fields['recommended_next_step']


def test_deterministic_mail_document_agent_extracts_drive_only_assignment() -> None:
    packet = EvidencePacket(
        source_type='mail_document',
        source_window='mail-docs:drive',
        messages=[
            EvidenceMessage(
                source_id='drive-plan-1',
                source_url='https://drive.mock/project-alpha/plan',
                text='프로젝트 Alpha 실행 계획\n\n담당: 이준호\n업무: 제안서 초안 검토\n기한: 2026-05-20',
                author='owner@example.com',
                timestamp='2026-05-13T10:00:00+09:00',
                permission_level='restricted',
                metadata={'source_type': 'drive', 'parser_status': 'parsed'},
                source_snippet_override='담당: 이준호 업무: 제안서 초안 검토 기한: 2026-05-20',
            ),
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    result = MailDocumentAgent(model=DeterministicMailDocumentAgentModel()).run(packet)
    candidate = result.candidates[0]

    assert candidate.item_type == 'todo'
    assert candidate.permission_level == 'restricted'
    assert candidate.payload_fields['assignee'] == '이준호'
    assert candidate.payload_fields['due_date'] == '2026-05-20'
    assert '제안서 초안 검토' in candidate.summary
