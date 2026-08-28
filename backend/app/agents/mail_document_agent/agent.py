import re
from dataclasses import dataclass
from typing import Any, Protocol

from backend.app.agent_runtime import (
    AgentManifest,
    AgentRunResult,
    EvidenceMessage,
    EvidencePacket,
    TokenUsage,
    build_evidence_cache_key,
    estimate_agent_run_cost,
)
from backend.app.agents.mail_document_agent.workflow import (
    run_mail_document_agent_workflow,
)

# 메일 및 문서 에이전트의 상수 정의
MAIL_DOCUMENT_AGENT_NAME = 'mail_document_agent'
MAIL_DOCUMENT_AGENT_PROMPT_VERSION = 'mail-document-history:v1'
MAIL_DOCUMENT_AGENT_MODEL_NAME = 'fake-mail-document-agent-model'

# 에이전트의 명세(Manifest) 정의: 소유자, 권한, 기능 등을 기술
MAIL_DOCUMENT_AGENT_MANIFEST = AgentManifest(
    name=MAIL_DOCUMENT_AGENT_NAME,
    owner='Developer B',
    input_contract='EvidencePacket',
    output_contract='AgentRunResult',
    prompt_versions=(MAIL_DOCUMENT_AGENT_PROMPT_VERSION,),
    supported_permissions=('internal', 'restricted'),
    capabilities=('timeline_extraction', 'history_generation', 'decision_extraction'),
)

_RESERVED_REVIEW_PAYLOAD_FIELDS = {
    'title',
    'summary',
    'agent_name',
    'agent_run_id',
    'prompt_version',
    'cache_key',
    'estimated_cost_usd',
    'token_usage',
    'uncertainty_reason',
    'source_ids',
    'source_types',
    'source_urls',
    'source_authors',
}


@dataclass(frozen=True)
class MailDocumentAgentModelResponse:
    """에이전트 모델의 추출 결과 데이터 구조"""

    title: str
    summary: str
    item_type: str
    confidence_score: float
    input_tokens: int
    output_tokens: int
    model_name: str | None = None
    uncertainty_reason: str | None = None
    is_business_related: bool = True
    project_tag: str | None = None
    structured_data: dict[str, Any] | None = None
    model_provider: str | None = None
    model_reasoning_effort: str | None = None
    route_version: str | None = None
    output_contract_version: str | None = None


class MailDocumentAgentModel(Protocol):
    """에이전트 모델이 구현해야 할 인터페이스(프로토콜)"""

    def extract(self, packet: EvidencePacket) -> MailDocumentAgentModelResponse:
        raise NotImplementedError


class DeterministicMailDocumentAgentModel:
    """LLM 없이 규칙 기반으로 정보를 추출하는 테스트용 결정론적 모델"""

    def extract(self, packet: EvidencePacket) -> MailDocumentAgentModelResponse:
        combined_text = '\n'.join(message.text for message in packet.messages)
        input_tokens = max(1, len(combined_text) // 4)
        if _looks_like_non_business_evidence(combined_text):
            return MailDocumentAgentModelResponse(
                title='비업무 메일/문서',
                summary='검토 큐에 올릴 회사 업무 증거가 확인되지 않았습니다.',
                item_type='history_event',
                confidence_score=0.2,
                input_tokens=input_tokens,
                output_tokens=32,
                uncertainty_reason='personal_or_low_signal_evidence',
                is_business_related=False,
            )
        calendar_response = _extract_calendar_candidate(
            packet, input_tokens=input_tokens
        )
        if calendar_response is not None:
            return calendar_response
        title = '메일 및 문서 히스토리 후보'
        summary = 'Gmail 및 Google Drive 증거 데이터가 검토 가능한 회사 메모리 후보로 요약되었습니다.'
        item_type = 'history_event'
        structured_data: dict[str, Any] = {}

        assignment = _extract_assignment(packet)
        if assignment:
            title = str(assignment.get('title') or '업무 지시 후보')
            summary = str(
                assignment.get('task_summary')
                or assignment.get('evidence_sentence')
                or summary
            )
            item_type = 'todo'
            structured_data = assignment

        # 특정 키워드에 따른 결과 분류 로직
        if 'PostgreSQL' in combined_text and 'Redis' in combined_text:
            title = 'Redis 및 PostgreSQL 역할 분담 결정'
            summary = '메일 및 문서 증거에 따르면, Redis는 일시적인 작업 상태를 처리하고 PostgreSQL은 영구적인 기록 소스로 유지됩니다.'
            item_type = 'decision_record'
            structured_data = {'decision_summary': summary, **structured_data}
        elif 'confidential pricing' in combined_text.lower():
            title = '검토가 필요한 제한된 문서 증거'
            summary = 'Google Drive 증거에 권한 확인이 필요한 기밀 가격 정보가 포함되어 있습니다.'
            item_type = 'history_event'
        elif 'budget' in combined_text.lower() or 'revenue' in combined_text.lower():
            title = '분기별 예산 및 매출 전략 업데이트됨'
            summary = '파싱된 문서 증거에 따르면 분기별 예산 및 채용 계획이 업데이트되었습니다.'
            item_type = 'history_event'
        elif (
            'contract review' in combined_text.lower()
            or 'due friday' in combined_text.lower()
        ):
            title = '계약서 검토 일정 예약됨'
            summary = '이번 주 금요일까지 계약서 검토를 완료해야 한다는 내용이 이메일/문서에서 추출되었습니다.'
            item_type = 'todo'
            structured_data = {
                **structured_data,
                'task_summary': summary,
                'due_date': structured_data.get('due_date') or '금요일',
                'evidence_reason': structured_data.get('evidence_reason')
                or '기한이 포함된 업무 지시 표현이 있습니다.',
            }
        elif packet.messages:
            summary = _business_context_summary(packet.messages[0])

        # 토큰 사용량 및 신뢰도 계산 (모의 계산)
        output_tokens = max(32, len(summary) // 4)
        confidence_score = 0.8
        uncertainty_reason = _parser_uncertainty_reason(packet)
        if uncertainty_reason:
            confidence_score = _parser_uncertainty_confidence(packet)

        return MailDocumentAgentModelResponse(
            title=title,
            summary=summary,
            item_type=item_type,
            confidence_score=confidence_score,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            uncertainty_reason=uncertainty_reason,
            is_business_related=True,
            project_tag='General',
            structured_data=structured_data,
        )


@dataclass(frozen=True)
class MailDocumentAgent:
    """메일 및 문서 데이터를 처리하여 검토 후보를 생성하는 에이전트 클래스"""

    model: MailDocumentAgentModel
    input_cost_per_1m: float = 0.15
    output_cost_per_1m: float = 0.60

    def run(self, packet: EvidencePacket) -> AgentRunResult:
        """증거 패킷을 입력받아 모델을 실행하고 결과를 AgentRunResult로 반환"""
        model_response, candidates, project_routing_result = (
            run_mail_document_agent_workflow(
                packet=packet,
                model=self.model,
                normalize_item_type=_normalized_item_type,
                safe_payload_fields=_safe_payload_fields,
            )
        )
        # 비용 및 토큰 사용량 기록
        token_usage = TokenUsage(
            input_tokens=model_response.input_tokens
            + project_routing_result.input_tokens,
            output_tokens=model_response.output_tokens
            + project_routing_result.output_tokens,
        )
        cost = estimate_agent_run_cost(
            model_name=model_response.model_name or MAIL_DOCUMENT_AGENT_MODEL_NAME,
            token_usage=token_usage,
            input_cost_per_1m=self.input_cost_per_1m,
            output_cost_per_1m=self.output_cost_per_1m,
            cache_hit=False,
        )
        # 캐싱을 위한 키 생성
        cache_key = build_evidence_cache_key(packet, MAIL_DOCUMENT_AGENT_PROMPT_VERSION)

        return AgentRunResult(
            agent_name=MAIL_DOCUMENT_AGENT_NAME,
            prompt_version=MAIL_DOCUMENT_AGENT_PROMPT_VERSION,
            candidates=candidates,
            cost=cost,
            cache_key=cache_key,
            model_provider=model_response.model_provider,
            model_reasoning_effort=model_response.model_reasoning_effort,
            route_version=model_response.route_version,
            output_contract_version=model_response.output_contract_version,
        )


def _parser_uncertainty_reason(packet: EvidencePacket) -> str | None:
    """문서 파싱 상태가 불완전한 경우 그 이유를 추출하는 헬퍼 함수"""
    uncertain_messages = [
        message
        for message in packet.messages
        if message.metadata.get('source_type') == 'drive'
        and message.metadata.get('parser_status')
        and message.metadata.get('parser_status') != 'parsed'
    ]
    if not uncertain_messages:
        return None
    details = []
    for message in uncertain_messages:
        status = message.metadata.get('parser_status')
        reason = message.metadata.get('parser_status_reason') or 'unknown_reason'
        details.append(f'{message.source_id}={status}({reason})')
    return f'Some document evidence is not body-parsed: {", ".join(details)}'


def _parser_uncertainty_confidence(packet: EvidencePacket) -> float:
    """파싱 상태에 따라 신뢰도 점수를 조정하는 헬퍼 함수"""
    statuses = {
        str(message.metadata.get('parser_status'))
        for message in packet.messages
        if message.metadata.get('source_type') == 'drive'
    }
    if 'unsupported' in statuses:
        return 0.3  # 지원되지 않는 형식인 경우 낮은 신뢰도
    return 0.42  # 일반적인 파싱 오류/제한 사항인 경우 중간 신뢰도


def _normalized_item_type(item_type: str) -> str:
    if item_type == 'decision':
        return 'decision_record'
    if item_type == 'timeline':
        return 'timeline_event'
    return item_type


def _safe_payload_fields(fields: dict[str, Any] | None) -> dict[str, Any]:
    if not fields:
        return {}
    return {
        key: value
        for key, value in fields.items()
        if key not in _RESERVED_REVIEW_PAYLOAD_FIELDS
    }


def _extract_calendar_candidate(
    packet: EvidencePacket,
    *,
    input_tokens: int,
) -> MailDocumentAgentModelResponse | None:
    calendar_messages = [
        message
        for message in packet.messages
        if message.metadata.get('source_type') == 'calendar'
    ]
    if not calendar_messages:
        return None
    message = calendar_messages[0]
    text = message.text.strip()
    lowered = text.lower()
    status = str(message.metadata.get('event_status') or '').lower()
    if status and status not in {'confirmed', 'tentative'}:
        return _calendar_not_reviewable_response(
            input_tokens=input_tokens, reason='calendar_event_not_confirmed'
        )

    if _looks_like_calendar_action(text):
        summary = _calendar_summary_sentence(message)
        structured_data = {
            **_calendar_payload_fields(message),
            'task_summary': summary,
            'evidence_sentence': message.source_snippet,
            'evidence_reason': 'Calendar evidence contains preparation, deadline, or follow-up wording.',
            'source_type': 'calendar',
            'action_required': 'true',
            'recommended_next_step': summary,
            'summary_quality': 'actionable_calendar_event',
        }
        due_date = _calendar_due_date(message.metadata)
        if due_date:
            structured_data['due_date'] = due_date
        return MailDocumentAgentModelResponse(
            title=_calendar_title(message),
            summary=summary,
            item_type='todo',
            confidence_score=0.78,
            input_tokens=input_tokens,
            output_tokens=max(32, len(summary) // 4),
            is_business_related=True,
            project_tag='General',
            structured_data=structured_data,
        )

    if _looks_like_calendar_timeline_event(text):
        summary = _calendar_summary_sentence(message)
        return MailDocumentAgentModelResponse(
            title=_calendar_title(message),
            summary=summary,
            item_type='timeline_event',
            confidence_score=0.82,
            input_tokens=input_tokens,
            output_tokens=max(32, len(summary) // 4),
            is_business_related=True,
            project_tag='General',
            structured_data={
                **_calendar_payload_fields(message),
                'result_summary': summary,
                'evidence_reason': 'Calendar evidence confirms a meeting, milestone, or schedule.',
                'source_type': 'calendar',
                'summary_quality': 'calendar_timeline_event',
            },
        )

    personal_cues = (
        'dentist',
        'doctor',
        'personal',
        'private',
        'birthday',
        'lunch',
        'vacation',
        'holiday',
    )
    business_cues = (
        'project',
        'customer',
        'client',
        'launch',
        'milestone',
        'proposal',
        'contract',
        'meeting',
    )
    reason = (
        'personal_or_low_signal_calendar_event'
        if any(cue in lowered for cue in personal_cues)
        and not any(cue in lowered for cue in business_cues)
        else 'low_signal_calendar_event'
    )
    return _calendar_not_reviewable_response(input_tokens=input_tokens, reason=reason)


def _calendar_not_reviewable_response(
    *,
    input_tokens: int,
    reason: str,
) -> MailDocumentAgentModelResponse:
    return MailDocumentAgentModelResponse(
        title='Non-reviewable Calendar event',
        summary='Calendar evidence did not contain a reviewable company work object.',
        item_type='history_event',
        confidence_score=0.2,
        input_tokens=input_tokens,
        output_tokens=32,
        uncertainty_reason=reason,
        is_business_related=False,
        structured_data={
            'reviewability_decision': 'not_reviewable',
            'summary_quality': reason,
        },
    )


def _calendar_payload_fields(message: EvidenceMessage) -> dict[str, str]:
    metadata = message.metadata
    fields = {
        'calendar_id': _metadata_string(metadata, 'calendar_id'),
        'calendar_name': _metadata_string(metadata, 'calendar_summary')
        or _metadata_string(metadata, 'calendar_name'),
        'calendar_start': _metadata_string(metadata, 'event_start')
        or _metadata_string(metadata, 'start'),
        'calendar_end': _metadata_string(metadata, 'event_end')
        or _metadata_string(metadata, 'end'),
        'calendar_location': _metadata_string(metadata, 'location'),
        'calendar_organizer': _metadata_string(metadata, 'organizer_email'),
        'calendar_attendee_summary': _calendar_attendee_summary(metadata),
        'event_context_key': _metadata_string(metadata, 'event_context_key'),
    }
    return {key: value for key, value in fields.items() if value}


def _calendar_attendee_summary(metadata: dict[str, Any]) -> str:
    value = metadata.get('attendee_domains')
    if isinstance(value, list):
        return ', '.join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return value
    return ''


def _metadata_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) else ''


def _looks_like_calendar_action(text: str) -> bool:
    lowered = text.lower()
    return any(
        cue in lowered
        for cue in (
            'prepare',
            'preparation',
            'deadline',
            'due',
            'follow-up',
            'follow up',
            'todo',
            'action item',
            'submit',
            'review by',
            'please',
            '마감',
            '준비',
            '후속',
        )
    )


def _looks_like_calendar_timeline_event(text: str) -> bool:
    lowered = text.lower()
    return any(
        cue in lowered
        for cue in (
            'meeting',
            'milestone',
            'launch',
            'kickoff',
            'review',
            'confirmed',
            'customer',
            'client',
            'project',
            'contract',
            '회의',
            '일정',
            '확정',
        )
    )


def _calendar_title(message: EvidenceMessage) -> str:
    first_line = next(
        (line.strip() for line in message.text.splitlines() if line.strip()), ''
    )
    return first_line[:200] if first_line else 'Calendar event candidate'


def _calendar_summary_sentence(message: EvidenceMessage) -> str:
    summary = _business_context_summary(message)
    start = _metadata_string(message.metadata, 'event_start') or _metadata_string(
        message.metadata, 'start'
    )
    calendar_name = _metadata_string(message.metadata, 'calendar_summary')
    suffix = ' '.join(
        part
        for part in [
            f'Calendar: {calendar_name}' if calendar_name else '',
            f'Start: {start}' if start else '',
        ]
        if part
    )
    return f'{summary} {suffix}'.strip()[:500]


def _calendar_due_date(metadata: dict[str, Any]) -> str:
    start = _metadata_string(metadata, 'event_start') or _metadata_string(
        metadata, 'start'
    )
    return start.split('T', 1)[0] if start else ''


def _extract_assignment(packet: EvidencePacket) -> dict[str, str]:
    for message in packet.messages:
        text = message.text.strip()
        if not _looks_like_work_assignment(text):
            continue
        sentence = _assignment_sentence(text)
        assignee = _extract_assignee(text)
        due_date = _extract_due_date(text, message.metadata)
        task_summary = _extract_task_summary(text, sentence)
        title = _action_title(text, task_summary)
        fields = {
            'title': title[:200],
            'task_summary': (task_summary or sentence or text[:160]).strip()[:500],
            'evidence_sentence': (sentence or message.source_snippet).strip()[:500],
            'evidence_reason': '담당자, 기한, 요청/검토/준비 같은 업무 지시 표현이 원문에 포함되어 있습니다.',
            'source_type': str(
                message.metadata.get('source_type') or packet.source_type
            ),
            'business_context': _business_context_summary(message),
            'action_required': 'true',
            'recommended_next_step': _recommended_next_step(
                task_summary or sentence, text
            ),
            'summary_quality': 'actionable',
        }
        source_subject = _extract_subject(text)
        if source_subject:
            fields['source_subject'] = source_subject
        if assignee:
            fields['assignee'] = assignee
        if due_date:
            fields['due_date'] = due_date
        counterparty = _extract_counterparty(text)
        if counterparty:
            fields['counterparty'] = counterparty
        project_tag = _extract_project_tag(text)
        if project_tag:
            fields['project_tag'] = project_tag
        return fields
    return {}


def _looks_like_work_assignment(text: str) -> bool:
    lowered = text.lower()
    cues = (
        '담당',
        '요청',
        '검토',
        '준비',
        '완료',
        '기한',
        '까지',
        '회신',
        '공유',
        '승인',
        '결정',
        '확인',
        '부탁',
        '파일럿',
        '제안',
        '도입',
        '미팅',
        '일정',
        '담당',
        '요청',
        '검토',
        '준비',
        '완료',
        '기한',
        '까지',
        'todo',
        'due',
        'owner',
        'assign',
        'review by',
        'please',
    )
    return any(cue in lowered for cue in cues)


def _looks_like_non_business_evidence(text: str) -> bool:
    lowered = text.lower()
    non_business_cues = (
        'no work talk',
        'personal plans',
        'free for lunch',
        'lunch this weekend',
        'just catching up',
        'newsletter',
        'weekly digest',
        'promotional announcements',
        'unsubscribe',
        '개인 일정',
        '점심 약속',
        '주말 약속',
        '업무 관련 없음',
    )
    if not any(cue in lowered for cue in non_business_cues):
        return False
    business_cues = (
        'project',
        'contract',
        'proposal',
        'budget',
        'revenue',
        'redis',
        'postgresql',
        'deadline',
        'due ',
        '프로젝트',
        '계약',
        '제안',
        '예산',
        '매출',
        '기한',
        '마감',
        '검토 요청',
        '승인 요청',
    )
    return not any(cue in lowered for cue in business_cues)


def _assignment_sentence(text: str) -> str:
    paragraphs = [
        line.strip()
        for line in re.split(r'[\n\r]+', text)
        if line.strip() and not _is_header_line(line)
    ]
    for paragraph in paragraphs:
        if _looks_like_work_assignment(paragraph):
            return paragraph
    sentences = [
        part.strip() for part in re.split(r'(?<=[.!?。])\s+', text) if part.strip()
    ]
    for sentence in sentences:
        if _looks_like_work_assignment(sentence):
            return sentence
    return paragraphs[0] if paragraphs else ''


def _extract_subject(text: str) -> str:
    match = re.search(r'(?im)^subject:\s*(.+)$', text)
    return match.group(1).strip() if match else ''


def _action_title(text: str, task_summary: str) -> str:
    subject = _extract_subject(text)
    subject = re.sub(r'(?i)^(re|fw|fwd):\s*', '', subject).strip()
    subject = re.sub(r'^\[[^\]]+\]\s*', '', subject).strip()
    if 'K테크' in text and '파일럿' in text:
        return 'K테크 1개월 파일럿 제안 검토 및 회신'
    if subject:
        return subject[:200]
    return (task_summary or '업무 지시 후보')[:200]


def _business_context_summary(message: EvidenceMessage) -> str:
    text = _clean_email_headers(message.text)
    first_sentence = _first_business_sentence(text)
    if first_sentence:
        return first_sentence[:500]
    return '메일 및 문서 증거에서 검토 가능한 회사 업무 맥락이 확인되었습니다.'


def _recommended_next_step(task_summary: str, text: str) -> str:
    if 'K테크' in text and '파일럿' in text:
        return '파일럿 범위, 성공 지표, 일정 초안을 정리해 회신합니다.'
    cleaned = _clean_email_headers(task_summary).strip()
    if cleaned:
        return cleaned[:500]
    return '관련 업무 내용을 검토하고 필요한 후속 조치를 정리합니다.'


def _first_business_sentence(text: str) -> str:
    cleaned = ' '.join(line.strip() for line in text.splitlines() if line.strip())
    for part in re.split(r'(?<=[.!?。]|[다요음])\s+', cleaned):
        sentence = part.strip()
        if sentence and not _is_header_line(sentence):
            return sentence
    return cleaned.strip()


def _clean_email_headers(text: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not _is_header_line(line)
    ]
    return '\n'.join(lines)


def _is_header_line(line: str) -> bool:
    lowered = line.strip().lower()
    return lowered.startswith(('subject:', 'from:', 'date:', 'to:', 'cc:'))


def _extract_counterparty(text: str) -> str:
    match = re.search(
        r'([A-Za-z0-9가-힣]+(?:\s+[A-Za-z0-9가-힣]+)*\s*(?:솔루션즈|테크|컴퍼니|주식회사|팀))',
        text,
    )
    return match.group(1).strip() if match else ''


def _extract_assignee(text: str) -> str:
    label_match = re.search(
        r'(?:담당|owner|assignee)\s*[:：]\s*([^\n,]+)', text, re.IGNORECASE
    )
    if label_match:
        return _clean_assignee(label_match.group(1))
    nim_match = re.search(r'([가-힣A-Za-z0-9._+-]{2,40})님[,은는\s]', text)
    if nim_match:
        return _clean_assignee(nim_match.group(1))
    return ''


def _clean_assignee(value: str) -> str:
    return re.sub(r'(님|께서|은|는)$', '', value.strip()).strip()


def _extract_due_date(text: str, metadata: dict) -> str:
    label_match = re.search(
        r'(?:기한|마감|due(?: date)?)\s*[:：]\s*([^\n,]+)', text, re.IGNORECASE
    )
    if label_match:
        return label_match.group(1).strip()
    until_match = re.search(
        r'((?:\d{4}[-./]\d{1,2}[-./]\d{1,2})|(?:이번\s*)?[월화수목금토일]요일|오늘|내일)\s*까지',
        text,
    )
    if until_match:
        return until_match.group(1).strip()
    start = metadata.get('start')
    if isinstance(start, str) and start:
        return start[:10]
    return ''


def _extract_task_summary(text: str, fallback_sentence: str) -> str:
    task_match = re.search(r'(?:업무|task)\s*[:：]\s*([^\n]+)', text, re.IGNORECASE)
    if task_match:
        return task_match.group(1).strip()
    cleaned = _clean_email_headers(fallback_sentence).strip()
    return cleaned or fallback_sentence


def _extract_project_tag(text: str) -> str:
    match = re.search(
        r'(프로젝트\s*[A-Za-z0-9가-힣_-]+|Project\s+[A-Za-z0-9가-힣_-]+)',
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ''
