import re
from dataclasses import dataclass
from typing import Protocol

from backend.app.agent_runtime import (
    AgentManifest,
    AgentRunResult,
    EvidencePacket,
    ReviewCandidate,
    TokenUsage,
    build_evidence_cache_key,
    estimate_agent_run_cost,
)

MEMORY_EXTRACTION_MODEL_NAME = 'fake-memory-extraction-model'

TIMELINE_AGENT_NAME = 'timeline_agent'
TIMELINE_AGENT_PROMPT_VERSION = 'timeline-extraction:v1'
HISTORY_AGENT_NAME = 'history_agent'
HISTORY_AGENT_PROMPT_VERSION = 'history-extraction:v1'
DECISION_RECORD_AGENT_NAME = 'decision_record_agent'
DECISION_RECORD_AGENT_PROMPT_VERSION = 'decision-record-extraction:v1'
TODO_AGENT_NAME = 'todo_agent'
TODO_AGENT_PROMPT_VERSION = 'todo-extraction:v1'

TIMELINE_AGENT_MANIFEST = AgentManifest(
    name=TIMELINE_AGENT_NAME,
    owner='Developer C',
    input_contract='EvidencePacket',
    output_contract='AgentRunResult',
    prompt_versions=(TIMELINE_AGENT_PROMPT_VERSION,),
    supported_permissions=('internal', 'restricted'),
    capabilities=('timeline_extraction',),
)
HISTORY_AGENT_MANIFEST = AgentManifest(
    name=HISTORY_AGENT_NAME,
    owner='Developer C',
    input_contract='EvidencePacket',
    output_contract='AgentRunResult',
    prompt_versions=(HISTORY_AGENT_PROMPT_VERSION,),
    supported_permissions=('internal', 'restricted'),
    capabilities=('history_generation',),
)
DECISION_RECORD_AGENT_MANIFEST = AgentManifest(
    name=DECISION_RECORD_AGENT_NAME,
    owner='Developer C',
    input_contract='EvidencePacket',
    output_contract='AgentRunResult',
    prompt_versions=(DECISION_RECORD_AGENT_PROMPT_VERSION,),
    supported_permissions=('internal', 'restricted'),
    capabilities=('decision_extraction',),
)
TODO_AGENT_MANIFEST = AgentManifest(
    name=TODO_AGENT_NAME,
    owner='Developer C',
    input_contract='EvidencePacket',
    output_contract='AgentRunResult',
    prompt_versions=(TODO_AGENT_PROMPT_VERSION,),
    supported_permissions=('internal', 'restricted'),
    capabilities=('todo_extraction',),
)


@dataclass(frozen=True)
class MemoryExtractionModelResponse:
    title: str
    summary: str
    item_type: str
    confidence_score: float
    input_tokens: int
    output_tokens: int
    payload_fields: dict[str, str]
    uncertainty_reason: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    model_reasoning_effort: str | None = None
    route_version: str | None = None
    output_contract_version: str | None = None


class MemoryExtractionModel(Protocol):
    def extract(self, packet: EvidencePacket) -> MemoryExtractionModelResponse:
        raise NotImplementedError


class DeterministicTimelineModel:
    def extract(self, packet: EvidencePacket) -> MemoryExtractionModelResponse:
        combined_text = _combined_text(packet)
        summary = _first_sentence(combined_text) or '업무 결과가 발생했습니다.'
        if 'QA' in combined_text or '배포' in combined_text:
            title = 'QA and deployment milestone captured'
            summary = 'QA 완료 및 배포 진행 내역이 timeline 후보로 감지되었습니다.'
        else:
            title = '회사 타임라인 후보'
        return _response(
            packet=packet,
            item_type='timeline_event',
            title=title,
            summary=summary,
            payload_fields={'result_summary': summary},
            confidence_score=0.78,
        )


class DeterministicHistoryModel:
    def extract(self, packet: EvidencePacket) -> MemoryExtractionModelResponse:
        combined_text = _combined_text(packet)
        reason = '관련 evidence를 바탕으로 업무 맥락과 이유를 검토해야 합니다.'
        title = '회사 히스토리 후보'
        if '이유' in combined_text or 'because' in combined_text.lower():
            title = 'Business reason history captured'
            reason = '고객 데모 일정 또는 업무 제약 때문에 실행 과정이 변경된 것으로 보입니다.'
        return _response(
            packet=packet,
            item_type='history_event',
            title=title,
            summary=reason,
            payload_fields={'reason': reason},
            confidence_score=0.79,
        )


class DeterministicDecisionRecordModel:
    def extract(self, packet: EvidencePacket) -> MemoryExtractionModelResponse:
        combined_text = _combined_text(packet)
        title = '의사결정 후보'
        decision_summary = (
            _first_sentence(combined_text) or '의사결정 후보가 감지되었습니다.'
        )
        if 'Redis' in combined_text and 'PostgreSQL' in combined_text:
            title = 'Redis and PostgreSQL responsibility decision'
            decision_summary = 'Redis는 작업 상태 공유에 사용하고 PostgreSQL은 영구 기록 저장소로 유지하는 결정 후보입니다.'
        elif '결정' in combined_text:
            title = 'Explicit decision cue captured'
        return _response(
            packet=packet,
            item_type='decision_record',
            title=title,
            summary=decision_summary,
            payload_fields={'decision_summary': decision_summary},
            confidence_score=0.84,
        )


class DeterministicTodoModel:
    def extract(self, packet: EvidencePacket) -> MemoryExtractionModelResponse:
        combined_text = _combined_text(packet)
        assignment = _extract_assignment(packet)
        title = assignment.get('title') or '후속 업무 후보'
        priority = 'medium'
        priority_reason = '후속 확인이 필요한 업무 항목입니다.'
        if (
            'TODO' in combined_text
            or '권한' in combined_text
            or '런칭' in combined_text
        ):
            title = 'Verify launch permission readiness'
            priority = 'high'
            priority_reason = (
                '런칭 전 OAuth/권한 검증은 제품 신뢰성과 보안에 직접 영향을 줍니다.'
            )
        if assignment:
            priority = 'high' if assignment.get('due_date') else priority
            priority_reason = (
                assignment.get('task_summary')
                or assignment.get('evidence_sentence')
                or priority_reason
            )
        return _response(
            packet=packet,
            item_type='todo',
            title=title,
            summary=priority_reason,
            payload_fields={
                'priority': priority,
                'priority_reason': priority_reason,
                **assignment,
            },
            confidence_score=0.82,
        )


@dataclass(frozen=True)
class _MemoryExtractionAgent:
    agent_name: str
    prompt_version: str
    model: MemoryExtractionModel
    input_cost_per_1m: float = 0.15
    output_cost_per_1m: float = 0.60

    def run(self, packet: EvidencePacket) -> AgentRunResult:
        model_response = self.model.extract(packet)
        candidate = ReviewCandidate(
            item_type=model_response.item_type,
            title=model_response.title,
            summary=model_response.summary,
            source_links=packet.source_links,
            source_snippets=packet.source_snippets,
            confidence_score=model_response.confidence_score,
            permission_level=packet.strictest_permission,
            uncertainty_reason=model_response.uncertainty_reason,
            payload_fields=model_response.payload_fields,
        )
        candidate.validate_evidence()
        token_usage = TokenUsage(
            input_tokens=model_response.input_tokens,
            output_tokens=model_response.output_tokens,
        )
        cost = estimate_agent_run_cost(
            model_name=model_response.model_name or MEMORY_EXTRACTION_MODEL_NAME,
            token_usage=token_usage,
            input_cost_per_1m=self.input_cost_per_1m,
            output_cost_per_1m=self.output_cost_per_1m,
            cache_hit=False,
        )
        return AgentRunResult(
            agent_name=self.agent_name,
            prompt_version=self.prompt_version,
            candidates=[candidate],
            cost=cost,
            cache_key=build_evidence_cache_key(packet, self.prompt_version),
            model_provider=model_response.model_provider,
            model_reasoning_effort=model_response.model_reasoning_effort,
            route_version=model_response.route_version,
            output_contract_version=model_response.output_contract_version,
        )


class TimelineAgent(_MemoryExtractionAgent):
    def __init__(self, model: MemoryExtractionModel) -> None:
        super().__init__(TIMELINE_AGENT_NAME, TIMELINE_AGENT_PROMPT_VERSION, model)


class HistoryAgent(_MemoryExtractionAgent):
    def __init__(self, model: MemoryExtractionModel) -> None:
        super().__init__(HISTORY_AGENT_NAME, HISTORY_AGENT_PROMPT_VERSION, model)


class DecisionRecordAgent(_MemoryExtractionAgent):
    def __init__(self, model: MemoryExtractionModel) -> None:
        super().__init__(
            DECISION_RECORD_AGENT_NAME, DECISION_RECORD_AGENT_PROMPT_VERSION, model
        )


class TodoAgent(_MemoryExtractionAgent):
    def __init__(self, model: MemoryExtractionModel) -> None:
        super().__init__(TODO_AGENT_NAME, TODO_AGENT_PROMPT_VERSION, model)


@dataclass(frozen=True)
class ValidationAgent:
    min_confidence: float = 0.7

    def accept(self, candidate: ReviewCandidate) -> bool:
        try:
            candidate.validate_evidence()
        except ValueError:
            return False
        if candidate.confidence_score < self.min_confidence:
            return False
        if candidate.item_type == 'decision_record':
            return bool(
                candidate.payload_fields.get('decision_summary') or candidate.summary
            )
        if candidate.item_type == 'timeline_event':
            return bool(
                candidate.payload_fields.get('result_summary') or candidate.summary
            )
        if candidate.item_type == 'history_event':
            return bool(candidate.payload_fields.get('reason') or candidate.summary)
        if candidate.item_type == 'todo':
            return bool(
                candidate.payload_fields.get('priority')
                and candidate.payload_fields.get('priority_reason')
            )
        return False


def _response(
    *,
    packet: EvidencePacket,
    item_type: str,
    title: str,
    summary: str,
    payload_fields: dict[str, str],
    confidence_score: float,
) -> MemoryExtractionModelResponse:
    combined_text = _combined_text(packet)
    return MemoryExtractionModelResponse(
        title=title,
        summary=summary,
        item_type=item_type,
        confidence_score=confidence_score,
        input_tokens=max(1, len(combined_text) // 4),
        output_tokens=max(32, len(summary) // 4),
        payload_fields=payload_fields,
    )


def _extract_assignment(packet: EvidencePacket) -> dict[str, str]:
    for message in packet.messages:
        text = message.text.strip()
        if not _looks_like_work_assignment(text):
            continue
        sentence = _assignment_sentence(text)
        task_summary = _extract_task_summary(text, sentence)
        fields = {
            'title': (_extract_subject(text) or task_summary or '업무 지시 후보')[:200],
            'task_summary': (task_summary or sentence or text[:160]).strip()[:500],
            'evidence_sentence': (sentence or message.source_snippet).strip()[:500],
            'evidence_reason': '담당자, 기한, 요청/검토/준비 같은 업무 지시 표현이 원문에 포함되어 있습니다.',
            'source_type': str(
                message.metadata.get('source_type') or packet.source_type
            ),
        }
        assignee = _extract_assignee(text)
        due_date = _extract_due_date(text, message.metadata)
        project_tag = _extract_project_tag(text)
        if assignee:
            fields['assignee'] = assignee
        if due_date:
            fields['due_date'] = due_date
        if project_tag:
            fields['project_tag'] = project_tag
        return fields
    return {}


def _looks_like_work_assignment(text: str) -> bool:
    lowered = text.lower()
    return any(
        cue in lowered
        for cue in (
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
    )


def _assignment_sentence(text: str) -> str:
    lines = [
        line.strip()
        for line in re.split(r'[\n\r]+', text)
        if line.strip() and not line.strip().lower().startswith('subject:')
    ]
    for line in lines:
        if _looks_like_work_assignment(line):
            return line
    sentences = [
        part.strip() for part in re.split(r'(?<=[.!?。])\s+', text) if part.strip()
    ]
    for sentence in sentences:
        if _looks_like_work_assignment(sentence):
            return sentence
    return lines[0] if lines else ''


def _extract_subject(text: str) -> str:
    match = re.search(r'(?im)^subject:\s*(.+)$', text)
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
    return (
        re.sub(r'(?im)^subject:\s*.+$', '', fallback_sentence).strip()
        or fallback_sentence
    )


def _extract_project_tag(text: str) -> str:
    match = re.search(
        r'(프로젝트\s*[A-Za-z0-9가-힣_-]+|Project\s+[A-Za-z0-9가-힣_-]+)',
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ''


def _combined_text(packet: EvidencePacket) -> str:
    return '\n'.join(message.text for message in packet.messages)


def _first_sentence(text: str) -> str:
    return text.strip().split('\n', maxsplit=1)[0][:240]
