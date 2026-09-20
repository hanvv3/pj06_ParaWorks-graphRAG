import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime import (
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
    ReviewCandidate,
)
from backend.app.agents.slack_agent.agent import SlackAgent
from backend.app.agents.slack_agent.quality import classify_slack_work_signal
from backend.app.models import AgentRun, DocumentChunk, ReviewItem, Source

_DECISION_KEYWORDS = (
    '결정',
    '합의',
    '선택',
    '확정',
    '도입',
    '사용합니다',
    'decision',
    'decided',
    'agreed',
)
_ACTION_KEYWORDS = (
    'todo',
    '해야',
    '진행',
    '확인',
    '배포',
    '테스트',
    '마감',
    'action',
    'follow up',
    'next',
)
_TECH_COST_KEYWORDS = (
    '비용',
    '예산',
    'api',
    'llm',
    'rag',
    'pgvector',
    'postgres',
    'slack',
    'oauth',
    'gemini',
    'openai',
    'langgraph',
)
_LOW_SIGNAL_KEYWORDS = ('넵', '네', '좋아요', '확인', '굿', '감사')
_PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}

def create_slack_agent_review_items(
    *,
    db: Session,
    agent: SlackAgent,
    permission_context: PermissionContext,
    source_window: str,
    max_messages: int | None = None,
    newest_first: bool = False,
    selection_strategy: str = 'chronological',
    source_ids: list[str] | None = None,
) -> list[ReviewItem]:
    packet = build_slack_evidence_packet(
        db=db,
        permission_context=permission_context,
        source_window=source_window,
        max_messages=max_messages,
        newest_first=newest_first,
        selection_strategy=selection_strategy,
        source_ids=source_ids,
    )
    if not packet.messages:
        return []

    result = agent.run(packet)
    agent_run = AgentRun(
        agent_name=result.agent_name,
        prompt_version=result.prompt_version,
        status='complete',
        source_window=packet.source_window,
        cache_key=result.cache_key,
        model_name=result.cost.model_name,
        input_tokens=result.cost.token_usage.input_tokens,
        output_tokens=result.cost.token_usage.output_tokens,
        total_tokens=result.cost.token_usage.total_tokens,
        estimated_cost_usd=result.cost.estimated_cost_usd,
        permission_level=packet.strictest_permission,
        metadata_={
            'source_type': packet.source_type,
            'message_count': len(packet.messages),
            'cache_hit': result.cost.cache_hit,
            'selection_strategy': selection_strategy,
            'evidence_summary': _evidence_summary(packet),
        },
    )
    db.add(agent_run)
    db.flush()

    review_items: list[ReviewItem] = []

    for candidate in result.candidates:
        candidate.validate_evidence()

        # Phase 2: ?숈쟻 ?쒓렇 ?꾪뙆 (Back-propagation)
        back_propagate_slack_tags(db, candidate)

        topic_tag = candidate.payload_fields.get('topic_tag', 'N/A')
        project_key, is_new_project = _determine_project_from_tag(topic_tag, candidate.summary)

        payload = {
            'title': candidate.title,
            'summary': candidate.summary,
            'category': candidate.payload_fields.get('category', 'Ad-hoc'),
            'topic_tag': topic_tag,
            'importance': candidate.payload_fields.get('importance', 'Medium'),
            'assignee': candidate.payload_fields.get('assignee'),
            'due_date': candidate.payload_fields.get('due_date'),
            'project_key': project_key,
            'is_new_project': is_new_project,
            'agent_name': result.agent_name,
            'agent_run_id': agent_run.id,
            'prompt_version': result.prompt_version,
            'cache_key': result.cache_key,
            'estimated_cost_usd': result.cost.estimated_cost_usd,
            'token_usage': {
                'input_tokens': result.cost.token_usage.input_tokens,
                'output_tokens': result.cost.token_usage.output_tokens,
                'total_tokens': result.cost.token_usage.total_tokens,
            },
            'uncertainty_reason': candidate.uncertainty_reason,
        }

        review_item = ReviewItem(
            status='pending_review',
            item_type=candidate.item_type,
            payload=payload,
            source_links=candidate.source_links,
            source_snippets=candidate.source_snippets,
            confidence_score=candidate.confidence_score,
            permission_level=candidate.permission_level,
        )
        db.add(review_item)
        review_items.append(review_item)

    db.commit()
    for review_item in review_items:
        db.refresh(review_item)

    return review_items

def _determine_project_from_tag(topic_tag: str, summary: str) -> tuple[str | None, bool]:
    searchable = f'{topic_tag} {summary}'.strip()
    if not searchable or topic_tag in {'N/A', 'None', 'Ad-hoc', 'ad-hoc', '미정'}:
        return 'ad-hoc', False

    dynamic_key = re.sub(r'[^a-z0-9가-힣]+', '-', searchable.lower()).strip('-')
    if dynamic_key:
        return f'project-{dynamic_key[:48]}', True
    return 'ad-hoc', False

def back_propagate_slack_tags(db: Session, candidate: ReviewCandidate) -> None:
    """異붿텧??吏?앹쓽 移댄뀒怨좊━/?좏뵿 ?뺣낫瑜??먮낯 ?щ옓 硫붿떆吏 泥?겕????쟾?뚰빀?덈떎."""
    source_ids = []
    for url in candidate.source_links:
        # URL?먯꽌 p ?ㅼ쓽 ?レ옄 16?먮━ 異붿텧 (?щ옓 ID 洹쒖튃)
        # ?? https://.../archives/C123/p1715000000000100 -> C123:1715000000.000100
        if '/archives/' in url and '/p' in url:
            parts = url.split('/archives/')[-1].split('/')
            channel_id = parts[0]
            raw_ts = parts[1].split('p')[-1].split('?')[0]
            if len(raw_ts) >= 16:
                formatted_ts = f"{raw_ts[:10]}.{raw_ts[10:]}"
                source_ids.append(f"{channel_id}:{formatted_ts}")

    if not source_ids:
        return

    source_pks = db.scalars(
        select(Source.id).where(Source.source_id.in_(source_ids))
    ).all()

    if source_pks:
        chunks = db.scalars(
            select(DocumentChunk).where(DocumentChunk.source_id.in_(source_pks))
        ).all()
        for chunk in chunks:
            chunk.metadata_['category'] = candidate.payload_fields.get('category')
            chunk.metadata_['topic_tag'] = candidate.payload_fields.get('topic_tag')
            chunk.metadata_['importance'] = candidate.payload_fields.get('importance')


def build_slack_evidence_packet(
    *,
    db: Session,
    permission_context: PermissionContext,
    source_window: str,
    max_messages: int | None = None,
    newest_first: bool = False,
    selection_strategy: str = 'chronological',
    source_ids: list[str] | None = None,
) -> EvidencePacket:
    query = (
        select(DocumentChunk, Source)
        .join(Source, DocumentChunk.source_id == Source.id)
        .where(Source.source_type == 'slack')
        .where(Source.permission_level.in_(permission_context.allowed_permission_levels))
        .where(DocumentChunk.permission_level.in_(permission_context.allowed_permission_levels))
        .order_by(DocumentChunk.id)
    )
    if source_ids is not None:
        query = query.where(Source.source_id.in_(source_ids))
    rows = db.execute(query).all()
    rows = [(row[0], row[1]) for row in rows]
    importance_scores: dict[int, int] = {}

    if selection_strategy == 'ranked':
        ranked_rows = _dedupe_and_rank_slack_rows(rows)
        rows = [(chunk, source) for chunk, source, _score in ranked_rows]
        importance_scores = {chunk.id: score for chunk, _source, score in ranked_rows}
    elif newest_first:
        rows = sorted(rows, key=lambda row: _source_sort_timestamp(row[1]), reverse=True)

    if max_messages is not None:
        rows = rows[:max(max_messages, 0)]

    messages = [
        EvidenceMessage(
            source_id=source.source_id,
            source_url=source.source_url,
            text=chunk.text,
            author=source.author,
            timestamp=str(source.raw_metadata.get('ts') or source.created_at.isoformat()),
            permission_level=max(
                (source.permission_level, chunk.permission_level),
                key=lambda level: _PERMISSION_RANK.get(level, len(_PERMISSION_RANK)),
            ),
            metadata=_slack_message_metadata(
                chunk=chunk,
                source=source,
                evidence_rank=index,
                importance_score=importance_scores.get(chunk.id),
                selection_strategy=selection_strategy,
            ),
        )
        for index, (chunk, source) in enumerate(rows, start=1)
    ]

    # 而⑦뀓?ㅽ듃???꾨줈?앺듃 紐⑸줉 二쇱엯
    from backend.app.models import Project
    from backend.app.projects.classifier import CANONICAL_PROJECTS
    db_projects = db.scalars(select(Project)).all()
    seen_keys = {p.project_key for p in db_projects}
    active_projects = [{'project_key': p.project_key, 'name': p.name, 'summary': p.summary} for p in db_projects]
    for p in CANONICAL_PROJECTS:
        if p.project_key not in seen_keys:
            active_projects.append({'project_key': p.project_key, 'name': p.name, 'summary': p.summary})

    return EvidencePacket(
        source_type='slack',
        source_window=source_window,
        messages=messages,
        permission_context=permission_context,
        context={'projects': active_projects},
    )


def _source_sort_timestamp(source: Source) -> float:
    raw_ts = source.raw_metadata.get('ts') if source.raw_metadata else None
    try:
        return float(raw_ts)
    except (TypeError, ValueError):
        return source.created_at.timestamp()


def _dedupe_and_rank_slack_rows(
    rows: list[tuple[DocumentChunk, Source]],
) -> list[tuple[DocumentChunk, Source, int]]:
    best_by_text: dict[str, tuple[DocumentChunk, Source, int]] = {}
    for chunk, source in rows:
        signal = classify_slack_work_signal(chunk.text)
        if not signal.is_reviewable:
            continue
        dedupe_key = _normalize_evidence_text(chunk.text) or source.source_id
        score = _evidence_importance_score(chunk, source)
        score += signal.score
        current = best_by_text.get(dedupe_key)
        if current is None or _rank_sort_key(chunk, source, score) > _rank_sort_key(*current):
            best_by_text[dedupe_key] = (chunk, source, score)

    return sorted(
        best_by_text.values(),
        key=lambda row: _rank_sort_key(*row),
        reverse=True,
    )


def _rank_sort_key(chunk: DocumentChunk, source: Source, score: int) -> tuple[int, float, int]:
    return (score, _source_sort_timestamp(source), chunk.id)


def _normalize_evidence_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def _evidence_importance_score(chunk: DocumentChunk, source: Source) -> int:
    text = chunk.text.lower()
    metadata = source.raw_metadata or {}
    score = 0

    if any(keyword in text for keyword in _DECISION_KEYWORDS):
        score += 60
    if any(keyword in text for keyword in _ACTION_KEYWORDS):
        score += 35
    if any(keyword in text for keyword in _TECH_COST_KEYWORDS):
        score += 15
    if 40 <= len(chunk.text) <= 1200:
        score += 5
    if metadata.get('thread_ts') and metadata.get('thread_ts') != metadata.get('ts'):
        score += 5
    if metadata.get('reply_count'):
        score += min(int(metadata.get('reply_count') or 0), 5)
    if any(keyword in text for keyword in _LOW_SIGNAL_KEYWORDS) and len(chunk.text) < 80:
        score -= 20

    return score


def _slack_message_metadata(
    *,
    chunk: DocumentChunk,
    source: Source,
    evidence_rank: int,
    importance_score: int | None,
    selection_strategy: str,
) -> dict[str, object]:
    raw_metadata = source.raw_metadata or {}
    metadata: dict[str, object] = {
        'chunk_id': chunk.id,
        'source_pk': source.id,
        'channel_id': raw_metadata.get('channel_id'),
    }
    if selection_strategy == 'ranked':
        metadata.update(
            {
                'selection_strategy': selection_strategy,
                'evidence_rank': evidence_rank,
                'importance_score': importance_score or 0,
                'thread_ts': raw_metadata.get('thread_ts'),
                'is_thread_reply': bool(raw_metadata.get('thread_ts') and raw_metadata.get('thread_ts') != raw_metadata.get('ts')),
                'is_thread_parent': bool(raw_metadata.get('reply_count')),
            }
        )
    return metadata


def _evidence_summary(packet: EvidencePacket) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for index, message in enumerate(packet.messages, start=1):
        rank = message.metadata.get('evidence_rank') or index
        summary.append(
            {
                'rank': rank,
                'source_id': message.source_id,
                'source_url': message.source_url,
                'timestamp': message.timestamp,
                'author': message.author,
                'permission_level': message.permission_level,
                'channel_id': message.metadata.get('channel_id'),
                'importance_score': message.metadata.get('importance_score', 0),
                'snippet': message.text[:240],
            }
        )
    return summary
