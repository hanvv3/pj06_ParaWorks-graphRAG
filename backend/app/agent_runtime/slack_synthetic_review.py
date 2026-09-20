"""Internal local synthetic Slack entry into the existing C.5 Review pipeline.

No API route or connector factory selects this catalog. The public Review source
DTO and its five default agents are unchanged. Approval is always a separate
authorized human transition, with the same evidence binding and provenance.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime import (
    AgentWorkflowState,
    LangChainInvocationPayload,
    build_agent_workflow,
)
from backend.app.agent_runtime.canonical_sources import resolve_source_versions
from backend.app.agent_runtime.review_v2_agents import (
    ReviewAgentCatalog,
    _ReviewAgentAdapter,
)
from backend.app.agent_runtime.review_v2_drafting import ReviewDraftService
from backend.app.agent_runtime.review_v2_preflight import (
    build_prepared_review_identity,
    create_or_reuse_review_thread,
)
from backend.app.agents.slack_agent import (
    SLACK_AGENT_MANIFEST,
    SlackAgent,
    SlackAgentModel,
)
from backend.app.agents.slack_agent.llm import render_slack_llm_prompt
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_authority import resolve_exact_source_authority
from backend.app.ingestion.source_versions import source_version_refs
from backend.app.models import ReviewItem, Source


class _LocalSyntheticSlackReviewCatalog(ReviewAgentCatalog):
    approved_manifests = {'slack_agent': SLACK_AGENT_MANIFEST}
    ordered_names = ('slack_agent',)


@dataclass(frozen=True)
class _LocalCreationIdentity:
    client_request_id: str | None = None


def create_local_slack_review_items(
    *,
    db: Session,
    model: SlackAgentModel,
    user: DemoUser,
    source_ids: list[str] | None = None,
    settings: Settings | None = None,
) -> list[ReviewItem]:
    settings = settings or get_settings()
    sources = _current_local_sources(
        db,
        source_ids=source_ids,
        user=user,
        limit=settings.agent_llm_max_evidence_messages,
    )
    if not sources:
        return []
    resolved = resolve_source_versions(
        db, refs=tuple(source_version_refs(sources)), actor=user, settings=settings
    )
    prepared = build_prepared_review_identity(
        source_refs=resolved, agent_names=('slack_agent',), settings=settings
    )
    catalog = _LocalSyntheticSlackReviewCatalog(
        [
            _ReviewAgentAdapter(
                manifest=SLACK_AGENT_MANIFEST,
                allowed_item_types=frozenset(
                    {'history_event', 'timeline_event', 'todo', 'decision_record'}
                ),
                agent=SlackAgent(model=model),
                estimated_output_tokens=512,
                model_name='local-synthetic-slack',
                model_route_version='local-synthetic-slack:v1',
                input_cost_per_1m=settings.agent_llm_input_cost_per_1m_tokens,
                output_cost_per_1m=settings.agent_llm_output_cost_per_1m_tokens,
                max_cost_usd=settings.agent_llm_max_estimated_cost_usd,
                deterministic=True,
                invocation_renderer=lambda packet: LangChainInvocationPayload(
                    (('user', render_slack_llm_prompt(packet)),)
                ),
            )
        ]
    )
    thread = create_or_reuse_review_thread(
        db,
        prepared=prepared,
        request=_LocalCreationIdentity(),
        actor=user,
        settings=settings,
    ).thread
    thread_id = thread.thread_id
    existing_ids = set(
        db.scalars(
            select(ReviewItem.id).where(ReviewItem.workflow_thread_id == thread_id)
        )
    )
    db.rollback()
    service = ReviewDraftService(
        session_factory=sessionmaker(bind=db.get_bind(), expire_on_commit=False),
        catalog=catalog,
        settings=settings,
    )

    def draft_node(state):
        result = service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id=user.id,
            allowed_permission_levels=tuple(user.permission_levels),
        )
        return state.complete_node(
            'draft_slack', review_item_ids=list(result.review_item_ids)
        )

    # The existing graph runtime remains responsible for graph execution.
    graph = build_agent_workflow((('draft_slack', draft_node),))
    graph.run(AgentWorkflowState(objective='local_synthetic_slack_review', inputs={}))
    db.expire_all()
    return list(
        db.scalars(
            select(ReviewItem).where(
                ReviewItem.workflow_thread_id == thread_id,
                ReviewItem.id.not_in(existing_ids),
            )
        )
    )


def _current_local_sources(db, *, source_ids, user, limit):
    query = select(Source).where(
        Source.source_type == 'slack',
        Source.permission_level.in_(user.permission_levels),
    )
    if source_ids is not None:
        query = query.where(Source.source_id.in_(source_ids))
    # A small explicit source window. Parent context consumes this same budget.
    selected = []
    for source in db.scalars(query.order_by(Source.id.desc()).limit(max(limit, 0))):
        metadata = source.raw_metadata or {}
        if metadata.get('fixture_origin') != 'local-synthetic-slack:v1':
            continue
        if resolve_exact_source_authority(db, source=source) is None:
            continue
        selected.append(source)
    by_id = {source.source_id: source for source in selected}
    for source in list(selected):
        metadata = source.raw_metadata or {}
        parent_ts = metadata.get('parent_ts')
        if not parent_ts or len(by_id) >= limit:
            continue
        parent_id = f'{metadata.get("channel_id")}:{parent_ts}'
        parent = db.scalar(select(Source).where(Source.source_id == parent_id))
        if (
            parent is not None
            and parent.permission_level in user.permission_levels
            and parent.raw_metadata.get('workspace_url')
            == metadata.get('workspace_url')
            and parent.raw_metadata.get('fixture_origin') == 'local-synthetic-slack:v1'
            and resolve_exact_source_authority(db, source=parent) is not None
        ):
            by_id[parent_id] = parent
    return list(by_id.values())
