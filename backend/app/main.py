from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import AutoReviewKeyBootstrapService
from backend.app.agent_runtime.checkpointing import (
    CheckpointRuntime,
    build_checkpoint_runtime,
)
from backend.app.agent_runtime.graph_versions import (
    GraphVersionRegistry,
    register_company_memory_review_v2,
)
from backend.app.agent_runtime.model_router import ReviewModelUnavailableError
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_agents import (
    APPROVED_REVIEW_AGENT_MANIFESTS,
    build_review_agent_catalog,
)
from backend.app.agent_runtime.review_v2_drafting import (
    ReviewDraftError,
    ReviewDraftService,
)
from backend.app.agent_runtime.review_v2_service import (
    ReviewModelReadiness,
    ReviewWorkflowService,
)
from backend.app.api.v1.router import api_router
from backend.app.core.config import Settings, get_settings
from backend.app.db.session import SessionLocal
from backend.app.models.agent_workflows import AgentWorkflowThread
from backend.app.review.auto_review_source_reconciliation import (
    SourceReconciliationResult,
    build_source_reconciliation_service,
)
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    DEFAULT_REVIEW_AGENT_NAMES,
)

CheckpointRuntimeFactory = Callable[[Settings], CheckpointRuntime]
WorkflowSessionFactory = Callable[[], Session]

_NONTERMINAL_REVIEW_THREAD_STATUSES = (
    'created',
    'drafting',
    'checkpoint_pending',
    'awaiting_human_review',
    'resuming',
    'checkpoint_failed',
)


class _UnavailableReviewDraftService:
    @staticmethod
    def preview_prepared(**_kwargs):
        raise ReviewDraftError(
            'model_unavailable',
            'review model is unavailable',
        )

    @staticmethod
    def draft(**_kwargs):
        raise ReviewDraftError(
            'model_unavailable',
            'review model is unavailable',
        )


def create_app(
    *,
    checkpoint_runtime_factory: CheckpointRuntimeFactory = build_checkpoint_runtime,
    workflow_session_factory: WorkflowSessionFactory = SessionLocal,
) -> FastAPI:
    settings = get_settings()
    checkpoint_runtime = checkpoint_runtime_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        graph_registry = GraphVersionRegistry()
        register_company_memory_review_v2(graph_registry)
        preserve_existing_review_threads = (
            not settings.langgraph_review_v2_enabled
            and _has_nonterminal_review_v2_threads(workflow_session_factory)
        )
        checkpoint_runtime.start(
            preserve_existing_review_threads=preserve_existing_review_threads
        )
        try:
            key_bootstrap_service = AutoReviewKeyBootstrapService(
                session_factory=workflow_session_factory,
                settings=settings,
            )
            key_bootstrap_result = key_bootstrap_service.ensure_initialized()
            source_reconciliation = _recover_source_reconciliation_batch(
                workflow_session_factory, settings=settings, limit=100
            )
            try:
                catalog = build_review_agent_catalog(settings)
                agent_registry = catalog.registry
                draft_service = ReviewDraftService(
                    session_factory=workflow_session_factory,
                    catalog=catalog,
                    settings=settings,
                )
                model_readiness = ReviewModelReadiness(ready=True)
            except ReviewModelUnavailableError:
                catalog = None
                agent_registry = AgentRegistry()
                for name in DEFAULT_REVIEW_AGENT_NAMES:
                    agent_registry.register(APPROVED_REVIEW_AGENT_MANIFESTS[name])
                draft_service = _UnavailableReviewDraftService()
                model_readiness = ReviewModelReadiness(
                    ready=False,
                    error_code='model_unavailable',
                )
            review_workflow_service = ReviewWorkflowService(
                session_factory=workflow_session_factory,
                settings=settings,
                checkpoint_runtime=checkpoint_runtime,
                graph_registry=graph_registry,
                agent_registry=agent_registry,
                draft_service=draft_service,
                model_readiness=model_readiness,
            )
            app.state.agent_checkpoint_runtime = checkpoint_runtime
            app.state.agent_graph_registry = graph_registry
            app.state.review_agent_catalog = catalog
            app.state.review_agent_registry = agent_registry
            app.state.review_model_readiness = model_readiness
            app.state.review_workflow_service = review_workflow_service
            app.state.auto_review_key_bootstrap = key_bootstrap_result
            app.state.auto_review_source_reconciliation = source_reconciliation
            yield
        finally:
            checkpoint_runtime.close()

    app = FastAPI(title='ParaWorks Harness', lifespan=lifespan)

    @app.get('/health')
    def health() -> dict[str, bool | str]:
        return {'status': 'ok', 'service': 'paraworks', 'demo_mode': settings.paraworks_demo_mode}

    app.include_router(api_router)
    return app


def _has_nonterminal_review_v2_threads(
    session_factory: WorkflowSessionFactory,
) -> bool:
    try:
        with session_factory() as db:
            existing = db.scalar(
                select(AgentWorkflowThread.thread_id).where(
                    AgentWorkflowThread.workflow_name
                    == COMPANY_MEMORY_REVIEW_WORKFLOW,
                    AgentWorkflowThread.graph_version
                    == COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
                    AgentWorkflowThread.status.in_(
                        _NONTERMINAL_REVIEW_THREAD_STATUSES
                    ),
                )
            )
            db.rollback()
            return existing is not None
    except SQLAlchemyError:
        return False


def _recover_source_reconciliation_batch(
    session_factory: WorkflowSessionFactory,
    *,
    settings: Settings,
    limit: int,
) -> SourceReconciliationResult:
    try:
        with session_factory() as db:
            return build_source_reconciliation_service(
                db, settings=settings
            ).recover_stale_sources(limit=limit)
    except (SQLAlchemyError, ValueError):
        return SourceReconciliationResult(
            failure_count=1,
            remaining_count=1,
            readiness=False,
        )


app = create_app()
