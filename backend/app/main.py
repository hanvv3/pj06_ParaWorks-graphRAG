from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_call_recovery import (
    AutoReviewCallRecoveryResult,
    AutoReviewCallRecoveryService,
)
from backend.app.admin.auto_review_keys import AutoReviewKeyBootstrapService
from backend.app.agent_runtime.auto_review_orchestrator import (
    AutoReviewValidationOrchestrator,
)
from backend.app.agent_runtime.auto_review_validation_store import (
    AutoReviewValidationStore,
)
from backend.app.agent_runtime.auto_review_validator import (
    AutoReviewValidatorFactory,
)
from backend.app.agent_runtime.checkpointing import (
    CheckpointRuntime,
    build_checkpoint_runtime,
)
from backend.app.agent_runtime.graph_versions import (
    GraphVersionRegistry,
    register_company_memory_review_versions,
)
from backend.app.agent_runtime.model_router import ReviewModelUnavailableError
from backend.app.agent_runtime.rag_v2_composition import build_rag_v2_runtime
from backend.app.agent_runtime.rag_v2_registry import (
    RagGraphRegistration,
)
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
from backend.app.agent_runtime.review_v21_service import ReviewV21Service
from backend.app.agent_runtime.review_workflow_facade import (
    DatabaseV21LaunchAuthority,
    ReviewWorkflowFacade,
)
from backend.app.api.v1.router import api_router
from backend.app.core.config import Settings, get_settings
from backend.app.db.session import SessionLocal
from backend.app.models.agent_workflows import AgentWorkflowThread
from backend.app.review.auto_review_quality_revoke import (
    AutoReviewQualityRevokeService,
)
from backend.app.review.auto_review_source_reconciliation import (
    SourceReconciliationResult,
    build_source_reconciliation_service,
)
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    DEFAULT_REVIEW_AGENT_NAMES,
)

CheckpointRuntimeFactory = Callable[[Settings], CheckpointRuntime]
WorkflowSessionFactory = Callable[[], Session]
RagGraphFactory = Callable[[], RagGraphRegistration]

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
    rag_graph_factory: RagGraphFactory | None = None,
) -> FastAPI:
    settings = get_settings()
    checkpoint_runtime = checkpoint_runtime_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        graph_registry = GraphVersionRegistry()
        register_company_memory_review_versions(graph_registry)
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
            rag_shadow_recovered_count = _recover_rag_shadow_batch(
                workflow_session_factory,
                settings=settings,
                limit=100,
            )
            rag_runtime = build_rag_v2_runtime(
                settings=settings,
                session_factory=workflow_session_factory,
                registration=rag_graph_factory() if rag_graph_factory else None,
            )
            validation_store = AutoReviewValidationStore(
                session_factory=workflow_session_factory,
                settings=settings,
            )
            validation_orchestrator = AutoReviewValidationOrchestrator(
                store=validation_store,
                validator_factory=AutoReviewValidatorFactory(
                    settings=settings,
                    dispatcher=validation_store,
                ),
            )
            source_reconciliation = _recover_source_reconciliation_batch(
                workflow_session_factory, settings=settings, limit=100
            )
            call_recovery = _recover_auto_review_calls_batch(
                workflow_session_factory, limit=100
            )
            quality_remediation_recovered = _recover_quality_remediation_batch(
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
                v21_launch_authority=(
                    DatabaseV21LaunchAuthority(settings=settings)
                    if settings.auto_review_mode != 'disabled'
                    else None
                ),
            )
            review_v21_service = ReviewV21Service(
                launch_service=review_workflow_service,
                session_factory=workflow_session_factory,
                settings=settings,
                checkpoint_runtime=checkpoint_runtime,
                graph_registry=graph_registry,
                agent_registry=agent_registry,
            )
            review_workflow_facade = ReviewWorkflowFacade(
                session_factory=workflow_session_factory,
                settings=settings,
                v20=review_workflow_service,
                v21=review_v21_service,
            )
            app.state.agent_checkpoint_runtime = checkpoint_runtime
            app.state.agent_graph_registry = graph_registry
            app.state.rag_graph_registry = rag_runtime.graph_registry
            app.state.agent_manifest_registry = rag_runtime.manifest_registry
            app.state.rag_application_facade = rag_runtime.facade
            app.state.rag_shadow_recovered_count = rag_shadow_recovered_count
            app.state.review_agent_catalog = catalog
            app.state.review_agent_registry = agent_registry
            app.state.review_model_readiness = model_readiness
            app.state.review_workflow_service = review_workflow_facade
            app.state.review_workflow_v20_service = review_workflow_service
            app.state.review_workflow_v21_service = review_v21_service
            app.state.auto_review_key_bootstrap = key_bootstrap_result
            app.state.auto_review_validation_store = validation_store
            app.state.auto_review_validation_orchestrator = validation_orchestrator
            app.state.auto_review_source_reconciliation = source_reconciliation
            app.state.auto_review_call_recovery = call_recovery
            app.state.auto_review_quality_remediation_recovered = (
                quality_remediation_recovered
            )
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
                    AgentWorkflowThread.graph_version.in_((
                        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
                        COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
                    )),
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


def _recover_quality_remediation_batch(
    session_factory: WorkflowSessionFactory,
    *,
    settings: Settings,
    limit: int,
) -> int:
    try:
        with session_factory() as db:
            return AutoReviewQualityRevokeService(
                db, settings=settings
            ).recover_pending_remediation(limit=limit)
    except (SQLAlchemyError, ValueError):
        return 0


def _recover_auto_review_calls_batch(
    session_factory: WorkflowSessionFactory,
    *,
    limit: int,
) -> AutoReviewCallRecoveryResult:
    try:
        return AutoReviewCallRecoveryService(
            session_factory=session_factory,
        ).recover(limit=limit)
    except (SQLAlchemyError, ValueError):
        return AutoReviewCallRecoveryResult(
            extraction_remaining=1,
            validation_remaining=1,
            failure_count=1,
        )


def _recover_rag_shadow_batch(
    session_factory: WorkflowSessionFactory,
    *,
    settings: Settings,
    limit: int,
) -> int | None:
    """Recover only durable paid shadow projection owners; never call providers."""
    from backend.app.agent_runtime.rag_cost_ledger import RagCostLedgerError
    from backend.app.agent_runtime.rag_provider_transport import (
        RagProviderTransportError,
        _assemble_rag_request_cost_authority,
    )

    try:
        with session_factory() as db:
            if db.get_bind().dialect.name != 'postgresql':
                return 0
            assembly = _assemble_rag_request_cost_authority(
                settings=settings,
                session=db,
            )
            recovered = assembly.store.recover_incomplete_shadow_runs(limit=limit)
            return len(recovered)
    except (SQLAlchemyError, RagCostLedgerError, RagProviderTransportError):
        # Pending rows remain fail-closed and provider-ineligible. ``None`` is
        # retained in app state so an unavailable recovery is never reported as
        # a successful zero-row scan.
        return None


app = create_app()
