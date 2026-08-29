import re
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime import AgentRegistry, EvidencePacket, PermissionContext
from backend.app.agent_runtime.canonical_sources import ReviewWorkflowPreflightError
from backend.app.agent_runtime.review_v2_preflight import (
    V21PreparedReviewConfig,
    find_matching_review_thread,
    prepare_review_request,
)
from backend.app.agents.mail_document_agent import (
    MAIL_DOCUMENT_AGENT_MANIFEST,
    MAIL_DOCUMENT_SOURCE_TYPES,
    DeterministicMailDocumentAgentModel,
    MailDocumentAgent,
    MailDocumentLlmProviderError,
    MailDocumentLlmSettings,
    build_langchain_mail_document_agent_model,
    build_mail_document_evidence_packet,
    build_mail_document_llm_preflight,
    create_mail_document_agent_review_items_for_changed_sources,
)
from backend.app.agents.memory_extraction_agent import (
    DECISION_RECORD_AGENT_MANIFEST,
    HISTORY_AGENT_MANIFEST,
    TIMELINE_AGENT_MANIFEST,
    TODO_AGENT_MANIFEST,
    build_memory_extraction_agent_preflight,
)
from backend.app.agents.slack_agent import (
    DeterministicSlackAgentModel,
    SlackAgent,
    SlackLlmProviderError,
    SlackLlmSettings,
    build_langchain_slack_agent_model,
    build_slack_evidence_packet,
    build_slack_llm_preflight,
    create_slack_agent_review_items,
)
from backend.app.agents.slack_agent.sync_service import trigger_slack_agent_analysis
from backend.app.connectors.factory import (
    ConnectorNotConfiguredError,
    get_sync_connector,
)
from backend.app.connectors.google_oauth import (
    GOOGLE_OAUTH_CONNECTOR_TYPES,
    GoogleOAuthConfigurationError,
    GoogleOAuthError,
    GoogleOAuthStateSigner,
    build_google_oauth_install_url,
    complete_google_oauth_callback,
)
from backend.app.connectors.mock import CONNECTOR_TYPES
from backend.app.connectors.registry import list_connector_manifests
from backend.app.connectors.slack import SlackApiError
from backend.app.connectors.slack_oauth import (
    LOCAL_TOKEN_VAULT,
    SlackOAuthConfigurationError,
    build_slack_oauth_install_url,
    complete_slack_direct_connect,
    complete_slack_oauth_callback,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.core.redaction import redact_secret_text
from backend.app.db.session import get_db
from backend.app.ingestion.source_versions import (
    ReviewBatchMode,
    SourceVersionRef,
    source_version_refs,
    with_review_batch_marker,
)
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import (
    DocumentChunk,
    IntegrationConnection,
    ReviewItem,
    Source,
    SyncJob,
)
from backend.app.projects.classifier import create_project_assignment_review_items
from backend.app.review.evidence_visibility import (
    ReviewEvidenceNotFound,
    ReviewEvidenceVisibilityService,
)
from backend.app.schemas.review_workflow import (
    DEFAULT_REVIEW_AGENT_NAMES,
    ReviewWorkflowRunRequest,
)
from backend.app.services.audit import record_audit_log

router = APIRouter(prefix='/integrations', tags=['integrations'])
DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]


class IntegrationSyncRequest(BaseModel):
    selected_channel_ids: list[str] | None = None
    run_async: bool = False


class SlackLlmRunRequest(BaseModel):
    confirm_paid_run: bool = False


SYNC_REQUEST_BODY = Body(default=None)


@router.get('')
def list_integrations(settings: AppSettings) -> list[dict[str, object]]:
    return [
        {
            'type': manifest.connector_type,
            'display_name': manifest.display_name,
            'mode': manifest.mode,
            'status': 'ready',
            'auth_type': manifest.auth_type,
            'required_scopes': list(manifest.required_scopes),
            'sync_strategy': manifest.sync_strategy,
            'cost_policy': manifest.cost_policy,
        }
        for manifest in list_connector_manifests(demo_mode=settings.paraworks_demo_mode)
    ]


@router.get('/connections')
def list_integration_connections(db: DbSession) -> list[dict[str, object]]:
    connections = (
        db.query(IntegrationConnection)
        .order_by(
            IntegrationConnection.connector_type, IntegrationConnection.workspace_name
        )
        .all()
    )
    return [
        {
            'connector_type': connection.connector_type,
            'workspace_id': connection.workspace_id,
            'workspace_name': connection.workspace_name,
            'status': connection.status,
            'credential_status': 'available'
            if LOCAL_TOKEN_VAULT.resolve(connection.token_ref)
            else 'missing',
            'masked_bot_token': connection.masked_bot_token,
            'scopes': connection.scopes,
        }
        for connection in connections
    ]


@router.get('/slack/runtime-status')
def get_slack_runtime_status(
    db: DbSession,
    settings: AppSettings,
    user: CurrentUser,
) -> dict[str, object]:
    connection = db.scalar(
        select(IntegrationConnection)
        .where(IntegrationConnection.connector_type == 'slack')
        .order_by(IntegrationConnection.id.desc())
    )
    latest_sync = db.scalar(
        select(SyncJob)
        .where(SyncJob.connector_type == 'slack')
        .order_by(SyncJob.id.desc())
    )
    credential_status = (
        'available'
        if connection and LOCAL_TOKEN_VAULT.resolve(connection.token_ref)
        else 'missing'
    )

    return {
        'connector_type': 'slack',
        'mode': 'mock' if settings.paraworks_demo_mode else 'live',
        'configured_channel_ids': _configured_channel_ids(settings.slack_channel_ids),
        'selected_channel_ids': _configured_channel_ids(settings.slack_channel_ids),
        'channel_options': _slack_channel_options(settings.slack_channel_ids),
        'connection_status': connection.status if connection else 'disconnected',
        'credential_status': credential_status,
        'latest_sync': _sync_job_response(db=db, user=user, job=latest_sync),
        'latest_sync_summary': _sync_job_summary(latest_sync),
        'last_error': _sync_error_response(latest_sync),
        'agent_bridge': _slack_agent_bridge(db, user),
        'cost_policy': {
            'status_lookup_triggers_sync': False,
            'status_lookup_triggers_llm': False,
            'thread_reply_fetch_is_incremental': True,
        },
    }


@router.get('/{connector_type}/runtime-status')
def get_google_runtime_status(
    connector_type: str,
    db: DbSession,
    settings: AppSettings,
    user: CurrentUser,
) -> dict[str, object]:
    if connector_type not in GOOGLE_OAUTH_CONNECTOR_TYPES:
        raise HTTPException(status_code=404, detail='Connector not found')

    connection = db.scalar(
        select(IntegrationConnection)
        .where(IntegrationConnection.connector_type == connector_type)
        .order_by(IntegrationConnection.id.desc())
    )
    latest_sync = db.scalar(
        select(SyncJob)
        .where(SyncJob.connector_type == connector_type)
        .order_by(SyncJob.id.desc())
    )
    credential_status = (
        'available'
        if connection and LOCAL_TOKEN_VAULT.resolve(connection.token_ref)
        else 'missing'
    )

    return {
        'connector_type': connector_type,
        'mode': 'mock' if settings.paraworks_demo_mode else 'live',
        'connection_status': connection.status if connection else 'disconnected',
        'credential_status': credential_status,
        'account_name': connection.workspace_name if connection else None,
        'latest_sync': _sync_job_response(db=db, user=user, job=latest_sync),
        'cost_policy': {
            'status_lookup_triggers_sync': False,
            'status_lookup_triggers_llm': False,
        },
    }


@router.post('/{connector_type}/sync')
def sync_connector(
    connector_type: str,
    background_tasks: BackgroundTasks,
    db: DbSession,
    settings: AppSettings,
    user: CurrentUser,
    request: IntegrationSyncRequest | None = SYNC_REQUEST_BODY,
) -> dict[str, object]:
    if connector_type not in CONNECTOR_TYPES:
        raise HTTPException(status_code=404, detail='Connector not found')

    selected_channel_ids = (
        _clean_channel_ids(request.selected_channel_ids)
        if request is not None
        and request.selected_channel_ids is not None
        and connector_type == 'slack'
        else None
    )

    if request is not None and request.run_async:
        job = _create_queued_sync_job(db=db, connector_type=connector_type)
        pending_review_count = _pending_review_count(db, user)
        background_tasks.add_task(
            _run_connector_sync_background,
            db=db,
            settings=settings,
            user=user,
            job_id=job.job_id,
            connector_type=connector_type,
            selected_channel_ids=selected_channel_ids,
        )
        return {
            'job_id': job.job_id,
            'connector_type': connector_type,
            'status': job.status,
            'created_review_items': 0,
            'fetched_events': 0,
            'skipped_events': 0,
            'parser_status_counts': {},
            'changed_source_ids': [],
            'changed_source_refs': [],
            'agent_generated_items': 0,
            'project_assignment_items': 0,
            'pending_review_count': pending_review_count,
        }

    try:
        return _perform_connector_sync(
            db=db,
            user=user,
            settings=settings,
            connector_type=connector_type,
            selected_channel_ids=selected_channel_ids,
        )
    except ConnectorNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MailDocumentLlmProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except SlackApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _perform_connector_sync(
    *,
    db: Session,
    user: DemoUser,
    settings: Settings,
    connector_type: str,
    selected_channel_ids: list[str] | None,
    job_id: str | None = None,
) -> dict[str, object]:
    agent_review_items = 0
    project_assignment_items = 0
    connector = get_sync_connector(
        connector_type,
        settings,
        db=db,
        slack_channel_ids_override=selected_channel_ids,
    )
    v2_mode = _uses_explicit_review_mode(
        connector_type=connector_type,
        settings=settings,
    )
    if v2_mode:
        result = sync_connector_events(
            db=db,
            connector=connector,
            job_id=job_id,
            review_batch_mode='v2_explicit',
        )
    else:
        result = sync_connector_events(db=db, connector=connector, job_id=job_id)
    changed_source_ids = getattr(result, 'changed_source_ids', [])
    changed_source_refs = _visible_source_refs(
        db=db,
        user=user,
        refs=getattr(result, 'changed_source_refs', []),
    )
    if result.status == 'complete' and not v2_mode:
        _mark_sync_job_agent_review_running(
            db=db,
            job_id=result.job_id,
            fetched_events=result.fetched_events,
            skipped_events=result.skipped_events,
        )
    legacy_generation_succeeded = False
    if result.status == 'complete' and not v2_mode and not changed_source_ids:
        recovery_source_ids = _connector_source_ids_for_review(
            db=db,
            connector_type=connector_type,
            user=user,
        )
        suppress_legacy = _legacy_batch_owned_by_v2(
            db=db,
            user=user,
            settings=settings,
            connector_type=connector_type,
            source_ids=recovery_source_ids,
        )
        if (
            not suppress_legacy
            and recovery_source_ids
            and not _has_connector_agent_review_items(
                db=db,
                connector_type=connector_type,
            )
        ):
            agent_review_items = _run_connector_agent_review(
                db=db,
                user=user,
                settings=settings,
                connector_type=connector_type,
                source_ids=recovery_source_ids,
            )
            legacy_generation_succeeded = True
        if not suppress_legacy and not _skip_project_assignment_after_agent_review(
            connector_type=connector_type,
            settings=settings,
            agent_review_items=agent_review_items,
        ):
            project_assignment_items = len(create_project_assignment_review_items(db))
            legacy_generation_succeeded = True
        if legacy_generation_succeeded and connector_type in GOOGLE_OAUTH_CONNECTOR_TYPES:
            _mark_review_batch_sources(
                db=db,
                source_ids=recovery_source_ids,
                mode='legacy_inline',
            )

    if result.status == 'complete' and not v2_mode and changed_source_ids:
        suppress_legacy = _legacy_batch_owned_by_v2(
            db=db,
            user=user,
            settings=settings,
            connector_type=connector_type,
            source_ids=changed_source_ids,
        )
        if not suppress_legacy:
            agent_review_items = _run_connector_agent_review(
                db=db,
                user=user,
                settings=settings,
                connector_type=connector_type,
                source_ids=changed_source_ids,
            )
            if not _skip_project_assignment_after_agent_review(
                connector_type=connector_type,
                settings=settings,
                agent_review_items=agent_review_items,
            ):
                project_assignment_items = len(create_project_assignment_review_items(db))
            if connector_type in GOOGLE_OAUTH_CONNECTOR_TYPES:
                _mark_review_batch_sources(
                    db=db,
                    source_ids=changed_source_ids,
                    mode='legacy_inline',
                )
    parser_status_counts = getattr(result, 'parser_status_counts', {})

    total_review_items = (
        result.created_review_items + agent_review_items + project_assignment_items
    )
    db.flush()
    pending_review_count = _pending_review_count(db, user)
    sync_job = db.scalar(select(SyncJob).where(SyncJob.job_id == result.job_id))
    if sync_job is not None:
        sync_job.status = result.status
        sync_job.progress_pct = 100
        sync_job.message = (
            f'fetched={result.fetched_events} '
            f'created_review_items={total_review_items} '
            f'skipped_events={result.skipped_events} '
            f'pending_review_items={pending_review_count}'
        )

    audit_metadata = {
        'job_id': result.job_id,
        'fetched_events': result.fetched_events,
        'created_review_items': total_review_items,
        'skipped_events': result.skipped_events,
        'parser_status_counts': parser_status_counts,
        'selected_channel_ids': selected_channel_ids,
        'agent_generated_items': agent_review_items,
        'project_assignment_items': project_assignment_items,
        'pending_review_count': pending_review_count,
    }
    audit_refs = _complete_visible_source_refs_for_ids(
        db=db,
        user=user,
        source_ids=changed_source_ids,
    )
    audit_metadata.update(
        changed_source_count=len(set(changed_source_ids)),
        review_batch_hmac=(
            _review_batch_hmac(
                db=db,
                user=user,
                settings=settings,
                refs=audit_refs,
            )
            if audit_refs
            else None
        ),
    )
    record_audit_log(
        db=db,
        actor=user,
        action='integration.sync',
        target_type='connector',
        target_id=connector_type,
        metadata=audit_metadata,
    )
    db.commit()

    return {
        'job_id': result.job_id,
        'connector_type': connector_type,
        'status': result.status,
        'created_review_items': total_review_items,
        'fetched_events': result.fetched_events,
        'skipped_events': result.skipped_events,
        'parser_status_counts': parser_status_counts,
        'changed_source_ids': changed_source_ids,
        'changed_source_refs': [asdict(ref) for ref in changed_source_refs],
        'agent_generated_items': agent_review_items,
        'project_assignment_items': project_assignment_items,
        'pending_review_count': pending_review_count,
    }


def _create_queued_sync_job(*, db: Session, connector_type: str) -> SyncJob:
    job = SyncJob(
        job_id=f'{connector_type}-{uuid4().hex}',
        connector_type=connector_type,
        status='queued',
        message='queued',
        progress_pct=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _run_connector_sync_background(
    *,
    db: Session,
    settings: Settings,
    user: DemoUser,
    job_id: str,
    connector_type: str,
    selected_channel_ids: list[str] | None,
) -> None:
    try:
        _perform_connector_sync(
            db=db,
            user=user,
            settings=settings,
            connector_type=connector_type,
            selected_channel_ids=selected_channel_ids,
            job_id=job_id,
        )
    except Exception as exc:
        _mark_sync_job_failed(db=db, job_id=job_id, message=f'failed: {exc}')
        record_audit_log(
            db=db,
            actor=user,
            action='integration.sync',
            target_type='connector',
            target_id=connector_type,
            status='failed',
            metadata={
                'job_id': job_id,
                'error': str(exc),
                'selected_channel_ids': selected_channel_ids,
            },
        )
        db.commit()


def _uses_explicit_review_mode(
    *,
    connector_type: str,
    settings: Settings,
) -> bool:
    return (
        connector_type in GOOGLE_OAUTH_CONNECTOR_TYPES
        and settings.langgraph_review_v2_enabled
    )


def _visible_source_refs(
    *,
    db: Session,
    user: DemoUser,
    refs: list[SourceVersionRef],
) -> list[SourceVersionRef]:
    if not refs:
        return []
    requested = set(refs)
    sources = db.scalars(
        select(Source).where(
            Source.source_id.in_([ref.source_id for ref in refs]),
            Source.permission_level.in_(tuple(user.permission_levels)),
        )
    ).all()
    return [ref for ref in source_version_refs(sources) if ref in requested]


def _complete_visible_source_refs_for_ids(
    *,
    db: Session,
    user: DemoUser,
    source_ids: list[str],
) -> list[SourceVersionRef] | None:
    if not source_ids:
        return []
    sources = db.scalars(
        select(Source).where(
            Source.source_id.in_(source_ids),
            Source.permission_level.in_(tuple(user.permission_levels)),
        )
    ).all()
    visible_source_ids = {source.source_id for source in sources}
    refs = source_version_refs(sources)
    if not visible_source_ids or {ref.source_id for ref in refs} != visible_source_ids:
        return None
    return refs


def _mark_review_batch_sources(
    *,
    db: Session,
    source_ids: list[str],
    mode: ReviewBatchMode,
) -> None:
    if not source_ids:
        return
    sources = db.scalars(
        select(Source).where(Source.source_id.in_(source_ids))
    ).all()
    for source in sources:
        source.raw_metadata = with_review_batch_marker(
            source,
            mode=mode,
        )


def _default_review_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for manifest in (
        MAIL_DOCUMENT_AGENT_MANIFEST,
        TIMELINE_AGENT_MANIFEST,
        HISTORY_AGENT_MANIFEST,
        DECISION_RECORD_AGENT_MANIFEST,
        TODO_AGENT_MANIFEST,
    ):
        registry.register(manifest)
    return registry


def _prepare_default_review_batch(
    *,
    db: Session,
    user: DemoUser,
    settings: Settings,
    refs: list[SourceVersionRef],
    v21_config: V21PreparedReviewConfig | None = None,
):
    if settings.auto_review_mode != 'disabled' and v21_config is None:
        raise ReviewWorkflowPreflightError(
            'cost_preview_changed',
            'V2.1 launch authority is unavailable',
        )
    request = ReviewWorkflowRunRequest(
        source_refs=[asdict(ref) for ref in refs],
        agent_names=list(DEFAULT_REVIEW_AGENT_NAMES),
    )
    prepared = prepare_review_request(
        db,
        request=request,
        actor=user,
        registry=_default_review_registry(),
        settings=settings,
        v21_config=v21_config,
    )
    return request, prepared


def _legacy_batch_owned_by_v2(
    *,
    db: Session,
    user: DemoUser,
    settings: Settings,
    connector_type: str,
    source_ids: list[str],
) -> bool:
    if connector_type not in GOOGLE_OAUTH_CONNECTOR_TYPES:
        return False
    refs = _complete_visible_source_refs_for_ids(
        db=db,
        user=user,
        source_ids=source_ids,
    )
    if not refs:
        return False
    try:
        _, prepared = _prepare_default_review_batch(
            db=db,
            user=user,
            settings=settings,
            refs=refs,
        )
        return (
            find_matching_review_thread(
                db,
                prepared=prepared,
                actor=user,
                settings=settings,
            )
            is not None
        )
    except ReviewWorkflowPreflightError:
        return False


def _review_batch_hmac(
    *,
    db: Session,
    user: DemoUser,
    settings: Settings,
    refs: list[SourceVersionRef],
) -> str | None:
    if not refs:
        return None
    _, prepared = _prepare_default_review_batch(
        db=db,
        user=user,
        settings=settings,
        refs=refs,
    )
    return prepared.input_hash


def _connector_uses_slack_llm_project_routing(
    *,
    connector_type: str,
    settings: Settings,
) -> bool:
    return connector_type == 'slack' and not settings.paraworks_demo_mode and bool(
        settings.openai_api_key or settings.gemini_api_key or settings.google_api_key
    )


def _skip_project_assignment_after_agent_review(
    *,
    connector_type: str,
    settings: Settings,
    agent_review_items: int,
) -> bool:
    if connector_type == 'slack':
        return True
    return agent_review_items > 0 and _connector_uses_slack_llm_project_routing(
        connector_type=connector_type,
        settings=settings,
    )


def _run_connector_agent_review(
    *,
    db: Session,
    user: DemoUser,
    settings: Settings,
    connector_type: str,
    source_ids: list[str],
) -> int:
    if connector_type == 'slack':
        if not settings.paraworks_demo_mode and (
            settings.openai_api_key
            or settings.gemini_api_key
            or settings.google_api_key
        ):
            return trigger_slack_agent_analysis(
                db=db,
                source_ids=source_ids,
                settings=settings,
            )

        review_items = create_slack_agent_review_items(
            db=db,
            agent=SlackAgent(model=DeterministicSlackAgentModel()),
            permission_context=_permission_context(user),
            source_window=f'sync:{connector_type}:changed',
            source_ids=source_ids,
        )
        return len(review_items)

    if connector_type in GOOGLE_OAUTH_CONNECTOR_TYPES:
        review_items = create_mail_document_agent_review_items_for_changed_sources(
            db=db,
            agent=_build_mail_document_review_agent(settings),
            permission_context=_permission_context(user),
            source_window=f'sync:{connector_type}:changed',
            source_ids=source_ids,
        )
        return len(review_items)

    return 0


def _has_connector_agent_review_items(
    *,
    db: Session,
    connector_type: str,
) -> bool:
    agent_names = {
        'slack': {'slack_agent'},
        'gmail': {'mail_document_agent'},
        'google_drive': {'mail_document_agent'},
        'google_calendar': {'mail_document_agent'},
        'drive': {'mail_document_agent'},
        'calendar': {'mail_document_agent'},
    }.get(connector_type, set())
    if not agent_names:
        return True
    return (
        db.scalar(
            select(ReviewItem.id)
            .where(ReviewItem.payload['agent_name'].as_string().in_(agent_names))
            .limit(1)
        )
        is not None
    )


def _connector_source_ids_for_review(
    *,
    db: Session,
    connector_type: str,
    user: DemoUser,
) -> list[str]:
    if connector_type == 'slack':
        return list(
            db.scalars(
                select(Source.source_id)
                .where(Source.source_type == 'slack')
                .order_by(Source.id)
            ).all()
        )
    if connector_type in GOOGLE_OAUTH_CONNECTOR_TYPES:
        return _mail_document_source_ids(db=db, user=user)
    return []


@router.get('/slack/oauth/install-url')
def get_slack_oauth_install_url(
    settings: AppSettings,
    redirect_uri: str | None = None,
) -> dict[str, object]:
    try:
        install = build_slack_oauth_install_url(
            settings=settings, redirect_uri=redirect_uri
        )
    except SlackOAuthConfigurationError:
        return {
            'connector_type': 'slack',
            'configured': False,
            'install_url': None,
            'state': None,
            'required_scopes': [],
        }

    return {
        'connector_type': install.connector_type,
        'configured': install.configured,
        'install_url': install.install_url,
        'state': install.state,
        'required_scopes': install.required_scopes,
        'code_verifier': install.code_verifier,
    }


@router.post('/slack/direct-connect')
def complete_slack_direct_install(
    db: DbSession,
    settings: AppSettings,
) -> dict[str, object]:
    try:
        connection = complete_slack_direct_connect(
            db=db,
            settings=settings,
        )
    except SlackOAuthConfigurationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SlackApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        'connector_type': connection.connector_type,
        'status': connection.status,
        'workspace_id': connection.workspace_id,
        'workspace_name': connection.workspace_name,
        'masked_bot_token': connection.masked_bot_token,
        'scopes': connection.scopes,
        'credential_status': 'available',
    }


@router.get('/slack/oauth/callback')
def complete_slack_oauth_install(
    code: str,
    state: str,
    db: DbSession,
    settings: AppSettings,
    redirect_uri: str | None = None,
) -> dict[str, object]:
    try:
        connection = complete_slack_oauth_callback(
            db=db,
            settings=settings,
            code=code,
            state=state,
            redirect_uri=redirect_uri,
        )
    except SlackOAuthConfigurationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SlackApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        'connector_type': connection.connector_type,
        'status': connection.status,
        'workspace_id': connection.workspace_id,
        'workspace_name': connection.workspace_name,
        'masked_bot_token': connection.masked_bot_token,
        'scopes': connection.scopes,
        'pkce_used': connection.raw_metadata.get('pkce_used', False),
    }


@router.get('/{connector_type}/oauth/install-url')
def get_google_oauth_install_url(
    connector_type: str,
    settings: AppSettings,
    redirect_uri: str | None = None,
) -> dict[str, object]:
    if connector_type not in GOOGLE_OAUTH_CONNECTOR_TYPES:
        raise HTTPException(status_code=404, detail='Connector not found')

    try:
        install = build_google_oauth_install_url(
            settings=settings,
            connector_type=connector_type,
            redirect_uri=redirect_uri,
        )
    except GoogleOAuthConfigurationError:
        return {
            'connector_type': connector_type,
            'configured': False,
            'install_url': None,
            'state': None,
            'required_scopes': [],
        }
    except GoogleOAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        'connector_type': install.connector_type,
        'configured': install.configured,
        'install_url': install.install_url,
        'state': install.state,
        'required_scopes': install.required_scopes,
        'code_verifier': install.code_verifier,
    }


@router.get('/google/oauth/callback')
def complete_google_oauth_install_from_state(
    code: str,
    state: str,
    db: DbSession,
    settings: AppSettings,
    redirect_uri: str | None = None,
) -> dict[str, object]:
    try:
        connector_type = (
            GoogleOAuthStateSigner(settings.google_oauth_state_secret)
            .validate(state)
            .connector_type
        )
        connection = complete_google_oauth_callback(
            db=db,
            settings=settings,
            connector_type=connector_type,
            code=code,
            state=state,
            token_vault=LOCAL_TOKEN_VAULT,
            redirect_uri=redirect_uri,
        )
    except GoogleOAuthConfigurationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GoogleOAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        'connector_type': connection.connector_type,
        'status': connection.status,
        'workspace_id': connection.workspace_id,
        'workspace_name': connection.workspace_name,
        'masked_bot_token': connection.masked_bot_token,
        'scopes': connection.scopes,
        'pkce_used': connection.raw_metadata.get('pkce_used', False),
    }


@router.get('/{connector_type}/oauth/callback')
def complete_google_oauth_install(
    connector_type: str,
    code: str,
    state: str,
    db: DbSession,
    settings: AppSettings,
    redirect_uri: str | None = None,
) -> dict[str, object]:
    if connector_type not in GOOGLE_OAUTH_CONNECTOR_TYPES:
        raise HTTPException(status_code=404, detail='Connector not found')

    try:
        connection = complete_google_oauth_callback(
            db=db,
            settings=settings,
            connector_type=connector_type,
            code=code,
            state=state,
            token_vault=LOCAL_TOKEN_VAULT,
            redirect_uri=redirect_uri,
        )
    except GoogleOAuthConfigurationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GoogleOAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        'connector_type': connection.connector_type,
        'status': connection.status,
        'workspace_id': connection.workspace_id,
        'workspace_name': connection.workspace_name,
        'masked_bot_token': connection.masked_bot_token,
        'scopes': connection.scopes,
        'pkce_used': connection.raw_metadata.get('pkce_used', False),
    }


@router.delete('/{connector_type}')
def disconnect_connector(
    connector_type: str,
    db: DbSession,
    user: CurrentUser,
) -> dict[str, str]:
    connection = db.scalar(
        select(IntegrationConnection)
        .where(IntegrationConnection.connector_type == connector_type)
        .order_by(IntegrationConnection.id.desc())
    )
    if not connection:
        raise HTTPException(status_code=404, detail='Connection not found')

    # Remove token from vault
    LOCAL_TOKEN_VAULT.remove_token(connection.token_ref)

    # Remove from database
    db.delete(connection)
    db.commit()

    record_audit_log(
        db=db,
        actor=user,
        action='integration.disconnect',
        target_type='connector',
        target_id=connector_type,
        metadata={'workspace_name': connection.workspace_name},
    )

    return {'status': 'disconnected', 'connector_type': connector_type}


@router.post('/slack/agent-review')
def run_slack_agent_review(db: DbSession, user: CurrentUser) -> dict[str, int | str]:
    agent = SlackAgent(model=DeterministicSlackAgentModel())
    review_items = create_slack_agent_review_items(
        db=db,
        agent=agent,
        permission_context=_permission_context(user),
        source_window='mock-slack:all',
    )
    record_audit_log(
        db=db,
        actor=user,
        action='agent.review.run',
        target_type='agent',
        target_id='slack_agent',
        metadata={'created_review_items': len(review_items)},
    )
    db.commit()

    return {
        'agent_name': 'slack_agent',
        'status': 'complete',
        'created_review_items': len(review_items),
    }


@router.get('/slack/agent-review/llm/preflight')
def get_slack_llm_agent_preflight(
    db: DbSession,
    settings: AppSettings,
    user: CurrentUser,
) -> dict[str, object]:
    llm_settings = _slack_llm_settings(settings)
    packet = _build_slack_llm_evidence_packet(db=db, user=user, settings=llm_settings)
    preflight = build_slack_llm_preflight(
        packet=packet,
        settings=llm_settings,
    )
    preflight['source_window'] = packet.source_window
    return preflight


@router.post('/slack/agent-review/llm')
def run_slack_llm_agent_review(
    request: SlackLlmRunRequest,
    db: DbSession,
    settings: AppSettings,
    user: CurrentUser,
) -> dict[str, int | str | float | dict[str, object]]:
    llm_settings = _slack_llm_settings(settings)
    packet = _build_slack_llm_evidence_packet(db=db, user=user, settings=llm_settings)
    preflight = build_slack_llm_preflight(packet=packet, settings=llm_settings)
    preflight['source_window'] = packet.source_window
    if preflight['action'] != 'run':
        raise HTTPException(status_code=400, detail=preflight)
    if not request.confirm_paid_run:
        raise HTTPException(
            status_code=400, detail='Paid LLM run requires confirm_paid_run=true'
        )

    try:
        agent = SlackAgent(
            model=build_langchain_slack_agent_model(llm_settings),
            input_cost_per_1m=llm_settings.input_cost_per_1m,
            output_cost_per_1m=llm_settings.output_cost_per_1m,
        )
        review_items = create_slack_agent_review_items(
            db=db,
            agent=agent,
            permission_context=_permission_context(user),
            source_window=_slack_llm_source_window(llm_settings),
            max_messages=llm_settings.max_evidence_messages,
            selection_strategy='ranked',
        )
    except SlackLlmProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    record_audit_log(
        db=db,
        actor=user,
        action='agent.review.llm_run',
        target_type='agent',
        target_id='slack_agent',
        metadata={
            'created_review_items': len(review_items),
            'preflight': preflight,
        },
    )
    db.commit()

    return {
        'agent_name': 'slack_agent',
        'status': 'complete',
        'created_review_items': len(review_items),
        'preflight': preflight,
    }


@router.post('/mail-docs/agent-review')
def run_mail_document_agent_review(
    db: DbSession, settings: AppSettings, user: CurrentUser
) -> dict[str, int | str]:
    agent = _build_mail_document_review_agent(settings)
    source_ids = _mail_document_source_ids(db=db, user=user)
    review_items = create_mail_document_agent_review_items_for_changed_sources(
        db=db,
        agent=agent,
        permission_context=_permission_context(user),
        source_window='mock-mail-docs:grouped',
        source_ids=source_ids,
    )
    record_audit_log(
        db=db,
        actor=user,
        action='agent.review.run',
        target_type='agent',
        target_id='mail_document_agent',
        metadata={
            'created_review_items': len(review_items),
            'source_count': len(source_ids),
            'source_window': 'mock-mail-docs:grouped',
            'selection_strategy': 'source_group',
        },
    )
    db.commit()

    return {
        'agent_name': 'mail_document_agent',
        'status': 'complete',
        'created_review_items': len(review_items),
    }


def _build_mail_document_review_agent(settings: Settings) -> MailDocumentAgent:
    llm_settings = _mail_document_llm_settings(settings)
    if _should_use_mail_document_llm(llm_settings):
        return MailDocumentAgent(
            model=build_langchain_mail_document_agent_model(llm_settings),
            input_cost_per_1m=llm_settings.input_cost_per_1m,
            output_cost_per_1m=llm_settings.output_cost_per_1m,
        )
    return MailDocumentAgent(model=DeterministicMailDocumentAgentModel())


def _should_use_mail_document_llm(settings: MailDocumentLlmSettings) -> bool:
    return settings.enabled and bool(settings.openai_api_key or settings.gemini_api_key)


@router.get('/mail-docs/agent-review/llm/preflight')
def get_mail_document_agent_preflight(
    db: DbSession,
    settings: AppSettings,
    user: CurrentUser,
) -> dict[str, object]:
    llm_settings = _mail_document_llm_settings(settings)
    packet = _build_mail_document_llm_evidence_packet(db=db, user=user, settings=llm_settings)
    preflight = build_mail_document_llm_preflight(packet=packet, settings=llm_settings)
    preflight['source_window'] = packet.source_window
    return preflight


@router.post('/mail-docs/agent-review/llm')
def run_mail_document_llm_agent_review(
    request: SlackLlmRunRequest,
    db: DbSession,
    settings: AppSettings,
    user: CurrentUser,
) -> dict[str, int | str | float | dict[str, object]]:
    llm_settings = _mail_document_llm_settings(settings)
    packet = _build_mail_document_llm_evidence_packet(db=db, user=user, settings=llm_settings)
    preflight = build_mail_document_llm_preflight(packet=packet, settings=llm_settings)
    preflight['source_window'] = packet.source_window
    if preflight['action'] != 'run':
        raise HTTPException(status_code=400, detail=preflight)
    if not request.confirm_paid_run:
        raise HTTPException(status_code=400, detail='Paid LLM run requires confirm_paid_run=true')

    try:
        agent = MailDocumentAgent(
            model=build_langchain_mail_document_agent_model(llm_settings),
            input_cost_per_1m=llm_settings.input_cost_per_1m,
            output_cost_per_1m=llm_settings.output_cost_per_1m,
        )
        review_items = create_mail_document_agent_review_items_for_changed_sources(
            db=db,
            agent=agent,
            permission_context=_permission_context(user),
            source_window=_mail_document_llm_source_window(llm_settings),
            source_ids=_mail_document_source_ids(db=db, user=user),
        )
    except MailDocumentLlmProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    record_audit_log(
        db=db,
        actor=user,
        action='agent.review.llm_run',
        target_type='agent',
        target_id='mail_document_agent',
        metadata={
            'created_review_items': len(review_items),
            'preflight': preflight,
            'source_window': packet.source_window,
            'evidence_message_count': len(packet.messages),
            'included_source_types': sorted({
                str(message.metadata.get('source_type'))
                for message in packet.messages
                if message.metadata.get('source_type')
            }),
            'parser_status_counts': _parser_status_counts(packet),
            'selection_strategy': 'source_group',
        },
    )
    db.commit()

    return {
        'agent_name': 'mail_document_agent',
        'status': 'complete',
        'created_review_items': len(review_items),
        'preflight': preflight,
    }


@router.get('/memory-extraction/agent-review/llm/preflight')
def get_memory_extraction_agent_preflight(
    db: DbSession, user: CurrentUser
) -> dict[str, object]:
    return build_memory_extraction_agent_preflight(
        db=db,
        permission_context=_permission_context(user),
        source_window='memory-extraction:preflight',
    )


def _configured_channel_ids(raw_channel_ids: str) -> list[str]:
    return [
        channel_id.strip()
        for channel_id in raw_channel_ids.split(',')
        if channel_id.strip()
    ]


def _slack_llm_settings(settings: Settings) -> SlackLlmSettings:
    return SlackLlmSettings(
        enabled=settings.agent_llm_enabled,
        provider_order=tuple(
            _configured_channel_ids(settings.agent_llm_provider_order)
        ),
        openai_api_key=settings.openai_api_key,
        gemini_api_key=settings.gemini_api_key or settings.google_api_key,
        openai_model=settings.agent_llm_openai_model,
        gemini_model=settings.agent_llm_gemini_model,
        input_cost_per_1m=settings.agent_llm_input_cost_per_1m_tokens,
        output_cost_per_1m=settings.agent_llm_output_cost_per_1m_tokens,
        max_estimated_cost_usd=settings.agent_llm_max_estimated_cost_usd,
        max_input_chars=settings.agent_llm_max_input_chars,
        max_evidence_messages=settings.agent_llm_max_evidence_messages,
        max_output_tokens=settings.agent_llm_max_output_tokens,
        temperature=settings.agent_llm_temperature,
        timeout_seconds=settings.agent_llm_timeout_seconds,
    )


def _mail_document_llm_settings(settings: Settings) -> MailDocumentLlmSettings:
    return MailDocumentLlmSettings(
        enabled=settings.agent_llm_enabled,
        provider_order=tuple(_configured_channel_ids(settings.agent_llm_provider_order)),
        openai_api_key=settings.openai_api_key,
        gemini_api_key=settings.gemini_api_key or settings.google_api_key,
        openai_model=settings.agent_llm_openai_model,
        gemini_model=settings.agent_llm_gemini_model,
        input_cost_per_1m=settings.agent_llm_input_cost_per_1m_tokens,
        output_cost_per_1m=settings.agent_llm_output_cost_per_1m_tokens,
        max_estimated_cost_usd=settings.agent_llm_max_estimated_cost_usd,
        max_input_chars=settings.agent_llm_max_input_chars,
        max_evidence_messages=settings.agent_llm_max_evidence_messages,
        max_output_tokens=settings.agent_llm_max_output_tokens,
        temperature=settings.agent_llm_temperature,
        timeout_seconds=settings.agent_llm_timeout_seconds,
    )


def _mail_document_llm_settings(settings: Settings) -> MailDocumentLlmSettings:
    return MailDocumentLlmSettings(
        enabled=settings.agent_llm_enabled,
        provider_order=tuple(_configured_channel_ids(settings.agent_llm_provider_order)),
        openai_api_key=settings.openai_api_key,
        gemini_api_key=settings.gemini_api_key or settings.google_api_key,
        openai_model=settings.agent_llm_openai_model,
        gemini_model=settings.agent_llm_gemini_model,
        input_cost_per_1m=settings.agent_llm_input_cost_per_1m_tokens,
        output_cost_per_1m=settings.agent_llm_output_cost_per_1m_tokens,
        max_estimated_cost_usd=settings.agent_llm_max_estimated_cost_usd,
        max_input_chars=settings.agent_llm_max_input_chars,
        max_evidence_messages=settings.agent_llm_max_evidence_messages,
        max_output_tokens=settings.agent_llm_max_output_tokens,
        temperature=settings.agent_llm_temperature,
        timeout_seconds=settings.agent_llm_timeout_seconds,
    )


def _slack_llm_source_window(settings: SlackLlmSettings) -> str:
    return f'slack:live:ranked:{settings.max_evidence_messages}'


def _mail_document_llm_source_window(settings: MailDocumentLlmSettings) -> str:
    return f'mail-docs:live:ranked:{settings.max_evidence_messages}'


def _build_slack_llm_evidence_packet(
    *,
    db: Session,
    user: CurrentUser,
    settings: SlackLlmSettings,
) -> EvidencePacket:
    return build_slack_evidence_packet(
        db=db,
        permission_context=_permission_context(user),
        source_window=_slack_llm_source_window(settings),
        max_messages=settings.max_evidence_messages,
        selection_strategy='ranked',
    )


def _build_mail_document_llm_evidence_packet(
    *,
    db: Session,
    user: CurrentUser,
    settings: MailDocumentLlmSettings,
) -> EvidencePacket:
    return build_mail_document_evidence_packet(
        db=db,
        permission_context=_permission_context(user),
        source_window=_mail_document_llm_source_window(settings),
        max_messages=settings.max_evidence_messages,
        selection_strategy='ranked',
    )


def _permission_context(user: DemoUser) -> PermissionContext:
    return PermissionContext(
        user_id=user.id,
        role=user.role,
        allowed_permission_levels=tuple(user.permission_levels),
    )


def _mail_document_source_ids(*, db: Session, user: DemoUser) -> list[str]:
    rows = db.scalars(
        select(Source.source_id)
        .join(DocumentChunk, DocumentChunk.source_id == Source.id)
        .where(Source.source_type.in_(MAIL_DOCUMENT_SOURCE_TYPES))
        .where(DocumentChunk.permission_level.in_(tuple(user.permission_levels)))
        .order_by(Source.id)
    ).all()
    seen: set[str] = set()
    source_ids: list[str] = []
    for source_id in rows:
        if source_id not in seen:
            source_ids.append(source_id)
            seen.add(source_id)
    return source_ids


def _parser_status_counts(packet: EvidencePacket) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in packet.messages:
        status = message.metadata.get('parser_status')
        if isinstance(status, str) and status:
            counts[status] = counts.get(status, 0) + 1
    return counts


def _clean_channel_ids(channel_ids: list[str] | None) -> list[str]:
    if not channel_ids:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for channel_id in channel_ids:
        normalized = channel_id.strip()
        if normalized and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)
    return cleaned


def _slack_channel_options(raw_channel_ids: str) -> list[dict[str, object]]:
    return [
        {
            'id': channel_id,
            'name': channel_id,
            'is_selected': True,
            'is_configured': True,
        }
        for channel_id in _configured_channel_ids(raw_channel_ids)
    ]


def _sync_job_response(
    *,
    db: Session,
    user: DemoUser,
    job: SyncJob | None,
) -> dict[str, object] | None:
    if job is None:
        return None
    refs: list[SourceVersionRef] = []
    if job.status == 'complete':
        sources = db.scalars(
            select(Source).where(
                Source.raw_metadata['last_changed_sync_job_id'].as_string()
                == job.job_id,
                Source.permission_level.in_(tuple(user.permission_levels)),
            )
        ).all()
        refs = source_version_refs(sources)
    return {
        'job_id': job.job_id,
        'status': job.status,
        'message': redact_secret_text(job.message),
        'progress_pct': job.progress_pct,
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'updated_at': job.updated_at.isoformat() if job.updated_at else None,
        'changed_source_refs': [asdict(ref) for ref in refs],
    }


def _mark_sync_job_agent_review_running(
    *,
    db: Session,
    job_id: str,
    fetched_events: int,
    skipped_events: int,
) -> None:
    sync_job = db.scalar(select(SyncJob).where(SyncJob.job_id == job_id))
    if sync_job is None:
        return
    sync_job.status = 'running'
    sync_job.progress_pct = 75
    sync_job.message = (
        f'fetched={fetched_events} '
        'agent_review=running '
        f'skipped_events={skipped_events}'
    )
    db.commit()


def _mark_sync_job_failed(*, db: Session, job_id: str, message: str) -> None:
    sync_job = db.scalar(select(SyncJob).where(SyncJob.job_id == job_id))
    if sync_job is None:
        return
    sync_job.status = 'failed'
    sync_job.progress_pct = 100
    sync_job.message = redact_secret_text(message)
    sync_job.updated_at = datetime.now(UTC)
    db.flush()


def _pending_review_count(
    db: Session, user: DemoUser | None = None
) -> int:
    if user is None:
        return 0
    items = tuple(
        db.scalars(
            select(ReviewItem).where(
                ReviewItem.status == 'pending_review'
            )
        ).all()
    )
    service = ReviewEvidenceVisibilityService(db)
    count = 0
    for item in items:
        try:
            service.project(item.id, user)
        except ReviewEvidenceNotFound:
            continue
        count += 1
    return count


def _sync_job_summary(job: SyncJob | None) -> dict[str, int] | None:
    if job is None:
        return None
    message = job.message or ''
    return {
        'fetched_events': _extract_count(message, 'fetched'),
        'created_review_items': _extract_count(message, 'created_review_items'),
        'skipped_events': _extract_count(message, 'skipped_events'),
    }


def _sync_error_response(job: SyncJob | None) -> dict[str, str] | None:
    if job is None or job.status != 'failed':
        return None
    message = redact_secret_text(job.message)
    code = (
        message.rsplit(':', maxsplit=1)[-1].strip()
        if ':' in message
        else 'unknown_error'
    )
    return {
        'code': code,
        'message': message,
        'action_hint': _slack_error_action_hint(code),
    }


def _slack_agent_bridge(
    db: Session, user: DemoUser | None = None
) -> dict[str, int | bool]:
    slack_source_count = (
        db.scalar(
            select(func.count())
            .select_from(Source)
            .where(Source.source_type == 'slack')
        )
        or 0
    )
    return {
        'slack_source_count': slack_source_count,
        'pending_review_count': _pending_review_count(db, user),
        'ready_for_agent_test': slack_source_count > 0,
    }


def _extract_count(message: str, key: str) -> int:
    match = re.search(rf'{re.escape(key)}=(\d+)', message)
    return int(match.group(1)) if match else 0


def _slack_error_action_hint(code: str) -> str:
    if code in {'not_in_channel', 'channel_not_found'}:
        return 'Slack 앱을 선택한 채널에 추가한 뒤 다시 동기화하세요.'
    if code == 'missing_scope':
        return 'Slack OAuth 권한 범위를 확인한 뒤 앱을 다시 설치하세요.'
    if code == 'rate_limited':
        return 'Slack API 제한이 풀린 뒤 다시 시도하세요.'
    return 'Slack 연결, 채널 권한, 토큰 상태를 확인한 뒤 다시 동기화하세요.'
