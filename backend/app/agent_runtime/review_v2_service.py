from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import cast
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, Interrupt
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.agent_runtime.canonical_sources import (
    ReviewWorkflowPreflightError,
    resolve_source_versions,
)
from backend.app.agent_runtime.checkpoint_execution import (
    CheckpointConfirmationError,
    checkpoint_config,
    invoke_and_confirm_checkpoint,
    require_resumable_checkpoint,
)
from backend.app.agent_runtime.checkpointing import (
    CheckpointRuntime,
    CheckpointUnavailableError,
)
from backend.app.agent_runtime.graph_versions import (
    GraphVersionRegistry,
    RuntimeVersionUnavailable,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_drafting import (
    ReviewDraftError,
    ReviewDraftResult,
)
from backend.app.agent_runtime.review_v2_preflight import (
    create_or_reuse_review_thread,
    prepare_review_request,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_versions import (
    CanonicalSourceType,
    SourceVersionRef,
)
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
)
from backend.app.models.review import ReviewItem
from backend.app.models.source import Source
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    DEFAULT_REVIEW_AGENT_NAMES,
    ReviewItemResolutionStatus,
    ReviewWorkflowDiagnosticResponse,
    ReviewWorkflowDryRunResponse,
    ReviewWorkflowErrorCode,
    ReviewWorkflowRunRequest,
)

_REVIEW_STATUSES = (
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
)
_TERMINAL_THREAD_STATUSES = frozenset({
    'completed',
    'needs_more_evidence',
    'failed',
    'cancelled',
})
_CANCELLABLE_THREAD_STATUSES = frozenset({
    'created',
    'drafting',
    'checkpoint_pending',
    'awaiting_human_review',
    'resuming',
    'checkpoint_failed',
})
_PUBLIC_ERROR_CODES = frozenset({
    'invalid_input',
    'not_found',
    'idempotency_key_reused',
    'evidence_changed',
    'permission_denied',
    'checkpoint_unavailable',
    'checkpoint_failed',
    'review_unresolved',
    'runtime_version_unavailable',
    'model_unavailable',
    'budget_exceeded',
    'concurrent_resume',
    'invalid_state_transition',
})


class ReviewWorkflowServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        bounded = code if code in _PUBLIC_ERROR_CODES else 'invalid_state_transition'
        self.code = cast(ReviewWorkflowErrorCode, bounded)
        super().__init__(bounded)


@dataclass(frozen=True)
class ReviewWorkflowStatus:
    thread_id: str
    status: str
    review_item_count: int
    review_status_counts: dict[ReviewItemResolutionStatus, int]
    durable: bool
    graph_version: str
    review_resolution_ready: bool
    checkpoint_resumable: bool
    resume_allowed: bool
    retry_allowed: bool
    created_at: datetime
    updated_at: datetime
    error_code: ReviewWorkflowErrorCode | None
    resume_error_code: ReviewWorkflowErrorCode | None


@dataclass(frozen=True)
class ReviewModelReadiness:
    ready: bool
    error_code: ReviewWorkflowErrorCode | None = None


@dataclass(frozen=True)
class _ThreadProjection:
    thread_id: str
    workflow_name: str
    graph_version: str
    input_hash: str
    evidence_version_hash: str
    checkpoint_thread_id: str
    checkpoint_store: str
    owner_subject_id: str
    status: str
    state_version: int
    review_item_ids: tuple[int, ...]
    review_status_counts: dict[ReviewItemResolutionStatus, int]
    created_at: datetime
    updated_at: datetime

    @property
    def review_item_count(self) -> int:
        return sum(self.review_status_counts.values())

    @property
    def review_resolution_ready(self) -> bool:
        return (
            self.review_item_count > 0
            and self.review_status_counts.get('pending_review', 0) == 0
        )


@dataclass(frozen=True)
class _CheckpointProbe:
    interrupt_state_version: int | None
    terminal_status: str | None


class _MissingCheckpointError(RuntimeError):
    pass


class _CorruptCheckpointError(RuntimeError):
    pass


class _UnusedLeaseService:
    def acquire(self, *, workflow_thread_id: str) -> str:
        del workflow_thread_id
        raise RuntimeError('unexpected review lease acquisition')


@dataclass(frozen=True)
class _PreparedDraftService:
    workflow_thread_id: str
    result: ReviewDraftResult

    def draft(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewDraftResult:
        del actor_subject_id, allowed_permission_levels
        if workflow_thread_id != self.workflow_thread_id:
            raise ReviewDraftError(
                'invalid_state_transition',
                'prepared draft belongs to another workflow',
            )
        return self.result


def require_thread_checkpoint_runtime(
    *,
    thread: AgentWorkflowThread | _ThreadProjection,
    runtime: CheckpointRuntime,
) -> BaseCheckpointSaver:
    if thread.checkpoint_store != runtime.readiness.checkpoint_store:
        raise CheckpointUnavailableError('checkpoint runtime mode mismatch')
    if not runtime.readiness.ready or runtime.saver is None:
        raise CheckpointUnavailableError('checkpoint runtime unavailable')
    return runtime.saver


class ReviewWorkflowService:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        settings: Settings,
        checkpoint_runtime: CheckpointRuntime,
        graph_registry: GraphVersionRegistry,
        agent_registry: AgentRegistry,
        draft_service: object,
        model_readiness: ReviewModelReadiness | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._checkpoint_runtime = checkpoint_runtime
        self._graph_registry = graph_registry
        self._agent_registry = agent_registry
        self._draft_service = draft_service
        self._model_readiness = model_readiness or ReviewModelReadiness(
            ready=True
        )
        self._now = now or (lambda: datetime.now(UTC))
        self._lifecycle_lock = RLock()

    def diagnostic(self) -> ReviewWorkflowDiagnosticResponse:
        if not self._settings.langgraph_review_v2_enabled:
            return ReviewWorkflowDiagnosticResponse(
                enabled=False,
                available=False,
                checkpoint_mode='disabled',
                durable=False,
                graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
                default_agent_names=list(DEFAULT_REVIEW_AGENT_NAMES),
                error_code=None,
            )
        readiness = self._checkpoint_runtime.readiness
        checkpoint_available = bool(
            readiness.ready and self._checkpoint_runtime.saver
        )
        available = checkpoint_available and self._model_readiness.ready
        if not self._model_readiness.ready:
            error_code: ReviewWorkflowErrorCode | None = 'model_unavailable'
        elif not checkpoint_available:
            error_code = 'checkpoint_unavailable'
        else:
            error_code = None
        return ReviewWorkflowDiagnosticResponse(
            enabled=True,
            available=available,
            checkpoint_mode=readiness.mode,
            durable=readiness.durable if checkpoint_available else False,
            graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
            default_agent_names=list(DEFAULT_REVIEW_AGENT_NAMES),
            error_code=error_code,
        )

    def dry_run(
        self,
        *,
        actor: DemoUser,
        request: ReviewWorkflowRunRequest,
    ) -> ReviewWorkflowDryRunResponse:
        self._require_new_run_availability()
        try:
            with self._session_factory() as db:
                prepared = prepare_review_request(
                    db,
                    request=request,
                    actor=actor,
                    registry=self._agent_registry,
                    settings=self._settings,
                )
                db.rollback()
            return self._draft_service.preview_prepared(
                prepared=prepared,
                actor_subject_id=actor.id,
                allowed_permission_levels=tuple(actor.permission_levels),
            )
        except ReviewWorkflowPreflightError as exc:
            self._raise_preflight_error(exc)
        except ReviewDraftError as exc:
            self._raise_preflight_error(exc)
        raise AssertionError('unreachable')

    def start(
        self,
        *,
        actor: DemoUser,
        request: ReviewWorkflowRunRequest,
    ) -> ReviewWorkflowStatus:
        with self._lifecycle_lock:
            self._require_new_run_availability()
            try:
                with self._session_factory() as db:
                    prepared = prepare_review_request(
                        db,
                        request=request,
                        actor=actor,
                        registry=self._agent_registry,
                        settings=self._settings,
                    )
                    db.rollback()
                preview = self._draft_service.preview_prepared(
                    prepared=prepared,
                    actor_subject_id=actor.id,
                    allowed_permission_levels=tuple(actor.permission_levels),
                )
                if preview.budget_status == 'over_budget':
                    raise ReviewWorkflowServiceError('budget_exceeded')
                with self._session_factory() as db:
                    preflight = create_or_reuse_review_thread(
                        db,
                        prepared=prepared,
                        request=request,
                        actor=actor,
                        settings=self._settings,
                    )
            except ReviewWorkflowPreflightError as exc:
                self._raise_preflight_error(exc)
            except ReviewDraftError as exc:
                self._raise_preflight_error(exc)

            if not preflight.created:
                return self.status(actor=actor, thread_id=preflight.thread.thread_id)

            thread_id = preflight.thread.thread_id
            draft_result = self._draft_with_bounded_retry(
                thread_id=thread_id,
                actor=actor,
            )
            self._ensure_checkpoint_pending(thread_id)
            projection = self._read_projection(actor=actor, thread_id=thread_id)
            try:
                builder = self._resolve_builder(projection)
                saver = require_thread_checkpoint_runtime(
                    thread=projection,
                    runtime=self._checkpoint_runtime,
                )
                graph = builder(saver)
                confirmation = invoke_and_confirm_checkpoint(
                    graph=graph,
                    saver=saver,
                    command_or_input={'workflow_thread_id': thread_id},
                    checkpoint_thread_id=projection.checkpoint_thread_id,
                    runtime_context=self._runtime_context(
                        actor,
                        draft_service=_PreparedDraftService(
                            workflow_thread_id=thread_id,
                            result=draft_result,
                        ),
                    ),
                    expect_interrupt=bool(draft_result.review_item_ids),
                )
            except CheckpointUnavailableError:
                self._mark_checkpoint_failed(thread_id)
                raise ReviewWorkflowServiceError('checkpoint_unavailable') from None
            except CheckpointConfirmationError:
                self._mark_checkpoint_failed(thread_id)
                raise ReviewWorkflowServiceError('checkpoint_failed') from None
            except ReviewDraftError as exc:
                self._mark_checkpoint_failed(thread_id)
                self._raise_preflight_error(exc)
            except PermissionError:
                self._mark_checkpoint_failed(thread_id)
                raise ReviewWorkflowServiceError('not_found') from None
            except ValueError as exc:
                self._mark_checkpoint_failed(thread_id)
                self._raise_value_error(exc)
            except ReviewWorkflowServiceError:
                raise
            except Exception:
                self._mark_checkpoint_failed(thread_id)
                raise ReviewWorkflowServiceError('checkpoint_failed') from None

            result_status = confirmation.result.get('status')
            if draft_result.review_item_ids:
                if not confirmation.interrupted:
                    self._mark_checkpoint_failed(thread_id)
                    raise ReviewWorkflowServiceError('checkpoint_failed')
                target_status = 'awaiting_human_review'
            else:
                if result_status != 'completed':
                    self._mark_checkpoint_failed(thread_id)
                    raise ReviewWorkflowServiceError('checkpoint_failed')
                target_status = 'completed'
            if not self._finish_initial_checkpoint(
                projection=self._read_projection(actor=actor, thread_id=thread_id),
                target_status=target_status,
            ):
                raise ReviewWorkflowServiceError('checkpoint_failed')
            return self.status(actor=actor, thread_id=thread_id)

    def status(
        self,
        *,
        actor: DemoUser,
        thread_id: str,
    ) -> ReviewWorkflowStatus:
        with self._lifecycle_lock:
            projection = self._read_projection(actor=actor, thread_id=thread_id)
            try:
                builder = self._resolve_builder(projection)
            except ReviewWorkflowServiceError:
                return self._status_response(
                    projection,
                    actor=actor,
                    checkpoint_resumable=False,
                    error_code='runtime_version_unavailable',
                    resume_error_code='runtime_version_unavailable',
                )

            checkpoint_resumable = False
            probe: _CheckpointProbe | None = None
            resume_error_code: ReviewWorkflowErrorCode | None = None
            try:
                saver = require_thread_checkpoint_runtime(
                    thread=projection,
                    runtime=self._checkpoint_runtime,
                )
                probe = self._probe_checkpoint(
                    projection=projection,
                    builder=builder,
                    saver=saver,
                )
                checkpoint_resumable = probe.interrupt_state_version is not None
            except (
                CheckpointUnavailableError,
                _MissingCheckpointError,
                _CorruptCheckpointError,
            ):
                resume_error_code = 'checkpoint_unavailable'

            reconciled = False
            if (
                checkpoint_resumable
                and projection.status in {'checkpoint_pending', 'checkpoint_failed'}
            ):
                reconciled = self._cas_status(
                    projection=projection,
                    target_status='awaiting_human_review',
                    checkpoint_confirmed=True,
                )
            elif (
                probe is not None
                and probe.terminal_status is not None
                and projection.status
                in {'checkpoint_pending', 'resuming', 'checkpoint_failed'}
            ):
                reconciled = self._cas_status(
                    projection=projection,
                    target_status=probe.terminal_status,
                    checkpoint_confirmed=True,
                    completed=True,
                )
            if reconciled:
                projection = self._read_projection(actor=actor, thread_id=thread_id)

            return self._status_response(
                projection,
                actor=actor,
                checkpoint_resumable=checkpoint_resumable,
                error_code=self._thread_error_code(projection.status),
                resume_error_code=resume_error_code,
            )

    def resume(
        self,
        *,
        actor: DemoUser,
        thread_id: str,
    ) -> ReviewWorkflowStatus:
        with self._lifecycle_lock:
            projection = self._read_projection(
                actor=actor,
                thread_id=thread_id,
                action='resume',
                require_current_versions=True,
            )
            builder = self._resolve_builder(projection)
            if projection.status in _TERMINAL_THREAD_STATUSES:
                raise ReviewWorkflowServiceError('invalid_state_transition')
            if projection.status not in {
                'awaiting_human_review',
                'checkpoint_failed',
            }:
                raise ReviewWorkflowServiceError('invalid_state_transition')
            if (
                projection.status == 'awaiting_human_review'
                and projection.review_status_counts.get('pending_review', 0)
            ):
                raise ReviewWorkflowServiceError('review_unresolved')
            saver = self._exact_saver_or_error(projection)

            if projection.status == 'checkpoint_failed':
                try:
                    probe = self._probe_checkpoint(
                        projection=projection,
                        builder=builder,
                        saver=saver,
                    )
                except (_MissingCheckpointError, _CorruptCheckpointError):
                    return self._repair_checkpoint(
                        projection=projection,
                        actor=actor,
                        builder=builder,
                    )
                except CheckpointUnavailableError:
                    raise ReviewWorkflowServiceError(
                        'checkpoint_unavailable'
                    ) from None
                if probe.terminal_status is not None:
                    if not self._cas_status(
                        projection=projection,
                        target_status=probe.terminal_status,
                        checkpoint_confirmed=True,
                        completed=True,
                    ):
                        raise ReviewWorkflowServiceError('concurrent_resume')
                    return self.status(actor=actor, thread_id=thread_id)
                if probe.interrupt_state_version is None:
                    raise ReviewWorkflowServiceError('checkpoint_unavailable')
                if not self._cas_status(
                    projection=projection,
                    target_status='awaiting_human_review',
                    checkpoint_confirmed=True,
                ):
                    raise ReviewWorkflowServiceError('concurrent_resume')
                return self.status(actor=actor, thread_id=thread_id)

            try:
                probe = self._probe_checkpoint(
                    projection=projection,
                    builder=builder,
                    saver=saver,
                )
            except (
                CheckpointUnavailableError,
                _MissingCheckpointError,
                _CorruptCheckpointError,
            ):
                raise ReviewWorkflowServiceError(
                    'checkpoint_unavailable'
                ) from None
            if probe.interrupt_state_version is None:
                raise ReviewWorkflowServiceError('checkpoint_unavailable')
            if not self._cas_status(
                projection=projection,
                target_status='resuming',
            ):
                raise ReviewWorkflowServiceError('concurrent_resume')
            try:
                claimed = self._read_projection(
                    actor=actor,
                    thread_id=thread_id,
                    action='resume',
                    require_current_versions=True,
                )
                fresh_saver = require_thread_checkpoint_runtime(
                    thread=claimed,
                    runtime=self._checkpoint_runtime,
                )
                fresh_graph = builder(fresh_saver)
                confirmation = invoke_and_confirm_checkpoint(
                    graph=fresh_graph,
                    saver=fresh_saver,
                    command_or_input=Command(resume={
                        'event': 'review_resolution_checked',
                        'state_version': claimed.state_version,
                    }),
                    checkpoint_thread_id=claimed.checkpoint_thread_id,
                    runtime_context=self._runtime_context(actor),
                    expect_interrupt=False,
                )
            except ReviewWorkflowServiceError:
                self._mark_status(thread_id, 'awaiting_human_review')
                raise
            except CheckpointUnavailableError:
                self._restore_after_resume_failure(claimed)
                raise ReviewWorkflowServiceError('checkpoint_unavailable') from None
            except CheckpointConfirmationError:
                self._mark_checkpoint_failed(thread_id)
                raise ReviewWorkflowServiceError('checkpoint_failed') from None
            except PermissionError:
                self._restore_after_resume_failure(claimed)
                raise ReviewWorkflowServiceError('not_found') from None
            except ValueError as exc:
                self._restore_after_resume_failure(claimed)
                self._raise_value_error(exc)
            except Exception:
                self._mark_checkpoint_failed(thread_id)
                raise ReviewWorkflowServiceError('checkpoint_failed') from None

            result_status = confirmation.result.get('status')
            if result_status not in {'completed', 'needs_more_evidence'}:
                self._mark_checkpoint_failed(thread_id)
                raise ReviewWorkflowServiceError('checkpoint_failed')
            current = self._read_projection(
                actor=actor,
                thread_id=thread_id,
                action='resume',
            )
            if current.status != 'resuming' or not self._cas_status(
                projection=current,
                target_status=cast(str, result_status),
                completed=True,
            ):
                raise ReviewWorkflowServiceError('concurrent_resume')
            return self.status(actor=actor, thread_id=thread_id)

    def cancel(
        self,
        *,
        actor: DemoUser,
        thread_id: str,
    ) -> ReviewWorkflowStatus:
        with self._lifecycle_lock:
            projection = self._read_projection(
                actor=actor,
                thread_id=thread_id,
                action='cancel',
            )
            if projection.status not in _CANCELLABLE_THREAD_STATUSES:
                raise ReviewWorkflowServiceError('invalid_state_transition')
            now = self._now()
            if not self._cas_status(
                projection=projection,
                target_status='cancelled',
                completed=True,
                cancellation=(now, actor.id),
            ):
                raise ReviewWorkflowServiceError('concurrent_resume')
            return self.status(actor=actor, thread_id=thread_id)

    def _require_new_run_availability(self) -> None:
        if not self._settings.langgraph_review_v2_enabled:
            raise ReviewWorkflowServiceError('not_found')
        if not self._model_readiness.ready:
            raise ReviewWorkflowServiceError('model_unavailable')
        readiness = self._checkpoint_runtime.readiness
        if (
            not readiness.ready
            or self._checkpoint_runtime.saver is None
            or readiness.mode == 'disabled'
        ):
            raise ReviewWorkflowServiceError('checkpoint_unavailable')

    def _draft_with_bounded_retry(
        self,
        *,
        thread_id: str,
        actor: DemoUser,
    ) -> ReviewDraftResult:
        for attempt in range(2):
            try:
                return self._draft_service.draft(
                    workflow_thread_id=thread_id,
                    actor_subject_id=actor.id,
                    allowed_permission_levels=tuple(actor.permission_levels),
                )
            except ReviewDraftError as exc:
                if exc.code != 'model_unavailable':
                    self._raise_preflight_error(exc)
                if attempt == 1:
                    self._mark_failed(thread_id)
                    raise ReviewWorkflowServiceError('model_unavailable') from None
        raise AssertionError('unreachable')

    def _runtime_context(
        self,
        actor: DemoUser,
        *,
        draft_service: object | None = None,
    ) -> dict[str, object]:
        allowed = tuple(sorted(actor.permission_levels))

        def permission_resolver(subject_id: str) -> Sequence[str]:
            return allowed if subject_id == actor.id else ()

        return {
            'session_factory': self._session_factory,
            'actor_subject_id': actor.id,
            'permission_resolver': permission_resolver,
            'agent_registry': self._agent_registry,
            'draft_service': draft_service or self._draft_service,
            'lease_service': _UnusedLeaseService(),
        }

    def _read_projection(
        self,
        *,
        actor: DemoUser,
        thread_id: str,
        action: str | None = None,
        require_current_versions: bool = False,
    ) -> _ThreadProjection:
        with self._session_factory() as db:
            thread = db.get(AgentWorkflowThread, thread_id)
            if (
                thread is None
                or thread.security_scope_id
                != self._settings.agent_runtime_security_scope_id
                or thread.workflow_name != COMPANY_MEMORY_REVIEW_WORKFLOW
            ):
                db.rollback()
                raise ReviewWorkflowServiceError('not_found')
            refs = tuple(
                db.scalars(
                    select(AgentWorkflowEvidenceRef)
                    .where(
                        AgentWorkflowEvidenceRef.workflow_thread_id == thread_id
                    )
                    .order_by(AgentWorkflowEvidenceRef.ordinal)
                ).all()
            )
            if not refs or (
                not require_current_versions
                and any(ref.canonical_table != 'sources' for ref in refs)
            ):
                db.rollback()
                raise ReviewWorkflowServiceError('not_found')
            source_ids = tuple(ref.canonical_row_id for ref in refs)
            sources = tuple(
                db.scalars(select(Source).where(Source.id.in_(source_ids))).all()
            )
            by_id = {source.id: source for source in sources}
            if len(by_id) != len(source_ids):
                db.rollback()
                raise ReviewWorkflowServiceError('not_found')
            for ref in refs:
                source = by_id[ref.canonical_row_id]
                if (
                    source.permission_level not in actor.permission_levels
                    or (
                        not require_current_versions
                        and source.source_type != ref.canonical_source_type
                    )
                ):
                    db.rollback()
                    raise ReviewWorkflowServiceError('not_found')
            if require_current_versions:
                canonical_refs = tuple(
                    SourceVersionRef(
                        source_type=cast(
                            CanonicalSourceType,
                            ref.canonical_source_type,
                        ),
                        source_id=by_id[ref.canonical_row_id].source_id,
                        version_or_signature=ref.content_signature,
                    )
                    for ref in refs
                )
                try:
                    resolved_refs = resolve_source_versions(
                        db,
                        refs=canonical_refs,
                        actor=actor,
                        settings=self._settings,
                    )
                except ReviewWorkflowPreflightError as exc:
                    db.rollback()
                    if exc.code == 'not_found':
                        raise ReviewWorkflowServiceError('not_found') from None
                    raise ReviewWorkflowServiceError('evidence_changed') from None
                for stored, resolved in zip(refs, resolved_refs, strict=True):
                    if (
                        stored.canonical_source_type != resolved.source_type
                        or stored.canonical_table != resolved.canonical_table
                        or stored.canonical_row_id != resolved.canonical_row_id
                        or stored.document_version_id
                        != resolved.document_version_id
                        or stored.external_revision != resolved.external_revision
                        or stored.content_signature != resolved.content_signature
                        or stored.permission_level_snapshot
                        != resolved.permission_level
                        or stored.content_fingerprint
                        != resolved.content_fingerprint
                    ):
                        db.rollback()
                        raise ReviewWorkflowServiceError('evidence_changed')
            items = tuple(
                db.scalars(
                    select(ReviewItem)
                    .where(ReviewItem.workflow_thread_id == thread_id)
                    .order_by(ReviewItem.id)
                ).all()
            )
            if any(
                item.permission_level not in actor.permission_levels
                for item in items
            ):
                db.rollback()
                raise ReviewWorkflowServiceError('not_found')
            self._authorize_action(thread, actor=actor, action=action)
            counts: dict[ReviewItemResolutionStatus, int] = {}
            for item in items:
                if item.status not in _REVIEW_STATUSES:
                    db.rollback()
                    raise ReviewWorkflowServiceError('invalid_state_transition')
                status = cast(ReviewItemResolutionStatus, item.status)
                counts[status] = counts.get(status, 0) + 1
            projection = _ThreadProjection(
                thread_id=thread.thread_id,
                workflow_name=thread.workflow_name,
                graph_version=thread.graph_version,
                input_hash=thread.input_hash,
                evidence_version_hash=thread.evidence_version_hash,
                checkpoint_thread_id=thread.checkpoint_thread_id,
                checkpoint_store=thread.checkpoint_store,
                owner_subject_id=thread.owner_subject_id,
                status=thread.status,
                state_version=thread.state_version,
                review_item_ids=tuple(item.id for item in items),
                review_status_counts=counts,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
            db.rollback()
            return projection

    @staticmethod
    def _authorize_action(
        thread: AgentWorkflowThread,
        *,
        actor: DemoUser,
        action: str | None,
    ) -> None:
        if action is None:
            return
        owner = thread.owner_subject_id == actor.id
        allowed = owner
        if action == 'resume':
            allowed = allowed or actor.role in {'reviewer', 'admin'}
        elif action == 'cancel':
            allowed = allowed or actor.role == 'admin'
        if not allowed:
            raise ReviewWorkflowServiceError('not_found')

    def _resolve_builder(self, projection: _ThreadProjection):
        try:
            return self._graph_registry.resolve(
                projection.workflow_name,
                projection.graph_version,
            )
        except RuntimeVersionUnavailable:
            raise ReviewWorkflowServiceError(
                'runtime_version_unavailable'
            ) from None

    def _exact_saver_or_error(
        self,
        projection: _ThreadProjection,
    ) -> BaseCheckpointSaver:
        try:
            return require_thread_checkpoint_runtime(
                thread=projection,
                runtime=self._checkpoint_runtime,
            )
        except CheckpointUnavailableError:
            raise ReviewWorkflowServiceError(
                'checkpoint_unavailable'
            ) from None

    @staticmethod
    def _probe_checkpoint(
        *,
        projection: _ThreadProjection,
        builder,
        saver: BaseCheckpointSaver,
    ) -> _CheckpointProbe:
        config = checkpoint_config(projection.checkpoint_thread_id)
        try:
            saved = saver.get_tuple(config)
        except Exception:
            raise CheckpointUnavailableError('checkpoint_unavailable') from None
        if saved is None:
            raise _MissingCheckpointError
        try:
            saved_config = saved.config
            configurable = saved_config['configurable']
            saved_identity = (
                configurable['thread_id'],
                configurable.get('checkpoint_ns', ''),
                configurable['checkpoint_id'],
            )
        except (AttributeError, KeyError, TypeError):
            raise _CorruptCheckpointError from None
        if (
            not isinstance(saved_config, Mapping)
            or not isinstance(configurable, Mapping)
            or saved_identity[0] != projection.checkpoint_thread_id
            or saved_identity[1] != ''
            or type(saved_identity[2]) is not str
            or not saved_identity[2]
        ):
            raise _CorruptCheckpointError
        try:
            require_resumable_checkpoint(
                saver,
                projection.checkpoint_thread_id,
            )
            graph = builder(saver)
            snapshot = graph.get_state(config)
        except CheckpointUnavailableError:
            raise
        except Exception:
            raise CheckpointUnavailableError('checkpoint_unavailable') from None
        try:
            snapshot_config = snapshot.config
            snapshot_values = snapshot_config['configurable']
            snapshot_identity = (
                snapshot_values['thread_id'],
                snapshot_values.get('checkpoint_ns', ''),
                snapshot_values['checkpoint_id'],
            )
            snapshot_next = tuple(snapshot.next)
            snapshot_state = snapshot.values
            interrupts = tuple(
                pending
                for task in snapshot.tasks
                for pending in getattr(task, 'interrupts', ())
            )
        except (AttributeError, KeyError, TypeError):
            raise _CorruptCheckpointError from None
        if (
            not isinstance(snapshot_config, Mapping)
            or not isinstance(snapshot_values, Mapping)
            or snapshot_identity != saved_identity
            or not isinstance(snapshot_state, Mapping)
        ):
            raise _CorruptCheckpointError
        expected_state_identity = {
            'workflow_thread_id': projection.thread_id,
            'graph_version': projection.graph_version,
            'input_hash': projection.input_hash,
            'evidence_version_hash': projection.evidence_version_hash,
        }
        if any(
            snapshot_state.get(field) != expected
            for field, expected in expected_state_identity.items()
        ):
            raise _CorruptCheckpointError
        snapshot_counts = snapshot_state.get('review_status_counts')
        snapshot_item_ids = snapshot_state.get('review_item_ids')
        if (
            not isinstance(snapshot_counts, Mapping)
            or set(snapshot_counts) != set(_REVIEW_STATUSES)
            or any(
                type(count) is not int or count < 0
                for count in snapshot_counts.values()
            )
            or sum(snapshot_counts.values()) != projection.review_item_count
            or type(snapshot_item_ids) is not list
            or any(type(item_id) is not int for item_id in snapshot_item_ids)
        ):
            raise _CorruptCheckpointError
        if not interrupts:
            if snapshot_next:
                raise _CorruptCheckpointError
            stored_status = snapshot_state.get('status')
            if stored_status not in {'completed', 'needs_more_evidence'}:
                raise _CorruptCheckpointError
            if projection.review_status_counts.get('pending_review', 0):
                expected_terminal_status = None
            elif projection.review_status_counts.get(
                'needs_more_evidence',
                0,
            ):
                expected_terminal_status = 'needs_more_evidence'
            else:
                expected_terminal_status = 'completed'
            if stored_status != expected_terminal_status:
                raise _CorruptCheckpointError
            return _CheckpointProbe(
                interrupt_state_version=None,
                terminal_status=cast(str, stored_status),
            )
        if (
            projection.review_item_count == 0
            or snapshot_next != ('await_human_review',)
        ):
            raise _CorruptCheckpointError
        if len(interrupts) != 1 or type(interrupts[0]) is not Interrupt:
            raise _CorruptCheckpointError
        payload = interrupts[0].value
        state_version = payload.get('state_version') if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get('event') != 'review_resolution_required'
            or type(state_version) is not int
        ):
            raise _CorruptCheckpointError
        return _CheckpointProbe(
            interrupt_state_version=state_version,
            terminal_status=None,
        )

    def _repair_checkpoint(
        self,
        *,
        projection: _ThreadProjection,
        actor: DemoUser,
        builder,
    ) -> ReviewWorkflowStatus:
        rotated = self._rotate_checkpoint_attempt(projection)
        if rotated is None:
            raise ReviewWorkflowServiceError('concurrent_resume')
        try:
            fresh = self._read_projection(
                actor=actor,
                thread_id=projection.thread_id,
                action='resume',
                require_current_versions=True,
            )
            prepared_result = ReviewDraftResult(
                review_item_ids=fresh.review_item_ids,
                review_status_counts=dict(fresh.review_status_counts),
            )
            fresh_saver = require_thread_checkpoint_runtime(
                thread=fresh,
                runtime=self._checkpoint_runtime,
            )
            graph = builder(fresh_saver)
            confirmation = invoke_and_confirm_checkpoint(
                graph=graph,
                saver=fresh_saver,
                command_or_input={'workflow_thread_id': fresh.thread_id},
                checkpoint_thread_id=fresh.checkpoint_thread_id,
                runtime_context=self._runtime_context(
                    actor,
                    draft_service=_PreparedDraftService(
                        workflow_thread_id=fresh.thread_id,
                        result=prepared_result,
                    ),
                ),
                expect_interrupt=bool(fresh.review_item_ids),
            )
        except CheckpointUnavailableError:
            self._mark_checkpoint_failed(projection.thread_id)
            raise ReviewWorkflowServiceError('checkpoint_unavailable') from None
        except ReviewWorkflowServiceError:
            self._mark_checkpoint_failed(projection.thread_id)
            raise
        except CheckpointConfirmationError:
            self._mark_checkpoint_failed(projection.thread_id)
            raise ReviewWorkflowServiceError('checkpoint_failed') from None
        except (ReviewDraftError, ValueError, PermissionError) as exc:
            self._mark_checkpoint_failed(projection.thread_id)
            if isinstance(exc, PermissionError):
                raise ReviewWorkflowServiceError('not_found') from None
            if isinstance(exc, ReviewDraftError):
                self._raise_preflight_error(exc)
            self._raise_value_error(exc)
        except Exception:
            self._mark_checkpoint_failed(projection.thread_id)
            raise ReviewWorkflowServiceError('checkpoint_failed') from None
        current = self._read_projection(
            actor=actor,
            thread_id=projection.thread_id,
            action='resume',
        )
        if fresh.review_item_ids:
            target_status = 'awaiting_human_review'
        elif confirmation.result.get('status') == 'completed':
            target_status = 'completed'
        else:
            self._mark_checkpoint_failed(projection.thread_id)
            raise ReviewWorkflowServiceError('checkpoint_failed')
        if current.status != 'checkpoint_pending' or not self._cas_status(
            projection=current,
            target_status=target_status,
            checkpoint_confirmed=True,
            completed=target_status == 'completed',
        ):
            raise ReviewWorkflowServiceError('checkpoint_failed')
        return self.status(actor=actor, thread_id=projection.thread_id)

    def _rotate_checkpoint_attempt(
        self,
        projection: _ThreadProjection,
    ) -> _ThreadProjection | None:
        now = self._now()
        with self._session_factory() as db:
            result = db.execute(
                update(AgentWorkflowThread)
                .where(
                    AgentWorkflowThread.thread_id == projection.thread_id,
                    AgentWorkflowThread.state_version == projection.state_version,
                    AgentWorkflowThread.status == 'checkpoint_failed',
                )
                .values(
                    checkpoint_thread_id=f'review-v2:{uuid4().hex}',
                    checkpoint_confirmed_at=None,
                    status='checkpoint_pending',
                    state_version=projection.state_version + 1,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                db.rollback()
                return None
            db.commit()
        with self._session_factory() as db:
            thread = db.get(AgentWorkflowThread, projection.thread_id)
            assert thread is not None
            rotated = _ThreadProjection(
                **{
                    **projection.__dict__,
                    'checkpoint_thread_id': thread.checkpoint_thread_id,
                    'status': thread.status,
                    'state_version': thread.state_version,
                    'updated_at': thread.updated_at,
                }
            )
            db.rollback()
        return rotated

    def _ensure_checkpoint_pending(self, thread_id: str) -> None:
        now = self._now()
        with self._session_factory() as db:
            thread = db.get(AgentWorkflowThread, thread_id)
            if thread is None:
                db.rollback()
                raise ReviewWorkflowServiceError('not_found')
            if thread.status == 'checkpoint_pending':
                db.rollback()
                return
            if thread.status not in {'created', 'drafting', 'checkpoint_failed'}:
                db.rollback()
                raise ReviewWorkflowServiceError('invalid_state_transition')
            expected_version = thread.state_version
            result = db.execute(
                update(AgentWorkflowThread)
                .where(
                    AgentWorkflowThread.thread_id == thread_id,
                    AgentWorkflowThread.state_version == expected_version,
                    AgentWorkflowThread.status == thread.status,
                )
                .values(
                    status='checkpoint_pending',
                    state_version=expected_version + 1,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                db.rollback()
                raise ReviewWorkflowServiceError('concurrent_resume')
            db.commit()

    def _finish_initial_checkpoint(
        self,
        *,
        projection: _ThreadProjection,
        target_status: str,
    ) -> bool:
        return projection.status == 'checkpoint_pending' and self._cas_status(
            projection=projection,
            target_status=target_status,
            checkpoint_confirmed=True,
            completed=target_status == 'completed',
        )

    def _cas_status(
        self,
        *,
        projection: _ThreadProjection,
        target_status: str,
        checkpoint_confirmed: bool = False,
        completed: bool = False,
        cancellation: tuple[datetime, str] | None = None,
    ) -> bool:
        now = self._now()
        values: dict[str, object] = {
            'status': target_status,
            'state_version': projection.state_version + 1,
            'updated_at': now,
        }
        if checkpoint_confirmed:
            values['checkpoint_confirmed_at'] = now
        if completed:
            values['completed_at'] = now
        if cancellation is not None:
            values['cancelled_at'] = cancellation[0]
            values['cancelled_by_subject_id'] = cancellation[1]
        with self._session_factory() as db:
            result = db.execute(
                update(AgentWorkflowThread)
                .where(
                    AgentWorkflowThread.thread_id == projection.thread_id,
                    AgentWorkflowThread.state_version == projection.state_version,
                    AgentWorkflowThread.status == projection.status,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                db.rollback()
                return False
            db.commit()
            return True

    def _mark_failed(self, thread_id: str) -> None:
        self._mark_status(thread_id, 'failed', completed=True)

    def _mark_checkpoint_failed(self, thread_id: str) -> None:
        self._mark_status(thread_id, 'checkpoint_failed')

    def _restore_after_resume_failure(self, projection: _ThreadProjection) -> None:
        self._cas_status(
            projection=projection,
            target_status='awaiting_human_review',
        )

    def _mark_status(
        self,
        thread_id: str,
        target_status: str,
        *,
        completed: bool = False,
    ) -> None:
        with self._session_factory() as db:
            thread = db.get(AgentWorkflowThread, thread_id)
            if thread is None or thread.status in _TERMINAL_THREAD_STATUSES:
                db.rollback()
                return
            projection = _ThreadProjection(
                thread_id=thread.thread_id,
                workflow_name=thread.workflow_name,
                graph_version=thread.graph_version,
                input_hash=thread.input_hash,
                evidence_version_hash=thread.evidence_version_hash,
                checkpoint_thread_id=thread.checkpoint_thread_id,
                checkpoint_store=thread.checkpoint_store,
                owner_subject_id=thread.owner_subject_id,
                status=thread.status,
                state_version=thread.state_version,
                review_item_ids=(),
                review_status_counts={},
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
            db.rollback()
        self._cas_status(
            projection=projection,
            target_status=target_status,
            completed=completed,
        )

    def _status_response(
        self,
        projection: _ThreadProjection,
        *,
        actor: DemoUser,
        checkpoint_resumable: bool,
        error_code: ReviewWorkflowErrorCode | None,
        resume_error_code: ReviewWorkflowErrorCode | None,
    ) -> ReviewWorkflowStatus:
        resume_authorized = self._action_allowed(
            projection,
            actor=actor,
            action='resume',
        )
        exact_runtime_ready = (
            projection.checkpoint_store
            == self._checkpoint_runtime.readiness.checkpoint_store
            and self._checkpoint_runtime.readiness.ready
            and self._checkpoint_runtime.saver is not None
        )
        retry_allowed = (
            projection.status == 'checkpoint_failed'
            and resume_authorized
            and exact_runtime_ready
            and error_code != 'runtime_version_unavailable'
        )
        return ReviewWorkflowStatus(
            thread_id=projection.thread_id,
            status=projection.status,
            review_item_count=projection.review_item_count,
            review_status_counts=dict(projection.review_status_counts),
            durable=projection.checkpoint_store == 'postgres',
            graph_version=projection.graph_version,
            review_resolution_ready=projection.review_resolution_ready,
            checkpoint_resumable=checkpoint_resumable,
            resume_allowed=(
                projection.status == 'awaiting_human_review'
                and projection.review_resolution_ready
                and checkpoint_resumable
                and resume_authorized
            ),
            retry_allowed=retry_allowed,
            created_at=projection.created_at,
            updated_at=projection.updated_at,
            error_code=error_code,
            resume_error_code=resume_error_code,
        )

    @staticmethod
    def _action_allowed(
        projection: _ThreadProjection,
        *,
        actor: DemoUser,
        action: str,
    ) -> bool:
        if projection.owner_subject_id == actor.id:
            return True
        if action == 'resume':
            return actor.role in {'reviewer', 'admin'}
        if action == 'cancel':
            return actor.role == 'admin'
        return False

    @staticmethod
    def _thread_error_code(status: str) -> ReviewWorkflowErrorCode | None:
        if status == 'checkpoint_failed':
            return 'checkpoint_failed'
        if status == 'failed':
            return 'model_unavailable'
        return None

    @staticmethod
    def _raise_preflight_error(exc: ReviewWorkflowPreflightError) -> None:
        code = exc.code
        if code == 'permission_denied':
            code = 'not_found'
        raise ReviewWorkflowServiceError(code) from None

    @staticmethod
    def _raise_value_error(exc: ValueError) -> None:
        code = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ''
        if code == 'permission_denied':
            code = 'not_found'
        raise ReviewWorkflowServiceError(code) from None
