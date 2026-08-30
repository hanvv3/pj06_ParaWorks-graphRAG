from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_call_recovery import (
    AutoReviewCallRecoveryService,
)
from backend.app.agent_runtime.checkpoint_execution import (
    CheckpointConfirmationError,
    checkpoint_config,
    invoke_and_confirm_checkpoint,
)
from backend.app.agent_runtime.checkpointing import (
    CheckpointRuntime,
    CheckpointUnavailableError,
)
from backend.app.agent_runtime.graph_versions import GraphVersionRegistry
from backend.app.agent_runtime.review_v2_service import (
    ReviewWorkflowServiceError,
)
from backend.app.agent_runtime.review_v21_state import (
    REVIEW_STATUS_ORDER_V21,
    validate_v21_checkpoint_state,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import (
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AuditLog,
    ReviewItem,
)
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
from backend.app.schemas.review_workflow import ReviewWorkflowErrorCode


@dataclass(frozen=True)
class ReviewWorkflowStatusV21:
    thread_id: str
    status: str
    review_item_count: int
    review_status_counts: dict[str, int]
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
    auto_review_mode: str
    auto_review_policy_version: str
    auto_review_enforce_percentage: int
    auto_approved_count: int
    human_review_required_count: int
    auto_review_fallback_count: int


class _UnavailableExtractionCoordinator:
    def draft(self, **_kwargs) -> None:
        raise ValueError('model_unavailable')


class _ZeroCallValidationCoordinator:
    def run_auto_review(self, **_kwargs) -> None:
        return None


class ReviewV21Service:
    """Dedicated lifecycle for the immutable V2.1 checkpoint schema.

    New-run preparation and signed launch remain owned by the Task 11 launch
    service. This class owns only V2.1 graph execution and five-status
    projection; it never uses the V2.0 checkpoint validator or mapper.
    """

    def __init__(
        self,
        *,
        launch_service: object,
        session_factory: Callable[[], Session],
        settings: Settings,
        checkpoint_runtime: CheckpointRuntime,
        graph_registry: GraphVersionRegistry,
        agent_registry: object,
        extraction_coordinator: object | None = None,
        validation_coordinator: object | None = None,
        permission_resolver: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self._launch = launch_service
        self._session_factory = session_factory
        self._settings = settings
        self._checkpoint_runtime = checkpoint_runtime
        self._graph_registry = graph_registry
        self._agent_registry = agent_registry
        self._extraction = extraction_coordinator or _UnavailableExtractionCoordinator()
        self._validation = validation_coordinator or _ZeroCallValidationCoordinator()
        self._permission_resolver = permission_resolver

    def dry_run(self, *, actor: DemoUser, request: object) -> object:
        return self._launch.dry_run(actor=actor, request=request)

    def diagnostic(self) -> object:
        diagnostic = self._launch.diagnostic()
        return diagnostic.model_copy(
            update={'graph_version': COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21}
        )

    def start(self, *, actor: DemoUser, request: object) -> ReviewWorkflowStatusV21:
        launched = self._launch.start(actor=actor, request=request)
        if launched.graph_version != COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21:
            raise ReviewWorkflowServiceError('runtime_version_unavailable')
        if launched.status != 'created':
            return self.status(actor=actor, thread_id=launched.thread_id)
        return self._invoke_initial(actor=actor, thread_id=launched.thread_id)

    def status(
        self, *, actor: DemoUser, thread_id: str
    ) -> ReviewWorkflowStatusV21:
        thread, request, counts = self._projection(actor=actor, thread_id=thread_id)
        resumable = False
        terminal: str | None = None
        saver = self._saver(thread)
        if saver is not None:
            try:
                graph = self._builder()(saver)
                snapshot = graph.get_state(checkpoint_config(thread.checkpoint_thread_id))
                if snapshot.values:
                    validate_v21_checkpoint_state(snapshot.values)
                    resumable = any(
                        getattr(task, 'interrupts', ()) for task in snapshot.tasks
                    )
                    self._validate_live_reconciliation(
                        thread_id=thread_id,
                        checkpoint_counts=cast(
                            dict[str, int],
                            snapshot.values['review_status_counts'],
                        ),
                        live_counts=counts,
                        interrupted=resumable,
                    )
                    if not resumable:
                        output = snapshot.values
                        phase = output.get('phase')
                        if phase in {'completed', 'needs_more_evidence'}:
                            terminal = cast(str, phase)
            except ReviewWorkflowServiceError:
                raise
            except Exception:
                resumable = False
        if thread.status in {
            'created', 'checkpoint_pending', 'resuming', 'checkpoint_failed'
        }:
            target = (
                'awaiting_human_review' if resumable else terminal
            )
            if target is not None:
                self._set_status(
                    thread.thread_id,
                    target,
                    completed=not resumable,
                    bump_version=False,
                )
                thread, request, counts = self._projection(
                    actor=actor, thread_id=thread_id
                )
        pending = counts['pending_review']
        ready = sum(counts.values()) > 0 and pending == 0
        return ReviewWorkflowStatusV21(
            thread_id=thread.thread_id,
            status=thread.status,
            review_item_count=sum(counts.values()),
            review_status_counts=counts,
            durable=thread.checkpoint_store == 'postgres',
            graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
            review_resolution_ready=ready,
            checkpoint_resumable=resumable,
            resume_allowed=(
                thread.status == 'awaiting_human_review'
                and ready
                and resumable
            ),
            retry_allowed=False,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
            error_code=(
                'checkpoint_failed' if thread.status == 'checkpoint_failed' else None
            ),
            resume_error_code=None,
            auto_review_mode=cast(str, request.auto_review_mode),
            auto_review_policy_version=cast(str, request.auto_review_policy_version),
            auto_review_enforce_percentage=cast(
                int, request.auto_review_enforce_percentage
            ),
            auto_approved_count=self._auto_approved_count(thread_id),
            human_review_required_count=pending,
            auto_review_fallback_count=pending,
        )

    def resume(
        self, *, actor: DemoUser, thread_id: str
    ) -> ReviewWorkflowStatusV21:
        # Reconcile the live rows against the immutable interrupted checkpoint
        # before allowing the graph to consume the acknowledgement.
        self.status(actor=actor, thread_id=thread_id)
        thread, _, counts = self._projection(actor=actor, thread_id=thread_id)
        if thread.status != 'awaiting_human_review':
            raise ReviewWorkflowServiceError('invalid_state_transition')
        if counts['pending_review']:
            raise ReviewWorkflowServiceError('review_unresolved')
        saver = self._require_saver(thread)
        graph = self._builder()(saver)
        self._set_status(thread_id, 'resuming', bump_version=False)
        try:
            confirmation = invoke_and_confirm_checkpoint(
                graph=graph,
                saver=saver,
                command_or_input=Command(resume={
                    'event': 'review_resolution_checked',
                    'state_version': thread.state_version,
                }),
                checkpoint_thread_id=thread.checkpoint_thread_id,
                runtime_context=self._runtime_context(actor),
                expect_interrupt=None,
            )
        except (CheckpointConfirmationError, CheckpointUnavailableError):
            self._set_status(thread_id, 'checkpoint_failed')
            raise ReviewWorkflowServiceError('checkpoint_failed') from None
        if confirmation.interrupted:
            target = 'awaiting_human_review'
        else:
            target = cast(str, confirmation.result.get('status'))
            if target not in {'completed', 'needs_more_evidence'}:
                self._set_status(thread_id, 'checkpoint_failed')
                raise ReviewWorkflowServiceError('checkpoint_failed')
        self._set_status(
            thread_id,
            target,
            completed=not confirmation.interrupted,
            bump_version=False,
        )
        return self.status(actor=actor, thread_id=thread_id)

    def cancel(
        self, *, actor: DemoUser, thread_id: str
    ) -> ReviewWorkflowStatusV21:
        thread, _, _ = self._projection(actor=actor, thread_id=thread_id)
        if actor.role != 'admin' and thread.owner_subject_id != actor.id:
            raise ReviewWorkflowServiceError('not_found')
        if thread.status in {'completed', 'needs_more_evidence', 'cancelled'}:
            raise ReviewWorkflowServiceError('invalid_state_transition')
        with self._session_factory() as db, db.begin():
            current = db.get(AgentWorkflowThread, thread_id)
            if current is None:
                raise ReviewWorkflowServiceError('not_found')
            now = datetime.now(UTC)
            current.status = 'cancelled'
            current.cancelled_at = now
            current.cancelled_by_subject_id = actor.id
            current.updated_at = now
        AutoReviewCallRecoveryService(
            session_factory=self._session_factory,
        ).cancel_workflow(workflow_thread_id=thread_id)
        return self.status(actor=actor, thread_id=thread_id)

    def _invoke_initial(
        self, *, actor: DemoUser, thread_id: str
    ) -> ReviewWorkflowStatusV21:
        thread, _, _ = self._projection(actor=actor, thread_id=thread_id)
        saver = self._require_saver(thread)
        graph = self._builder()(saver)
        self._set_status(thread_id, 'checkpoint_pending')
        try:
            confirmation = invoke_and_confirm_checkpoint(
                graph=graph,
                saver=saver,
                command_or_input={'workflow_thread_id': thread_id},
                checkpoint_thread_id=thread.checkpoint_thread_id,
                runtime_context=self._runtime_context(actor),
                expect_interrupt=None,
            )
        except ValueError as exc:
            self._set_status(thread_id, 'failed')
            code = str(exc) if str(exc) in {'model_unavailable'} else 'invalid_state_transition'
            raise ReviewWorkflowServiceError(code) from None
        except (CheckpointConfirmationError, CheckpointUnavailableError):
            self._set_status(thread_id, 'checkpoint_failed')
            raise ReviewWorkflowServiceError('checkpoint_failed') from None
        target = (
            'awaiting_human_review'
            if confirmation.interrupted
            else cast(str, confirmation.result.get('status'))
        )
        if target not in {
            'awaiting_human_review', 'completed', 'needs_more_evidence'
        }:
            self._set_status(thread_id, 'checkpoint_failed')
            raise ReviewWorkflowServiceError('checkpoint_failed')
        self._set_status(
            thread_id,
            target,
            completed=target != 'awaiting_human_review',
            bump_version=False,
        )
        return self.status(actor=actor, thread_id=thread_id)

    def _runtime_context(self, actor: DemoUser) -> dict[str, object]:
        def permissions(owner: str) -> Sequence[str]:
            if self._permission_resolver is not None:
                return self._permission_resolver(owner)
            return tuple(actor.permission_levels) if owner == actor.id else ()

        return {
            'session_factory': self._session_factory,
            'permission_resolver': permissions,
            'agent_registry': self._agent_registry,
            'extraction_coordinator': self._extraction,
            'validation_coordinator': self._validation,
        }

    def _projection(self, *, actor: DemoUser, thread_id: str):
        with self._session_factory() as db:
            thread = db.get(AgentWorkflowThread, thread_id)
            request = db.get(AgentWorkflowRequest, thread_id)
            if (
                thread is None
                or request is None
                or thread.graph_version != COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
                or thread.owner_subject_id != actor.id
                or thread.security_scope_id
                != self._settings.agent_runtime_security_scope_id
            ):
                db.rollback()
                raise ReviewWorkflowServiceError('not_found')
            items = tuple(db.scalars(select(ReviewItem).where(
                ReviewItem.workflow_thread_id == thread_id
            ).order_by(ReviewItem.id)).all())
            counts = dict.fromkeys(REVIEW_STATUS_ORDER_V21, 0)
            for item in items:
                if (
                    item.status not in counts
                    or item.permission_level not in actor.permission_levels
                ):
                    db.rollback()
                    raise ReviewWorkflowServiceError('not_found')
                counts[item.status] += 1
            db.expunge(thread)
            db.expunge(request)
            db.rollback()
        return thread, request, counts

    def _builder(self):
        try:
            return self._graph_registry.resolve(
                'company-memory-review', COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
            )
        except Exception:
            raise ReviewWorkflowServiceError('runtime_version_unavailable') from None

    def _saver(self, thread: AgentWorkflowThread):
        readiness = self._checkpoint_runtime.readiness
        if (
            readiness.ready
            and readiness.checkpoint_store == thread.checkpoint_store
            and self._checkpoint_runtime.saver is not None
        ):
            return self._checkpoint_runtime.saver
        return None

    def _require_saver(self, thread: AgentWorkflowThread):
        saver = self._saver(thread)
        if saver is None:
            raise ReviewWorkflowServiceError('checkpoint_unavailable')
        return saver

    def _set_status(
        self,
        thread_id: str,
        status: str,
        *,
        completed: bool = False,
        bump_version: bool = True,
    ) -> None:
        with self._session_factory() as db, db.begin():
            thread = db.get(AgentWorkflowThread, thread_id)
            if thread is None:
                raise ReviewWorkflowServiceError('not_found')
            now = datetime.now(UTC)
            thread.status = status
            if bump_version:
                thread.state_version += 1
            thread.updated_at = now
            if status in {'awaiting_human_review', 'completed', 'needs_more_evidence'}:
                thread.checkpoint_confirmed_at = now
            if completed:
                thread.completed_at = now

    def _auto_approved_count(self, thread_id: str) -> int:
        with self._session_factory() as db:
            count = len(tuple(db.scalars(select(ReviewItem.id).where(
                ReviewItem.workflow_thread_id == thread_id,
                ReviewItem.status == 'approved',
                ReviewItem.resolution_source == 'auto_policy',
            )).all()))
            db.rollback()
            return count

    def _validate_live_reconciliation(
        self,
        *,
        thread_id: str,
        checkpoint_counts: dict[str, int],
        live_counts: dict[str, int],
        interrupted: bool,
    ) -> None:
        if checkpoint_counts == live_counts:
            return
        if sum(checkpoint_counts.values()) != sum(live_counts.values()):
            raise ReviewWorkflowServiceError('invalid_state_transition')
        approved_to_revoked = live_counts['revoked'] - checkpoint_counts['revoked']
        if approved_to_revoked < 0 or (
            checkpoint_counts['approved'] - live_counts['approved']
            < approved_to_revoked
        ):
            raise ReviewWorkflowServiceError('invalid_state_transition')
        adjusted_checkpoint = dict(checkpoint_counts)
        adjusted_checkpoint['approved'] -= approved_to_revoked
        adjusted_checkpoint['revoked'] += approved_to_revoked
        if approved_to_revoked:
            self._verify_revoked_rows(thread_id=thread_id)
        if not interrupted:
            if adjusted_checkpoint != live_counts:
                raise ReviewWorkflowServiceError('invalid_state_transition')
            return
        pending_delta = adjusted_checkpoint['pending_review'] - live_counts[
            'pending_review'
        ]
        if pending_delta < 0:
            raise ReviewWorkflowServiceError('invalid_state_transition')
        increases = 0
        for status in ('approved', 'rejected', 'needs_more_evidence'):
            delta = live_counts[status] - adjusted_checkpoint[status]
            if delta < 0:
                raise ReviewWorkflowServiceError('invalid_state_transition')
            increases += delta
        if increases != pending_delta:
            raise ReviewWorkflowServiceError('invalid_state_transition')
        if pending_delta:
            self._verify_human_resolution_rows(thread_id=thread_id)

    def _verify_revoked_rows(self, *, thread_id: str) -> None:
        with self._session_factory() as db:
            rows = tuple(db.scalars(select(ReviewItem).where(
                ReviewItem.workflow_thread_id == thread_id,
                ReviewItem.status == 'revoked',
            )).all())
            valid = bool(rows) and all(
                row.resolution_source == 'auto_policy'
                and row.auto_validation_id is not None
                and row.revoked_at is not None
                for row in rows
            )
            db.rollback()
        if not valid:
            raise ReviewWorkflowServiceError('invalid_state_transition')

    def _verify_human_resolution_rows(self, *, thread_id: str) -> None:
        actions = {
            'approved': 'review.approve',
            'rejected': 'review.reject',
            'needs_more_evidence': 'review.request_more_evidence',
        }
        with self._session_factory() as db:
            rows = tuple(db.scalars(select(ReviewItem).where(
                ReviewItem.workflow_thread_id == thread_id,
                ReviewItem.resolution_source == 'human',
                ReviewItem.status.in_(tuple(actions)),
            )).all())
            for row in rows:
                audit = db.scalar(select(AuditLog.id).where(
                    AuditLog.target_type == 'review_item',
                    AuditLog.target_id == str(row.id),
                    AuditLog.action == actions[row.status],
                    AuditLog.status == 'success',
                    AuditLog.actor_id == row.reviewer_id,
                ))
                if row.reviewed_at is None or audit is None:
                    db.rollback()
                    raise ReviewWorkflowServiceError(
                        'invalid_state_transition'
                    )
            db.rollback()
        if not rows:
            raise ReviewWorkflowServiceError('invalid_state_transition')
