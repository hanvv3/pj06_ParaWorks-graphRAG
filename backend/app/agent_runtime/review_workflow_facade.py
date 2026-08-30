from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from math import ceil
from typing import Protocol

from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.auto_review_cost_policy import (
    AUTO_REVIEW_MAX_BATCH_COST_USD,
    AUTO_REVIEW_MAX_WORKFLOW_COST_USD,
    AUTO_REVIEW_REASONING_EFFORT,
    AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
    AUTO_REVIEW_VALIDATOR_MODEL,
    AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
    AUTO_REVIEW_VALIDATOR_PROVIDER,
    VALIDATION_COST_POLICY,
)
from backend.app.agent_runtime.canonical_sources import (
    ReviewWorkflowPreflightError,
    resolve_source_versions,
)
from backend.app.agent_runtime.review_v2_drafting import _build_exact_packet
from backend.app.agent_runtime.review_v2_preflight import V21PreparedReviewConfig
from backend.app.agent_runtime.review_v21_extraction import (
    ExtractionProviderSafetySnapshot,
    build_prepared_extraction_plan_set,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_versions import normalize_source_version_refs
from backend.app.models import (
    AgentWorkflowThread,
    AutoReviewProviderSafetyState,
    AutoReviewRuntimeKeyState,
)
from backend.app.review.auto_review_rollout import (
    AutoReviewRolloutPolicyService,
    resolve_effective_rollout,
)
from backend.app.schemas.auto_review import (
    AUTO_REVIEW_POLICY_VERSION,
    AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
    AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
)
from backend.app.schemas.review_workflow import COMPANY_MEMORY_REVIEW_GRAPH_VERSION


class ReviewLifecycle(Protocol):
    def diagnostic(self) -> object: ...

    def dry_run(self, *, actor: DemoUser, request: object) -> object: ...

    def start(self, *, actor: DemoUser, request: object) -> object: ...

    def status(self, *, actor: DemoUser, thread_id: str) -> object: ...

    def resume(self, *, actor: DemoUser, thread_id: str) -> object: ...

    def cancel(self, *, actor: DemoUser, thread_id: str) -> object: ...


class ReviewWorkflowFacadeError(RuntimeError):  # noqa: N818 - bounded error
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ReviewWorkflowFacade:
    """Selects new-run version by config and existing lifecycle by stored key."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        settings: Settings,
        v20: ReviewLifecycle,
        v21: ReviewLifecycle | None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._v20 = v20
        self._v21 = v21

    def diagnostic(self) -> object:
        return self._new_lifecycle().diagnostic()

    def dry_run(self, *, actor: DemoUser, request: object) -> object:
        return self._new_lifecycle().dry_run(actor=actor, request=request)

    def start(self, *, actor: DemoUser, request: object) -> object:
        return self._new_lifecycle().start(actor=actor, request=request)

    def status(self, *, actor: DemoUser, thread_id: str) -> object:
        return self._stored_lifecycle(actor=actor, thread_id=thread_id).status(
            actor=actor, thread_id=thread_id
        )

    def resume(self, *, actor: DemoUser, thread_id: str) -> object:
        return self._stored_lifecycle(actor=actor, thread_id=thread_id).resume(
            actor=actor, thread_id=thread_id
        )

    def cancel(self, *, actor: DemoUser, thread_id: str) -> object:
        return self._stored_lifecycle(actor=actor, thread_id=thread_id).cancel(
            actor=actor, thread_id=thread_id
        )

    def _new_lifecycle(self) -> ReviewLifecycle:
        if self._settings.auto_review_mode == 'disabled':
            return self._v20
        if self._v21 is None:
            raise ReviewWorkflowFacadeError('runtime_version_unavailable')
        return self._v21

    def _stored_lifecycle(
        self,
        *,
        actor: DemoUser,
        thread_id: str,
    ) -> ReviewLifecycle:
        with self._session_factory() as db:
            identity = db.execute(
                select(
                    AgentWorkflowThread.owner_subject_id,
                    AgentWorkflowThread.graph_version,
                ).where(AgentWorkflowThread.thread_id == thread_id)
            ).one_or_none()
            db.rollback()
        if identity is None or identity.owner_subject_id != actor.id:
            raise ReviewWorkflowFacadeError('not_found')
        if identity.graph_version == COMPANY_MEMORY_REVIEW_GRAPH_VERSION:
            return self._v20
        if (
            identity.graph_version == COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
            and self._v21 is not None
        ):
            return self._v21
        raise ReviewWorkflowFacadeError('runtime_version_unavailable')


class DatabaseV21LaunchAuthority:
    """Builds a read-only, registry-exact V2.1 launch configuration."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    def resolve_v21_config(
        self,
        *,
        db: Session,
        actor: DemoUser,
        request: object,
        operation: str,
    ) -> V21PreparedReviewConfig:
        del operation
        try:
            self._settings.require_auto_review_live_readiness()
            source_refs = normalize_source_version_refs(request.source_refs)
            agent_names = tuple(request.agent_names)
            resolved = resolve_source_versions(
                db,
                refs=source_refs,
                actor=actor,
                settings=self._settings,
            )
            packet = _build_exact_packet(
                db,
                refs=resolved,
                workflow_thread_id=None,
                actor_subject_id=actor.id,
                allowed_permission_levels=tuple(actor.permission_levels),
                settings=self._settings,
            )
            runtime = db.scalar(
                select(AutoReviewRuntimeKeyState).where(
                    AutoReviewRuntimeKeyState.component
                    == 'auto_review_trust_promotion'
                )
            )
            verifier = fingerprint_key_material_verifier(
                self._settings.agent_runtime_fingerprint_secret
            )
            if (
                runtime is None
                or not runtime.ready
                or runtime.fingerprint_key_version
                != self._settings.agent_runtime_fingerprint_key_version
                or runtime.fingerprint_key_material_verifier != verifier
            ):
                raise ValueError
            identities = (
                (
                    'extraction',
                    'openai',
                    'gpt-5.4-mini-2026-03-17',
                    'none',
                ),
                (
                    'validation',
                    AUTO_REVIEW_VALIDATOR_PROVIDER,
                    AUTO_REVIEW_VALIDATOR_MODEL,
                    AUTO_REVIEW_REASONING_EFFORT,
                ),
            )
            safety_rows = tuple(
                db.scalars(
                    select(AutoReviewProviderSafetyState)
                    .where(
                        tuple_(
                            AutoReviewProviderSafetyState.purpose,
                            AutoReviewProviderSafetyState.provider,
                            AutoReviewProviderSafetyState.model,
                            AutoReviewProviderSafetyState.reasoning_effort,
                        ).in_(identities)
                    )
                    .order_by(
                        AutoReviewProviderSafetyState.purpose,
                        AutoReviewProviderSafetyState.provider,
                        AutoReviewProviderSafetyState.model,
                        AutoReviewProviderSafetyState.reasoning_effort,
                    )
                ).all()
            )
            if len(safety_rows) != 2 or any(row.breaker_open for row in safety_rows):
                raise ValueError
            safety = {row.purpose: row for row in safety_rows}
            validation = safety['validation']
            expected_validation = (
                VALIDATION_COST_POLICY.cost_policy_version,
                VALIDATION_COST_POLICY.token_estimator_version,
                VALIDATION_COST_POLICY.tokenizer_encoding,
                VALIDATION_COST_POLICY.reply_priming_tokens,
                VALIDATION_COST_POLICY.framing_safety_tokens,
                AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
                AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
            )
            actual_validation = (
                validation.authorized_cost_policy_version,
                validation.token_estimator_version,
                validation.tokenizer_encoding,
                validation.reply_priming_tokens,
                validation.framing_safety_tokens,
                Decimal(validation.input_usd_per_1m),
                Decimal(validation.output_usd_per_1m),
            )
            if actual_validation != expected_validation:
                raise ValueError
            extraction = safety['extraction']
            plan_set = build_prepared_extraction_plan_set(
                packet=packet,
                selected_agent_names=agent_names,
                settings=self._settings,
                fingerprint_key_material_verifier=verifier,
                safety_snapshots=(
                    ExtractionProviderSafetySnapshot(
                        purpose=extraction.purpose,
                        provider=extraction.provider,
                        model=extraction.model,
                        reasoning_effort=extraction.reasoning_effort,
                        state_version=extraction.state_version,
                        cost_policy_version=(
                            extraction.authorized_cost_policy_version
                        ),
                        token_estimator_version=extraction.token_estimator_version,
                        tokenizer_encoding=extraction.tokenizer_encoding,
                        reply_priming_tokens=extraction.reply_priming_tokens,
                        framing_safety_tokens=extraction.framing_safety_tokens,
                        input_usd_per_1m=Decimal(extraction.input_usd_per_1m),
                        output_usd_per_1m=Decimal(extraction.output_usd_per_1m),
                        breaker_open=extraction.breaker_open,
                    ),
                ),
            )
            rollout = AutoReviewRolloutPolicyService(
                db, settings=self._settings
            ).peek_or_default(
                self._settings.agent_runtime_security_scope_id,
                AUTO_REVIEW_POLICY_VERSION,
            )
            effective = resolve_effective_rollout(
                rollout,
                configured_mode=self._settings.auto_review_mode,
                requested_percentage=self._settings.auto_review_enforce_percentage,
                global_demoted=False,
            )
            count = len(agent_names)
            validation_batches = ceil(count / 4)
            validation_ceiling = (
                Decimal(validation_batches) * AUTO_REVIEW_MAX_BATCH_COST_USD
            )
            total = plan_set.confirmed_cost_ceiling_usd + validation_ceiling
            if total > AUTO_REVIEW_MAX_WORKFLOW_COST_USD:
                raise ValueError
            first = plan_set.plans[0]
            return V21PreparedReviewConfig(
                configured_auto_review_mode=effective.mode,
                validator_provider=validation.provider,
                validator_model=validation.model,
                validator_reasoning_effort=validation.reasoning_effort,
                validator_prompt_version=AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
                validator_output_contract_version=(
                    AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION
                ),
                policy_version=AUTO_REVIEW_POLICY_VERSION,
                cost_policy_version=validation.authorized_cost_policy_version,
                fingerprint_key_version=runtime.fingerprint_key_version,
                fingerprint_key_material_verifier=verifier,
                token_estimator_version=validation.token_estimator_version,
                tokenizer_encoding=validation.tokenizer_encoding,
                max_input_tokens_per_batch=VALIDATION_COST_POLICY.max_input_tokens,
                max_output_tokens_per_batch=VALIDATION_COST_POLICY.max_output_tokens,
                reply_priming_tokens=validation.reply_priming_tokens,
                framing_safety_tokens=validation.framing_safety_tokens,
                max_validation_batches_per_workflow=(
                    VALIDATION_COST_POLICY.max_batches_per_workflow
                ),
                max_validation_candidates_per_batch=(
                    VALIDATION_COST_POLICY.max_candidates_per_batch
                ),
                max_validation_candidates_per_workflow=5,
                max_provider_attempts=VALIDATION_COST_POLICY.max_provider_attempts,
                provider_timeout_seconds=first.provider_timeout_seconds,
                provider_send_start_window_seconds=(
                    first.provider_send_start_window_seconds
                ),
                provider_attempt_lease_seconds=(
                    first.provider_attempt_lease_seconds
                ),
                provider_commit_grace_seconds=(
                    first.provider_commit_grace_seconds
                ),
                validator_input_usd_per_1m=Decimal(
                    validation.input_usd_per_1m
                ),
                validator_output_usd_per_1m=Decimal(
                    validation.output_usd_per_1m
                ),
                enforce_percentage=effective.percentage,
                authorized_percentage_at_launch=(
                    rollout.max_authorized_percentage
                ),
                rollout_authorization_generation=(
                    rollout.authorization_generation
                ),
                validation_provider_safety_state_version=validation.state_version,
                rollout_control_epoch=rollout.control_epoch,
                extraction_plan_set_hmac=plan_set.plan_set_hmac,
                extraction_provider_safety_snapshot_set_hmac=(
                    plan_set.provider_safety_snapshot_set_hmac
                ),
                confirmed_extraction_cost_ceiling_usd=(
                    plan_set.confirmed_cost_ceiling_usd
                ),
                confirmed_validation_cost_ceiling_usd=validation_ceiling,
                confirmed_total_cost_ceiling_usd=total,
                total_budget_limit_usd=AUTO_REVIEW_MAX_WORKFLOW_COST_USD,
                extraction_plan_identities=tuple(
                    plan.identity_payload() for plan in plan_set.plans
                ),
            )
        except ReviewWorkflowPreflightError:
            raise
        except Exception:
            raise ReviewWorkflowPreflightError(
                'cost_preview_changed',
                'V2.1 launch authority is unavailable',
            ) from None
