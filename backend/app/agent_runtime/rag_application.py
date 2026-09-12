from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Generic, Literal, TypeVar, get_args

from langsmith import tracing_context
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.provider_usage import RagResultOutcome
from backend.app.agent_runtime.rag_finalization import (
    AssistantProjectionTarget,
    _empty_answer_projection,
)
from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy
from backend.app.agent_runtime.rag_v2_contracts import (
    COMPANY_MEMORY_RAG_GRAPH_VERSION,
    COMPANY_MEMORY_RAG_WORKFLOW,
    RagSurface,
    resolved_rag_mode,
    resolved_rag_stage,
)
from backend.app.agent_runtime.rag_v2_identity import StrictUnicodeScalarValidator
from backend.app.agent_runtime.rag_v2_registry import RagGraphRegistry
from backend.app.agent_runtime.rag_v2_state import (
    AssistantExecutionDisposition,
    RagGraphOutput,
    RagRequestServices,
    RagRuntimeContext,
    RagRunTrace,
    SafeRagToolRecorder,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    PreparedRagRequestText,
    _prepared,
    _scan_value,
    prepare_direct_request_text,
)
from backend.app.assistant.delivery import AssistantDeliveryResult
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.rag.lexical_projection import tokenize_rag_lexical_query
from backend.app.schemas.rag import (
    AskV1Projection,
    ExactV1Projection,
    PublicRagErrorCode,
    SearchV1Projection,
)

T = TypeVar('T', bound=ExactV1Projection)
DirectRagApplicationOutcome = (
    RagResultOutcome
    | Literal[
        'permission_denied',
        'invalid_input',
        'input_safety_blocked',
        'input_scanner_unavailable',
        'persistence_failed',
    ]
)


@dataclass(frozen=True, slots=True)
class RagPublicErrorProjection:
    code: PublicRagErrorCode

    def __post_init__(self):
        if self.code not in get_args(PublicRagErrorCode):
            raise ValueError('unknown public RAG error code')


@dataclass(frozen=True, slots=True)
class DirectRagDeliveryResult(Generic[T]):
    public_status: Literal[200, 403, 409, 422, 500, 502, 503]
    body_kind: Literal[
        'success',
        'permission_error',
        'validation_error',
        'budget_error',
        'runtime_error',
        'generation_error',
        'persistence_error',
    ]
    application_outcome: DirectRagApplicationOutcome
    projection: T | None
    error: RagPublicErrorProjection | None

    def __post_init__(self):
        if self.public_status == 200:
            if (
                self.body_kind != 'success'
                or self.projection is None
                or self.error is not None
            ):
                raise ValueError('invalid success delivery')
        elif (
            self.projection is not None
            or self.error is None
            or self.body_kind == 'success'
        ):
            raise ValueError('invalid error delivery')


def direct_rag_error(code, *, component=None):
    if code not in get_args(PublicRagErrorCode):
        code = 'unexpected_internal_error'
    status, kind = {
        'permission_denied': (403, 'permission_error'),
        'input_safety_blocked': (422, 'validation_error'),
        'budget_exceeded': (409, 'budget_error'),
        'persistence_failed': (500, 'persistence_error'),
        'unexpected_internal_error': (500, 'runtime_error'),
        'model_provider_failed': (502, 'generation_error'),
        'structured_output_invalid': (502, 'generation_error'),
        'citation_validation_failed': (502, 'generation_error'),
    }.get(code, (503, 'runtime_error'))
    if code in {'provider_response_identity_invalid', 'provider_usage_overrun'}:
        if component not in {'query_embedding', 'answer_generation'}:
            return direct_rag_error('unexpected_internal_error')
        if component == 'answer_generation':
            status, kind = 502, 'generation_error'
    return DirectRagDeliveryResult(
        status, kind, code, None, RagPublicErrorProjection(code)
    )


class RagApplicationError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PreparedAssistantIngress:
    conversation_id: int | None
    owner_user_id: str
    prepared_text: PreparedRagRequestText
    prior_context_hmac: str
    capability_version: Literal['rag-v2-plain-text-citations:v1']


class AssistantIngressError(RuntimeError):
    def __init__(self, result: AssistantDeliveryResult):
        self.result = result
        super().__init__(result.application_outcome)


def _assistant_ingress_error(outcome):
    status, kind = {
        'invalid_input': (422, 'validation_error'),
        'input_safety_blocked': (422, 'validation_error'),
        'owner_not_found': (404, 'owner_not_found'),
        'permission_denied': (403, 'permission_error'),
        'runtime_version_unavailable': (502, 'generation_error'),
        'input_scanner_unavailable': (500, 'persistence_error'),
    }[outcome]
    return AssistantIngressError(AssistantDeliveryResult(
        outcome, None, kind, 'not_persisted', None, status,
    ))


def _assistant_unknown():
    return AssistantDeliveryResult(
        'commit_unknown', None, 'reconciliation_required', 'commit_unknown', None, 500,
    )


def validate_prepared_ingress(
    text: PreparedRagRequestText, *, surface: RagSurface, settings: Settings
) -> bool:
    """Authenticate server preparation before opening any request dependency."""
    if type(text) is not PreparedRagRequestText:
        raise ValueError('prepared request text is required')
    for value in (
        text.caller_text,
        text.normalized_current_user_text,
        text.retrieval_query_text,
        text.answer_question_text,
    ):
        StrictUnicodeScalarValidator.validate(value)
    key = fingerprint_secret_bytes(settings)[0]
    direct = prepare_direct_request_text(text.caller_text, key=key)
    if surface in {'ask', 'search'}:
        if direct != text:
            raise ValueError('direct request preparation changed')
        return True
    if surface != 'assistant' or text.query_context_version != 'assistant-context:v1':
        raise ValueError('request context does not match surface')
    if (
        len(text.caller_text) > 4000
        or len(text.retrieval_query_text) > 8000
        or not text.caller_text.strip()
    ):
        raise ValueError('assistant request bounds exceeded')
    expected = _prepared(
        caller_text=text.caller_text,
        normalized_current_user_text=direct.normalized_current_user_text,
        retrieval_query_text=text.retrieval_query_text,
        answer_question_text=text.caller_text.strip(),
        query_context_version='assistant-context:v1',
        key=key,
    )
    if expected != text:
        raise ValueError('assistant request preparation changed')
    _scan_value(text.retrieval_query_text)
    return len(tokenize_rag_lexical_query(text.retrieval_query_text)) <= 1000


def ingress_budget_refusal(settings: Settings) -> RagGraphOutput:
    return {
        'outcome': 'budget_exceeded',
        'answer_blocks': None,
        'selected_slot_ids': (),
        'evidence_projection': _empty_answer_projection(settings),
        'model_influence': (),
        'hidden_match_count': 0,
        'effective_backend': 'deterministic_lexical',
        'fallback_category': None,
        'charged_cost_usd': Decimal('0.000000'),
        'sanitized_trace': RagRunTrace(
            {'validate_input': 1},
            {'validate_input': 0},
            {'query_embedding': 0, 'answer_generation': 0},
            {},
            {},
        ),
    }


@dataclass(frozen=True, slots=True)
class RagApplicationFacade:
    _settings: Settings = field(repr=False)
    _session_factory: Callable[[], Session] = field(repr=False)
    _request_factory: Callable[..., AbstractContextManager[RagRequestServices]] = field(
        repr=False
    )
    _graph_registry: RagGraphRegistry = field(repr=False)
    _rollout: RagRolloutPolicy = field(default_factory=RagRolloutPolicy)

    @staticmethod
    def invoke_legacy_assistant_context(**kwargs):
        """Compatibility boundary for legacy answers and email RAG context."""
        from backend.app.agents.rag_orchestrator_agent import answer_question_with_rag
        return answer_question_with_rag(**kwargs)

    def prepare_assistant_ingress(
        self, *, actor: DemoUser, conversation_id: int | None, caller_text: str,
    ) -> PreparedAssistantIngress:
        return self._prepare_assistant_ingress(actor=actor,
            conversation_id=conversation_id, caller_text=caller_text)

    def _prepare_assistant_ingress(
        self, *, actor, conversation_id, caller_text, current_message_id=None,
    ):
        from sqlalchemy import select

        from backend.app.agent_runtime.fingerprints import keyed_fingerprint
        from backend.app.agent_runtime.rag_v2_identity import (
            ServerRagSecurityScopeResolver,
        )
        from backend.app.agents.rag_orchestrator_agent.v2_input import (
            AssistantContextMessage,
            RagInputSafetyError,
            RagInputScannerUnavailableError,
            prepare_assistant_request_text,
        )
        from backend.app.assistant.evidence_reader import AssistantEvidenceReader
        from backend.app.models import AssistantConversation, AssistantMessage

        try:
            StrictUnicodeScalarValidator.validate(caller_text)
            if not caller_text.strip() or len(caller_text) > 4000:
                raise ValueError('invalid input')
        except (TypeError, ValueError):
            raise _assistant_ingress_error('invalid_input') from None
        key = fingerprint_secret_bytes(self._settings)[0]
        with self._session_factory() as db, db.no_autoflush:
            if conversation_id is not None:
                owner = db.scalar(select(AssistantConversation.user_id).where(
                    AssistantConversation.id == conversation_id,
                ))
                if owner != actor.id:
                    raise _assistant_ingress_error('owner_not_found')
            try:
                prepare_direct_request_text(caller_text, key=key)
            except (RagInputSafetyError, RagInputScannerUnavailableError) as exc:
                raise _assistant_ingress_error(exc.code) from None
            scope = ServerRagSecurityScopeResolver(self._settings).resolve(db=db, actor=actor)
            if not scope.allowed_permission_levels:
                raise _assistant_ingress_error('permission_denied')
            rows = list(db.scalars(select(AssistantMessage).where(
                AssistantMessage.conversation_id == conversation_id,
            ).order_by(AssistantMessage.created_at, AssistantMessage.id))) if conversation_id else []
            if current_message_id is not None:
                current = next((row for row in rows if row.id == current_message_id), None)
                if (current is None or current.role != 'user'
                    or current.content != caller_text.strip()):
                    raise _assistant_ingress_error('runtime_version_unavailable')
                rows = [row for row in rows if row.id != current_message_id]
            reader = AssistantEvidenceReader(settings=self._settings)
            views = [reader.project_message(db=db, actor=actor, message=row) for row in rows]
            if any(not view.evidence_available for view in views):
                raise _assistant_ingress_error('runtime_version_unavailable')
            prior = [AssistantContextMessage(view.id, row.created_at, view.role, view.content)
                     for row, view in zip(rows, views, strict=True)]
            try:
                prepared = prepare_assistant_request_text(caller_text, prior, key=key)
            except RagInputScannerUnavailableError:
                raise _assistant_ingress_error('input_scanner_unavailable') from None
            except Exception:
                raise _assistant_ingress_error('runtime_version_unavailable') from None
            prior_hmac = keyed_fingerprint(
                {'owner': actor.id, 'conversation_id': conversation_id,
                 'retrieval_query_hmac': prepared.retrieval_query_hmac,
                 'prior': [{'id': row.message_id, 'created_at': row.created_at.isoformat(),
                            'role': row.role, 'content': row.content} for row in prior]},
                secret=key, schema_version='assistant-prepared-ingress:v1',
                policy_version='assistant-context:v1',
            )
        return PreparedAssistantIngress(conversation_id, actor.id, prepared, prior_hmac,
                                        'rag-v2-plain-text-citations:v1')

    def invoke_assistant(
        self, *, actor: DemoUser, conversation_id: int, user_message_id: int,
        prepared_ingress: PreparedAssistantIngress,
        tool_recorder: SafeRagToolRecorder | None = None,
    ) -> AssistantDeliveryResult:
        from backend.app.agent_runtime.rag_finalization import (
            AssistantFinalizationRecord,
            AssistantPreDispatchFailureFinalizer,
            RagFinalizationError,
            assistant_failed_parent_delivery,
            finalize_assistant_failed_parent,
        )
        from backend.app.assistant.delivery import _SAFE_GENERATION_FAILURE_OUTCOMES
        if (type(prepared_ingress) is not PreparedAssistantIngress
            or prepared_ingress.owner_user_id != actor.id
            or prepared_ingress.conversation_id not in {None, conversation_id}
            or prepared_ingress.capability_version != 'rag-v2-plain-text-citations:v1'):
            return _assistant_unknown()
        target = AssistantProjectionTarget(conversation_id, user_message_id, actor.id)
        try:
            expected = self._prepare_assistant_ingress(actor=actor,
                conversation_id=conversation_id, caller_text=prepared_ingress.prepared_text.caller_text,
                current_message_id=user_message_id)
            if expected != prepared_ingress:
                return _assistant_unknown()
        except Exception:
            return _assistant_unknown()
        execution = AssistantExecutionDisposition()
        try:
            result = self.invoke_graph(actor=actor, surface='assistant',
                prepared_text=prepared_ingress.prepared_text, assistant_target=target,
                assistant_execution=execution)
            record = result.get('assistant_finalization')
            if type(record) is AssistantFinalizationRecord:
                return _assistant_committed(record)
            if (result.get('outcome') in _SAFE_GENERATION_FAILURE_OUTCOMES | {'budget_exceeded'}
                and type(result.get('run_id')) is int):
                return finalize_assistant_failed_parent(
                    session_factory=self._session_factory, settings=self._settings,
                    actor=actor,
                    target=target, prepared_text=prepared_ingress.prepared_text,
                    parent_id=result['run_id'], outcome=result['outcome'],
                    effective_backend=result.get('effective_backend'),
                    fallback_category=result.get('fallback_category'))
            # A preflight budget refusal opens neither request services nor a claim.
            if (result.get('outcome') == 'budget_exceeded'
                and execution.phase == 'preflight' and not result.get('run_id')):
                return AssistantPreDispatchFailureFinalizer(
                    settings=self._settings, session_factory=self._session_factory, actor=actor,
                ).finalize(target=target, prepared_text=prepared_ingress.prepared_text,
                           outcome='budget_exceeded')
            return _assistant_unknown()
        except RagApplicationError as exc:
            # Only the same exception positively recorded at a request-owned
            # preclaim boundary may authorize a new parent. Cleanup replacements
            # and failures after admission/result publication are unknown.
            if execution.preclaim_failure is not exc:
                return _assistant_unknown()
            if exc.code in {'runtime_version_unavailable', 'retriever_not_configured',
                            'retriever_unavailable', 'model_unavailable',
                            'provider_safety_unavailable', 'budget_exceeded'}:
                return AssistantPreDispatchFailureFinalizer(
                    settings=self._settings, session_factory=self._session_factory, actor=actor,
                ).finalize(target=target, prepared_text=prepared_ingress.prepared_text,
                           outcome=exc.code)
            return _assistant_unknown()
        except Exception as exc:
            if isinstance(exc, RagFinalizationError):
                return assistant_failed_parent_delivery(
                    session_factory=self._session_factory,
                    parent_id=getattr(exc, 'run_id', None), target=target,
                )
            from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
            from backend.app.agent_runtime.rag_v2_registry import (
                RagRuntimeVersionUnavailableError,
            )
            from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
                QueryEmbeddingReadinessError,
            )
            outcome = ('runtime_version_unavailable' if isinstance(exc, RagRuntimeVersionUnavailableError)
                       else 'budget_exceeded' if isinstance(exc, RagBudgetExceededError)
                       else 'retriever_unavailable' if isinstance(exc, QueryEmbeddingReadinessError)
                       else None)
            if outcome is not None and execution.preclaim_failure is exc:
                return AssistantPreDispatchFailureFinalizer(settings=self._settings,
                    session_factory=self._session_factory, actor=actor).finalize(
                        target=target, prepared_text=prepared_ingress.prepared_text, outcome=outcome)
            return _assistant_unknown()

    def invoke_ask(
        self, *, actor: DemoUser, caller_text: str
    ) -> DirectRagDeliveryResult[AskV1Projection]:
        return self._invoke_direct(actor=actor, caller_text=caller_text, surface='ask')

    def invoke_search(
        self, *, actor: DemoUser, caller_text: str
    ) -> DirectRagDeliveryResult[SearchV1Projection]:
        return self._invoke_direct(
            actor=actor, caller_text=caller_text, surface='search'
        )

    def _invoke_direct(self, *, actor, caller_text, surface):
        from backend.app.agents.rag_orchestrator_agent.v2_input import (
            RagInputSafetyError,
            RagInputScannerUnavailableError,
        )

        # No session, retriever, or provider exists before the ingress scan.
        try:
            prepared = prepare_direct_request_text(
                caller_text,
                key=fingerprint_secret_bytes(self._settings)[0],
            )
        except (RagInputSafetyError, RagInputScannerUnavailableError) as exc:
            return direct_rag_error(exc.code)
        if self.execution_owner(surface) != 'v2':
            # Task21 owns shadow comparison. Public ownership remains legacy.
            return self._invoke_legacy(
                actor=actor, caller_text=caller_text, surface=surface
            )
        from backend.app.agent_runtime.rag_cost_ledger import RagCostPersistenceError
        from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
        from backend.app.agent_runtime.rag_finalization import RagFinalizationError
        from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
        from backend.app.agent_runtime.rag_provider_transport import (
            RagProviderSafetyRefusalError,
        )
        from backend.app.agent_runtime.rag_v2_registry import (
            RagRuntimeVersionUnavailableError,
        )
        from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
            QueryEmbeddingReadinessError,
        )

        try:
            result = self.invoke_graph(
                actor=actor, surface=surface, prepared_text=prepared
            )
            if result['outcome'] in get_args(PublicRagErrorCode):
                return direct_rag_error(
                    result['outcome'], component=result.get('error_component')
                )
            projection = _project_graph_result(
                result, surface=surface, caller_text=caller_text
            )
            return DirectRagDeliveryResult(
                200, 'success', result['outcome'], projection, None
            )
        except (RagApplicationError, RagRuntimeVersionUnavailableError) as exc:
            return direct_rag_error(exc.code)
        except (RagInputSafetyError, RagInputScannerUnavailableError) as exc:
            return direct_rag_error(exc.code)
        except RagBudgetExceededError:
            return direct_rag_error('budget_exceeded')
        except QueryEmbeddingReadinessError:
            return direct_rag_error('retriever_unavailable')
        except RagProviderSafetyRefusalError as exc:
            return direct_rag_error(exc.terminal.outcome)
        except RagProviderSafetyError:
            return direct_rag_error('provider_safety_unavailable')
        except (RagFinalizationError, RagCostPersistenceError):
            return direct_rag_error('persistence_failed')
        except Exception:
            return direct_rag_error('unexpected_internal_error')

    def _invoke_legacy(self, *, actor, caller_text, surface):
        from backend.app.agents.rag_orchestrator_agent import service
        from backend.app.permissions.service import can_access_permission

        with self._session_factory() as db:
            vector_store = _legacy_pgvector_search_store(db=db, settings=self._settings)
            if surface == 'ask':
                answer = service.answer_question_with_rag(
                    db=db,
                    user=actor,
                    question=caller_text,
                    settings=self._settings,
                    vector_store=vector_store,
                )
                projection = AskV1Projection(
                    **{
                        name: getattr(answer, name)
                        for name in (
                            'agent_name',
                            'prompt_version',
                            'question',
                            'answer',
                            'source_ids',
                            'source_links',
                            'source_snippets',
                            'citations',
                            'permission_level',
                            'hidden_match_count',
                            'permission_notice',
                            'agent_run_id',
                            'cache_key',
                        )
                    },
                    model_name=answer.cost.model_name,
                    estimated_cost_usd=answer.cost.estimated_cost_usd,
                    token_usage={
                        'input_tokens': answer.cost.token_usage.input_tokens,
                        'output_tokens': answer.cost.token_usage.output_tokens,
                        'total_tokens': answer.cost.token_usage.total_tokens,
                    },
                )
                outcome = (
                    'evidence_unavailable'
                    if answer.permission_notice == 'evidence_unavailable'
                    else 'supported'
                    if answer.source_ids
                    else 'hidden_only'
                    if answer.hidden_match_count
                    else 'no_match'
                )
            else:
                if vector_store is None:
                    candidates = service.retrieve_matching_evidence_candidates(
                        db=db, question=caller_text
                    )
                    candidates = service.filter_live_serving_candidates(
                        db=db, candidates=candidates
                    )
                    visible = [
                        c
                        for c in candidates
                        if can_access_permission(actor, c.permission_level)
                    ]
                    hidden = len(candidates) - len(visible)
                else:
                    result = vector_store.search(query=caller_text, user=actor, limit=5)
                    visible = service.filter_live_serving_candidates(
                        db=db,
                        candidates=service.candidates_from_vector_matches(
                            result.matches
                        ),
                    )
                    hidden = result.hidden_match_count
                projection = SearchV1Projection(
                    retrieval_backend='pgvector'
                    if vector_store is not None
                    else 'deterministic_lexical',
                    cost_policy={
                        'embedding_query_call': vector_store is not None,
                        'paid_llm_call': False,
                        'requires_pgvector_flag': True,
                    },
                    hidden_match_count=hidden,
                    permission_notice=_permission_notice(hidden),
                    results=[
                        {
                            'id': int(c.metadata.get('chunk_id') or index + 1),
                            'source_id': c.source_id,
                            'text': c.text,
                            'source_snippet': c.source_snippet,
                            'source_url': c.source_url,
                            'source_type': c.metadata.get('source_type'),
                            'permission_level': c.permission_level,
                            'relevance_score': c.relevance_score,
                            'matched_terms': c.matched_terms,
                            'citation': service.citation_from_candidate(c),
                            'parser_status': c.metadata.get('parser_status'),
                            'parser_status_reason': c.metadata.get(
                                'parser_status_reason'
                            ),
                            'revision_id': c.metadata.get('revision_id'),
                        }
                        for index, c in enumerate(visible)
                    ],
                )
                outcome = 'search_projected'
            return DirectRagDeliveryResult(200, 'success', outcome, projection, None)

    def execution_owner(self, surface: RagSurface) -> Literal['legacy', 'shadow', 'v2']:
        decision = self._rollout.decide(
            mode=resolved_rag_mode(self._settings),
            stage=resolved_rag_stage(self._settings),
            surface=surface,
        )
        return 'shadow' if decision.shadow_retrieval else decision.public_owner

    def invoke_graph(
        self,
        *,
        actor: DemoUser,
        surface: RagSurface,
        prepared_text: PreparedRagRequestText,
        assistant_target: AssistantProjectionTarget | None = None,
        assistant_execution: AssistantExecutionDisposition | None = None,
    ) -> RagGraphOutput:
        if assistant_execution is not None:
            assistant_execution.phase = 'preflight'
        if self.execution_owner(surface) != 'v2':
            # The later shadow bridge owns comparison; this never invokes the
            # product graph or silently substitutes for the legacy service.
            raise RagApplicationError('runtime_version_unavailable')
        settings = self._settings.model_copy(deep=True)
        if not validate_prepared_ingress(
            prepared_text, surface=surface, settings=settings
        ):
            return ingress_budget_refusal(settings)
        if surface == 'assistant' and assistant_target is None:
            raise ValueError('assistant persistence target is required')
        if assistant_target is not None and (
            surface != 'assistant' or assistant_target.owner_user_id != actor.id
        ):
            raise ValueError('assistant target does not match request owner')
        try:
            graph = self._graph_registry.resolve(
                COMPANY_MEMORY_RAG_WORKFLOW, COMPANY_MEMORY_RAG_GRAPH_VERSION
            )
        except Exception as exc:
            if assistant_execution is not None:
                assistant_execution.record_preclaim_failure(exc)
            raise
        request_actor = replace(
            actor, permission_levels=frozenset(actor.permission_levels)
        )
        if assistant_execution is not None:
            assistant_execution.phase = 'factory_entry'
        with (
            tracing_context(enabled=False),
            self._request_factory(
                session_factory=self._session_factory,
                settings=settings,
                actor=request_actor,
                surface=surface,
                assistant_target=assistant_target,
                **({'assistant_execution': assistant_execution} if assistant_execution is not None else {}),
            ) as services,
        ):
            if type(services) is not RagRequestServices:
                raise RagApplicationError('runtime_version_unavailable')
            if assistant_execution is not None:
                assistant_execution.phase = 'graph_preclaim'
            context = RagRuntimeContext(
                actor=request_actor,
                surface=surface,
                settings=settings,
                services=services,
                assistant_target=assistant_target,
                assistant_execution=assistant_execution,
            )
            try:
                result = graph.invoke(
                    {'prepared_text': prepared_text},
                    context=context,
                    config={'callbacks': [], 'metadata': {}, 'recursion_limit': 40},
                )
            except Exception as exc:
                if assistant_execution is not None:
                    assistant_execution.record_preclaim_failure(exc)
                    assistant_execution.phase = 'graph_failed'
                raise
            if assistant_execution is not None:
                assistant_execution.phase = 'graph_returned'
        if assistant_execution is not None:
            assistant_execution.phase = 'request_exited'
        return result


def _legacy_pgvector_search_store(*, db, settings):
    from backend.app.rag.search_store import build_pgvector_search_store

    return build_pgvector_search_store(db=db, settings=settings)


def _assistant_committed(record):
    from backend.app.assistant.delivery import _SUCCESS_OUTCOMES
    success = record.application_outcome in _SUCCESS_OUTCOMES
    budget = record.application_outcome == 'budget_exceeded'
    return AssistantDeliveryResult(
        record.application_outcome, record.assistant_message_id,
        'assistant_message' if success else 'budget_error' if budget else 'generation_error',
        'committed_success' if success else 'committed_safe_failure',
        record.parent_agent_run_id, 200 if success else 409 if budget else 502,
    )


def _permission_notice(hidden):
    return 'Some sources may be hidden by permissions.' if hidden else None


def _project_graph_result(result, *, surface, caller_text):
    """Copy only the committed carrier. Never resolve evidence or read cost rows."""
    from backend.app.agent_runtime.rag_finalization import (
        CanonicalRagProjection,
        RagFinalizationError,
        _output_permission,
    )

    canonical = result.get('committed_projection')
    if (
        type(canonical) is not CanonicalRagProjection
        or type(result.get('run_id')) is not int
        or result['run_id'] <= 0
    ):
        raise RagFinalizationError('committed direct projection is required')
    if surface == 'search':
        return SearchV1Projection(
            retrieval_backend=canonical.effective_backend,
            cost_policy={
                'embedding_query_call': bool(
                    result['sanitized_trace'].provider_attempt_counts['query_embedding']
                ),
                'paid_llm_call': False,
                'requires_pgvector_flag': True,
            },
            hidden_match_count=canonical.hidden_match_count,
            permission_notice=_permission_notice(canonical.hidden_match_count),
            results=[dict(row) for row in canonical.evidence.search_results],
        )
    component = result.get('generation_component')
    tokens = (0, 0)
    model = 'gpt-5.4-mini-2026-03-17'
    if component is not None:
        if (
            component.component != 'answer_generation'
            or component.terminal_outcome != 'component_succeeded'
        ):
            raise RagFinalizationError('validated generation usage is required')
        model = component.model
        tokens = (component.actual_input_tokens, component.actual_output_tokens)
        if any(type(n) is not int or n < 0 for n in tokens):
            raise RagFinalizationError('validated generation usage is required')
    elif result.get('deterministic_answer_generated'):
        model = 'fake-rag-v2-model'
    citations = [dict(c) for c in canonical.evidence.citations]
    return AskV1Projection(
        agent_name='rag_orchestrator_agent',
        prompt_version='rag-answer:v2',
        question=caller_text,
        answer=canonical.answer_text,
        source_ids=canonical.evidence.source_ids,
        source_links=canonical.evidence.source_links,
        source_snippets=canonical.evidence.source_snippets,
        citations=citations,
        permission_level=_output_permission(canonical),
        hidden_match_count=canonical.hidden_match_count,
        permission_notice='evidence_unavailable'
        if canonical.outcome == 'evidence_unavailable'
        else _permission_notice(canonical.hidden_match_count),
        agent_run_id=result['run_id'],
        cache_key=canonical.result_hmac,
        model_name=model,
        estimated_cost_usd=float(result['charged_cost_usd']),
        token_usage={
            'input_tokens': tokens[0],
            'output_tokens': tokens[1],
            'total_tokens': sum(tokens),
        },
    )
