from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from functools import wraps
from time import monotonic_ns

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from backend.app.agent_runtime.rag_cost_ledger import RagPreclaimSafetyRefusalError
from backend.app.agent_runtime.rag_finalization import (
    AssistantFinalizationRecord,
    CanonicalRagProjection,
    PreparedRagFinalization,
)
from backend.app.agent_runtime.rag_v2_contracts import (
    resolved_rag_backend,
    resolved_rag_mode,
    resolved_rag_stage,
)
from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.agent_runtime.rag_v2_state import (
    RagGraphInput,
    RagGraphOutput,
    RagGraphState,
    RagRuntimeContext,
    RagRunTrace,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ValidatedAnswerBlocks,
)
from backend.app.rag.evidence_projection import (
    CanonicalEvidenceProjector,
    _identity_from_evidence,
)
from backend.app.rag.pgvector_retriever import (
    PgVectorEvidenceRetriever,
    PgVectorFallbackRequiredError,
)
from backend.app.rag.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    rank_evidence_slots,
)


def _node(function):
    @wraps(function)
    def measured(state: RagGraphState, runtime: Runtime[RagRuntimeContext]):
        started = monotonic_ns()
        try:
            update = function(state, runtime)
        except Exception as exc:
            # Preserve evidence of an existing claim across the facade boundary.
            # Losing it must never mint a second provider-free parent.
            if state.get('run_id') is not None:
                exc.run_id = state['run_id']
            raise
        counts = {**state.get('node_counts', {}), function.__name__: 1}
        latencies = {
            **state.get('node_latency_ms', {}),
            function.__name__: max(0, (monotonic_ns() - started) // 1_000_000),
        }
        update.update(node_counts=counts, node_latency_ms=latencies)
        if 'outcome' in update:
            retrieval = state.get('retrieval')
            merged = {**state, **update}
            update['sanitized_trace'] = RagRunTrace(
                counts,
                latencies,
                {
                    'query_embedding': int(
                        merged.get('query_embedding_attempted', False)
                    ),
                    'answer_generation': int(
                        merged.get('answer_generation_attempted', False)
                    ),
                },
                {
                    'candidate_window': retrieval.trace.candidate_window_count
                    if retrieval
                    else 0,
                    'visible': len(retrieval.visible) if retrieval else 0,
                    'hidden': update['hidden_match_count'],
                },
                {
                    'current_text': state['prepared_text'].current_text_hmac,
                    'retrieval_query': state['prepared_text'].retrieval_query_hmac,
                    **(
                        {'scope': state['scope_fingerprint']}
                        if 'scope_fingerprint' in state
                        else {}
                    ),
                },
            )
        return update

    return measured


def _request(state, *, embedding=None):
    return RetrievalRequest(
        retrieval_query_text=state['prepared_text'].retrieval_query_text,
        security_scope=state['security_scope'],
        security_scope_fingerprint=state['scope_fingerprint'],
        query_embedding_result=embedding,
        candidate_scan_limit=50,
        visible_limit=state.get('visible_limit', 5),
        relevance_policy_version='rag-retrieval-policy:v2.0',
    )


def _transport(services):
    if services.provider_transport is not None:
        return services.provider_transport
    if services.provider_transport_factory is None:
        raise RuntimeError('provider transport is unavailable')
    return services.provider_transport_factory()


@_node
def validate_input(state: RagGraphState, runtime: Runtime[RagRuntimeContext]):
    from backend.app.agent_runtime.rag_application import (
        ingress_budget_refusal,
        validate_prepared_ingress,
    )

    if not validate_prepared_ingress(
        state['prepared_text'],
        surface=runtime.context.surface,
        settings=runtime.context.settings,
    ):
        return ingress_budget_refusal(runtime.context.settings)
    return {}


@_node
def resolve_current_permission_context(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    context = runtime.context
    scope = context.services.security_scope_resolver.resolve(
        db=context.services.db, actor=context.actor
    )
    if not scope.allowed_permission_levels:
        from backend.app.agent_runtime.rag_application import RagApplicationError
        raise RagApplicationError('permission_denied')
    return {
        'security_scope': scope,
        'scope_fingerprint': security_scope_fingerprint(
            scope, settings=context.settings
        ),
    }


@_node
def select_configured_backend(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    return {
        'configured_backend': resolved_rag_backend(runtime.context.settings),
        'visible_limit': 5 if runtime.context.surface == 'search' else 8,
    }


@_node
def preflight_retrieval_paid_cost_ceiling(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    context = runtime.context
    services = context.services
    backend = state['configured_backend']
    if services.sqlite_scope is not None:
        if context.assistant_execution is not None:
            context.assistant_execution.begin_admission()
        run_id, generations = services.sqlite_scope.admit(
            prepared_text=state['prepared_text'],
            surface=context.surface,
            security_scope=state['security_scope'],
        )
        return {'run_id': run_id, 'generations': generations}
    prepared_embedding = None
    if backend == 'pgvector':
        if services.query_embedding_adapter is None or services.index_readiness is None:
            from backend.app.agent_runtime.rag_application import RagApplicationError

            raise RagApplicationError('retriever_not_configured')
        prepared_embedding = services.query_embedding_adapter.prepare(
            _request(state),
            services.index_readiness.inspect(db=services.db),
        )
    components = tuple(
        (
            snapshot,
            services.cost_policy.reserve_answer_generation()
            if snapshot.component == 'answer_generation' and context.surface != 'search'
            else prepared_embedding.budget
            if snapshot.component == 'query_embedding' and prepared_embedding is not None
            else services.cost_policy.reserve_unused_component(snapshot.component),
        )
        for snapshot in services.policy_snapshots
    )
    if context.assistant_execution is not None:
        context.assistant_execution.begin_admission()
    run_id = services.allocate_run_id()
    services.cost_ledger.create_admission(
        agent_run_id=run_id,
        surface=context.surface,
        mode=resolved_rag_mode(context.settings),
        cutover_stage=resolved_rag_stage(context.settings),
        configured_backend=backend,
        query_context_version=state['prepared_text'].query_context_version,
        current_text_hmac=state['prepared_text'].current_text_hmac,
        retrieval_query_hmac=state['prepared_text'].retrieval_query_hmac,
        security_scope_fingerprint=state['scope_fingerprint'],
        admission_cache_identity_hmac=None,
        source_window=f'rag-v2:admission:{resolved_rag_mode(context.settings)}:{context.surface}:{backend}',
        components=components,
    )
    return {
        'run_id': run_id,
        'generations': services.load_generations(),
        **(
            {'prepared_embedding': prepared_embedding}
            if prepared_embedding is not None
            else {}
        ),
    }


@_node
def prepare_and_call_query_embedding(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    from backend.app.agent_runtime.rag_application import ingress_budget_refusal

    services = runtime.context.services
    prepared = state['prepared_embedding']
    try:
        transport = _transport(services)
    except Exception:
        return _terminalize_failure(state, runtime.context, 'retriever_unavailable')
    try:
        grant = services.cost_ledger.claim_component(
            run_id=state['run_id'], component='query_embedding', prepared=prepared.budget
        )
    except RagPreclaimSafetyRefusalError:
        return _terminalize_failure(state, runtime.context, 'provider_safety_unavailable')
    dispatch = transport.prepare(grant=grant, prepared=prepared)
    delivery = transport.dispatch_and_finalize(grant=grant, prepared=dispatch)
    if delivery.output is None:
        output = ingress_budget_refusal(runtime.context.settings)
        output.update(
            outcome=delivery.component_final.terminal_outcome,
            charged_cost_usd=delivery.component_final.parent_total_charged_cost_usd,
            query_embedding_attempted=True,
            error_component=delivery.component_final.component,
        )
        if runtime.context.assistant_target is not None:
            output['effective_backend'] = 'pgvector'
        return output
    return {
        'query_embedding_result': delivery.output,
        'query_embedding_attempted': True,
    }


@_node
def pgvector_retrieval(state: RagGraphState, runtime: Runtime[RagRuntimeContext]):
    retriever = runtime.context.services.retrievers.resolve('pgvector')
    if isinstance(retriever, PgVectorEvidenceRetriever):
        retriever = retriever.with_graph_fallback()
    try:
        result = retriever.invoke(
            _request(state, embedding=state['query_embedding_result']),
            config={'callbacks': []},
        )
    except PgVectorFallbackRequiredError as exc:
        return {'fallback_category': exc.category}
    except Exception:
        return _terminalize_failure(state, runtime.context, 'retriever_unavailable')
    if type(result) is not RetrievalResult:
        raise TypeError('typed retrieval result is required')
    return {'retrieval': result}


@_node
def keyword_retrieval_on_pgvector_runtime_failure(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    try:
        result = runtime.context.services.retrievers.resolve('keyword').invoke(
            _request(state), config={'callbacks': []}
        )
    except Exception:
        return _terminalize_failure(state, runtime.context, 'retriever_unavailable')
    if type(result) is not RetrievalResult or result.configured_backend != 'keyword':
        raise TypeError('typed lexical fallback result is required')
    return {
        'retrieval': replace(
            result,
            configured_backend='pgvector',
            query_embedding_receipt=state['query_embedding_result'].receipt,
            trace=replace(
                result.trace,
                provider_attempt_count=1,
                fallback_category=state['fallback_category'],
            ),
        )
    }


@_node
def keyword_retrieval(state: RagGraphState, runtime: Runtime[RagRuntimeContext]):
    try:
        result = runtime.context.services.retrievers.resolve('keyword').invoke(
            _request(state), config={'callbacks': []}
        )
    except Exception:
        return _terminalize_failure(state, runtime.context, 'retriever_unavailable')
    if type(result) is not RetrievalResult:
        raise TypeError('typed retrieval result is required')
    return {'retrieval': result}


def _terminalize_failure(state, context, outcome):
    from backend.app.agent_runtime.rag_application import (
        RagApplicationError,
        ingress_budget_refusal,
    )

    if context.assistant_target is not None:
        from backend.app.agent_runtime.rag_finalization import (
            AssistantInterComponentFailureFinalizer,
        )
        finalizer = AssistantInterComponentFailureFinalizer(settings=context.settings,
            target=context.assistant_target, prepared_text=state['prepared_text'], outcome=outcome,
            security_scope=state['security_scope'],
            effective_backend=(state['retrieval'].effective_backend if state.get('retrieval')
                else 'pgvector' if state.get('configured_backend') == 'pgvector' else 'deterministic_lexical'),
            fallback_category=state.get('fallback_category'))
        if context.services.sqlite_scope is not None:
            record = context.services.sqlite_scope.finalize_failure(finalizer)
            charge = Decimal('0.000000')
        else:
            terminal = context.services.cost_ledger.finalize_inter_component_failure(
                run_id=state['run_id'], outcome=outcome, assistant_finalizer=finalizer)
            record, charge = finalizer.record, terminal.total_charged_cost_usd
        output = ingress_budget_refusal(context.settings)
        output.update(outcome=outcome, charged_cost_usd=charge,
                      assistant_finalization=record)
        return output
    if context.services.sqlite_scope is not None:
        # SQLite has no phase1 commit: its coordinator rolls back the entire
        # request on a typed failure rather than imitating paid ledger authority.
        raise RagApplicationError(outcome) from None

    terminal = context.services.cost_ledger.finalize_inter_component_failure(
        run_id=state['run_id'], outcome=outcome
    )
    output = ingress_budget_refusal(context.settings)
    output.update(
        outcome=terminal.outcome, charged_cost_usd=terminal.total_charged_cost_usd
    )
    return output


@_node
def canonical_permission_guard(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    services = runtime.context.services
    result = state['retrieval']
    if (
        len(result.visible) > state['visible_limit']
        or not 0 <= result.hidden_match_count <= 20
        or not 0 <= result.trace.candidate_window_count <= 50
    ):
        raise ValueError('retrieval bounds exceeded')
    visible = []
    for candidate in result.visible:
        fresh = services.evidence_resolver.resolve_candidate(
            db=services.db,
            identity=_identity_from_evidence(candidate.evidence),
            scope=state['security_scope'],
        )
        if fresh is not None:
            visible.append(replace(candidate, evidence=fresh))
    return {'candidates': tuple(visible)}


@_node
def rank_and_bound_evidence(state: RagGraphState, runtime: Runtime[RagRuntimeContext]):
    ordered = tuple(
        candidate
        for kind in ('trusted_knowledge', 'raw_chunk')
        for candidate in state['candidates']
        if candidate.evidence.serving_kind == kind
    )
    return {'candidates': ordered[: state['visible_limit']]}


@_node
def assemble_server_evidence_slots(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    from backend.app.agent_runtime.auto_review_input_safety import (
        scan_auto_review_plaintext,
    )

    candidates = state['candidates']
    if runtime.context.surface != 'search':
        safe = tuple(
            candidate
            for candidate in candidates
            if scan_auto_review_plaintext(candidate.evidence.model_content).allowed
        )
        if candidates and not safe:
            return {'evidence_slots': (), 'safe_outcome': 'safety_filter_empty'}
        candidates = safe
    return {'evidence_slots': rank_evidence_slots(candidates)}


@_node
def route_surface(state: RagGraphState, runtime: Runtime[RagRuntimeContext]):
    if runtime.context.surface == 'search':
        return {
            'route': 'keyword_search'
            if state['configured_backend'] == 'keyword'
            else 'paid_search'
        }
    return {
        'route': 'answer'
        if state['evidence_slots']
        else 'paid_embedding_only_safe'
        if state.get('query_embedding_result') is not None
        else 'provider_free_safe'
    }


@_node
def commit_zero_provider_search_costs_and_mark_projection_pending(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    if runtime.context.services.sqlite_scope is not None:
        return {
            'pending': runtime.context.services.sqlite_scope.mark_projection_pending(
                state['run_id']
            )
        }
    corpus, index = state['generations']
    pending = runtime.context.services.cost_ledger.commit_provider_free_pending(
        run_id=state['run_id'],
        corpus_generation=corpus,
        vector_index_generation=index,
    )
    return {'pending': pending}


@_node
def commit_paid_or_prepared_cost_components_and_mark_projection_pending(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    corpus, index = state['generations']
    return {
        'pending': runtime.context.services.cost_ledger.load_pending_projection(
            run_id=state['run_id'],
            corpus_generation=corpus,
            vector_index_generation=index,
        )
    }


@_node
def project_visible_search_results(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    # Prepare canonical inputs only. Authoritative fresh projection belongs to
    # the finalizer's shared C.5 / ownership transaction, never this node.
    prepared = PreparedRagFinalization(
        product_kind='search',
        tentative_outcome='search_projected',
        prepared_text=state['prepared_text'],
        security_scope=state['security_scope'],
        query_embedding_result=state.get('query_embedding_result'),
        retrieval_result=state['retrieval'],
        evidence_slots=state['evidence_slots'],
        model_influence_observations=(),
        selected_slot_ids=(),
        validated_answer=None,
        canned_message_identity=None,
    )
    return {'prepared_finalization': prepared}


@_node
def finalize_run_and_search_projection(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    return _finalize_provider_free(state, runtime.context)


def _finalize_provider_free(state, context):
    services = context.services
    if services.sqlite_scope is not None:
        projection = services.sqlite_scope.finalize(
            state['pending'],
            state['prepared_finalization'],
            assistant_target=context.assistant_target,
        )
        return _projection_output(state, projection, charge=Decimal('0.000000'))
    charge = services.cost_ledger.total_charged_cost(state['run_id'])
    if services.db.new or services.db.dirty or services.db.deleted:
        raise RuntimeError('request has uncommitted writes before finalization')
    services.db.rollback()
    prepared = state['prepared_finalization']
    finalizer = services.finalizer_factory(state['pending'], prepared)
    projection = (
        finalizer.finalize_pre_generation_evidence_changed(
            state['pending'], prepared, assistant_target=context.assistant_target
        )
        if prepared.tentative_outcome == 'evidence_unavailable'
        else finalizer.finalize_paid_embedding_only_safe(
            state['pending'], prepared, assistant_target=context.assistant_target
        )
        if prepared.query_embedding_result is not None
        else finalizer.finalize_provider_free_safe(
            state['pending'], prepared, assistant_target=context.assistant_target
        )
    )
    return _projection_output(state, projection, charge=charge)


def _projection_output(state, projection, *, charge):
    assistant_record = projection if type(projection) is AssistantFinalizationRecord else None
    projection = _committed_canonical(projection)
    return {
        **({'assistant_finalization': assistant_record} if assistant_record else {}),
        'committed_projection': projection,
        'outcome': projection.outcome,
        'answer_blocks': None,
        'selected_slot_ids': (),
        'evidence_projection': projection.evidence,
        'model_influence': projection.model_influence,
        'hidden_match_count': projection.hidden_match_count,
        'effective_backend': projection.effective_backend,
        'fallback_category': state['retrieval'].trace.fallback_category,
        'charged_cost_usd': charge,
    }


def _committed_canonical(projection):
    if type(projection) is AssistantFinalizationRecord:
        projection = projection.canonical_projection
    if type(projection) is not CanonicalRagProjection:
        raise TypeError('finalizer requires a committed canonical projection')
    return projection


@_node
def provider_free_safe_outcome_finalizer(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    return _safe_outcome(state, runtime.context, embedding_only=False)


@_node
def paid_embedding_only_safe_outcome_finalizer(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    return _safe_outcome(state, runtime.context, embedding_only=True)


def _safe_outcome(state, context, *, embedding_only):
    corpus, index = state['generations']
    ledger = context.services.cost_ledger
    if state.get('pending') is not None:
        pending = state['pending']
    elif context.services.sqlite_scope is not None:
        pending = context.services.sqlite_scope.mark_projection_pending(state['run_id'])
    else:
        commit = (
            ledger.commit_embedding_only_pending
            if embedding_only
            else ledger.commit_provider_free_pending
        )
        pending = commit(
            run_id=state['run_id'],
            corpus_generation=corpus,
            vector_index_generation=index,
        )
    audit = (
        state.get('prepared_answer')
        if state.get('safe_outcome') == 'evidence_unavailable'
        else None
    )
    prepared = PreparedRagFinalization(
        product_kind='answer',
        tentative_outcome=state.get('safe_outcome')
        or ('hidden_only' if state['retrieval'].hidden_match_count else 'no_match'),
        prepared_text=state['prepared_text'],
        security_scope=state['security_scope'],
        query_embedding_result=state.get('query_embedding_result'),
        retrieval_result=state['retrieval'],
        evidence_slots=(),
        model_influence_observations=(),
        selected_slot_ids=(),
        validated_answer=None,
        canned_message_identity=(
            'rag-canned-evidence-unavailable:v1'
            if state.get('safe_outcome') == 'evidence_unavailable'
            else 'rag-canned-no-evidence:v1'
        ),
        rendered_input_hmac=audit.rendered_input_hmac if audit is not None else None,
        answer_model_config_snapshot_hmac=(
            audit.model_config_snapshot_hmac if audit is not None else None
        ),
        audit_only_prepared_observation_hmac=(
            state['prepared_model_influence'].aggregate_observation_hmac
            if audit is not None
            else None
        ),
    )
    return _finalize_provider_free(
        {**state, 'pending': pending, 'prepared_finalization': prepared}, context
    )


@_node
def prepare_and_preflight_answer_invocation(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
    from backend.app.agents.rag_orchestrator_agent.v2_answer import (
        RagAnswerModelBoundaryError,
    )

    services = runtime.context.services
    if services.answer_model is None:
        return _terminalize_failure(state, runtime.context, 'model_unavailable')
    try:
        draft = services.answer_model.prepare_input(
            question=state['prepared_text'].answer_question_text,
            slots=state['evidence_slots'],
            answer_question_hmac=state['prepared_text'].answer_question_hmac,
            retrieval_query_hmac=state['prepared_text'].retrieval_query_hmac,
        )
    except RagBudgetExceededError:
        return _terminalize_failure(state, runtime.context, 'budget_exceeded')
    except RagAnswerModelBoundaryError as exc:
        return _terminalize_failure(state, runtime.context, exc.code)
    corpus, index = state['generations']
    influence = CanonicalEvidenceProjector(
        db=services.db, settings=runtime.context.settings
    ).prepare_model_influence(
        draft.evidence_slots,
        scope=state['security_scope'],
        prepared_corpus_generation=corpus,
        prepared_index_generation=index
        if state.get('query_embedding_result')
        else None,
        prepared_readiness_hmac=state['prepared_embedding'].readiness_snapshot_hmac
        if state.get('query_embedding_result')
        else None,
        rendered_input_hmac=draft.rendered_input_hmac,
        graph_paths=state['retrieval'].graph_paths,
    )
    if not influence.observations:
        return _safe_outcome(
            {**state, 'safe_outcome': 'evidence_unavailable'},
            runtime.context,
            embedding_only=state.get('query_embedding_result') is not None,
        )
    prepared = services.answer_model.bind_influence(
        draft, model_influence=influence.observations
    )
    if services.sqlite_scope is not None:
        services.sqlite_scope.bind_answer_invocation(
            prepared,
            influence,
            model=services.answer_model,
            cost_policy=services.cost_policy,
        )
    else:
        services.cost_ledger.bind_answer_budget(
            run_id=state['run_id'],
            prepared=prepared.budget,
            invocation=prepared,
            model_influence=influence,
        )
    return {
        'prepared_answer': prepared,
        'prepared_model_influence': influence,
        'evidence_slots': prepared.evidence_slots,
    }


@_node
def generate_structured_answer_blocks(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    services = runtime.context.services
    prepared = state['prepared_answer']
    if services.sqlite_scope is not None:
        try:
            answer = services.sqlite_scope.generate_deterministic_answer(prepared)
        except Exception:
            if runtime.context.assistant_target is None:
                raise
            return _terminalize_failure(state, runtime.context, 'unexpected_internal_error')
        return {
            'validated_answer': answer,
            'answer_generation_attempted': False,
            'deterministic_answer_generated': True,
        }
    try:
        transport = _transport(services)
    except Exception:
        return _terminalize_failure(state, runtime.context, 'model_unavailable')
    try:
        grant = services.cost_ledger.claim_component(
            run_id=state['run_id'], component='answer_generation', prepared=prepared.budget
        )
    except RagPreclaimSafetyRefusalError:
        return _terminalize_failure(state, runtime.context, 'provider_safety_unavailable')
    dispatch = transport.prepare(grant=grant, prepared=prepared)
    from backend.app.agent_runtime.rag_provider_transport import (
        RagAnswerEvidenceChangedBeforeSendError,
    )

    try:
        delivery = transport.dispatch_and_finalize(grant=grant, prepared=dispatch)
    except RagAnswerEvidenceChangedBeforeSendError:
        corpus, index = state['generations']
        pending = services.cost_ledger.commit_answer_evidence_changed_pending(
            grant=grant,
            corpus_generation=corpus,
            vector_index_generation=index,
        )
        return _safe_outcome(
            {**state, 'pending': pending, 'safe_outcome': 'evidence_unavailable'},
            runtime.context,
            embedding_only=state.get('query_embedding_result') is not None,
        )
    if delivery.output is None:
        from backend.app.agent_runtime.rag_application import ingress_budget_refusal

        output = ingress_budget_refusal(runtime.context.settings)
        output.update(
            outcome=delivery.component_final.terminal_outcome,
            charged_cost_usd=delivery.component_final.parent_total_charged_cost_usd,
            answer_generation_attempted=True,
            error_component=delivery.component_final.component,
        )
        if runtime.context.assistant_target is not None:
            output.update(effective_backend=state['retrieval'].effective_backend,
                          fallback_category=state['retrieval'].trace.fallback_category)
        return output
    return {
        'validated_answer': delivery.output, 'answer_generation_attempted': True,
        'generation_component': delivery.component_final,
    }


@_node
def validate_claim_evidence_refs(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    answer = state['validated_answer']
    if type(answer) is not ValidatedAnswerBlocks or not set(
        answer.selected_slot_ids
    ).issubset({slot.slot_id for slot in state['prepared_answer'].evidence_slots}):
        return _terminalize_failure(
            state, runtime.context, 'citation_validation_failed'
        )
    return {'selected_slot_ids': answer.selected_slot_ids}


@_node
def commit_cost_components_and_mark_projection_pending(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    if runtime.context.services.sqlite_scope is not None:
        return {
            'pending': runtime.context.services.sqlite_scope.mark_projection_pending(
                state['run_id']
            )
        }
    corpus, index = state['generations']
    return {
        'pending': runtime.context.services.cost_ledger.load_pending_projection(
            run_id=state['run_id'],
            corpus_generation=corpus,
            vector_index_generation=index,
        )
    }


@_node
def revalidate_model_influence_and_selected_evidence(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    # Authenticate the transient handoff; fresh database reads remain inside the
    # finalizer's single authoritative phase-2 transaction.
    runtime.context.services.cost_policy.verify_answer_model_influence(
        state['evidence_slots'], state['prepared_model_influence'].observations
    )
    return {'influence_validated': True}


@_node
def recompute_bounded_hidden_count_and_selected_membership(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    membership = tuple(
        slot.slot_id
        for slot in state['evidence_slots']
        if slot.slot_id in state['selected_slot_ids']
    )
    if (
        not state['influence_validated']
        or set(membership) != set(state['selected_slot_ids'])
        or not 0 <= state['retrieval'].hidden_match_count <= 20
    ):
        raise ValueError('bounded finalization membership is invalid')
    return {'selected_membership': membership}


@_node
def project_selected_server_citations(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    answer = state['validated_answer']
    prepared = state['prepared_answer']
    return {
        'prepared_finalization': PreparedRagFinalization(
            product_kind='answer',
            tentative_outcome='supported' if answer.blocks else 'insufficient_evidence',
            prepared_text=state['prepared_text'],
            security_scope=state['security_scope'],
            query_embedding_result=state.get('query_embedding_result'),
            retrieval_result=state['retrieval'],
            evidence_slots=state['evidence_slots'],
            model_influence_observations=prepared.model_influence,
            selected_slot_ids=state['selected_slot_ids'],
            validated_answer=answer,
            canned_message_identity=None
            if answer.blocks
            else 'rag-canned-no-evidence:v1',
            prepared_model_influence=state['prepared_model_influence'],
            rendered_input_hmac=prepared.rendered_input_hmac,
            answer_model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
        )
    }


@_node
def finalize_run_and_answer_projection_or_assistant_message(
    state: RagGraphState, runtime: Runtime[RagRuntimeContext]
):
    services = runtime.context.services
    if services.sqlite_scope is not None:
        charge = Decimal('0.000000')
        projection = services.sqlite_scope.finalize(
            state['pending'],
            state['prepared_finalization'],
            assistant_target=runtime.context.assistant_target,
        )
    else:
        charge = services.cost_ledger.total_charged_cost(state['run_id'])
        if services.db.new or services.db.dirty or services.db.deleted:
            raise RuntimeError('request has uncommitted writes before finalization')
        services.db.rollback()
        finalizer = services.finalizer_factory(
            state['pending'], state['prepared_finalization']
        )
        projection = (
            finalizer.finalize_assistant(
                state['pending'],
                state['prepared_finalization'],
                runtime.context.assistant_target,
            )
            if runtime.context.assistant_target is not None
            else finalizer.finalize_direct(
                state['pending'], state['prepared_finalization']
            )
        )
    assistant_record = projection if type(projection) is AssistantFinalizationRecord else None
    projection = _committed_canonical(projection)
    return {
        **({'assistant_finalization': assistant_record} if assistant_record else {}),
        'committed_projection': projection,
        'outcome': projection.outcome,
        'answer_blocks': state['validated_answer']
        if projection.outcome == 'supported'
        else None,
        'selected_slot_ids': state['selected_slot_ids']
        if projection.outcome == 'supported'
        else (),
        'evidence_projection': projection.evidence,
        'model_influence': projection.model_influence,
        'hidden_match_count': projection.hidden_match_count,
        'effective_backend': projection.effective_backend,
        'fallback_category': state['retrieval'].trace.fallback_category,
        'charged_cost_usd': charge,
    }


def build_company_memory_rag_answer_v2_graph() -> CompiledStateGraph:
    builder = StateGraph(
        RagGraphState,
        input_schema=RagGraphInput,
        output_schema=RagGraphOutput,
        context_schema=RagRuntimeContext,
    )
    nodes = (
        validate_input,
        resolve_current_permission_context,
        select_configured_backend,
        preflight_retrieval_paid_cost_ceiling,
        keyword_retrieval,
        canonical_permission_guard,
        rank_and_bound_evidence,
        assemble_server_evidence_slots,
        route_surface,
        commit_zero_provider_search_costs_and_mark_projection_pending,
        project_visible_search_results,
        finalize_run_and_search_projection,
    )
    for node in nodes:
        builder.add_node(node.__name__, node)
    builder.add_node(
        'provider_free_safe_outcome_finalizer', provider_free_safe_outcome_finalizer
    )
    builder.add_node(
        'paid_embedding_only_safe_outcome_finalizer',
        paid_embedding_only_safe_outcome_finalizer,
    )
    answer_nodes = (
        prepare_and_preflight_answer_invocation,
        generate_structured_answer_blocks,
        validate_claim_evidence_refs,
        commit_cost_components_and_mark_projection_pending,
        revalidate_model_influence_and_selected_evidence,
        recompute_bounded_hidden_count_and_selected_membership,
        project_selected_server_citations,
        finalize_run_and_answer_projection_or_assistant_message,
    )
    for node in answer_nodes:
        builder.add_node(node.__name__, node)
    for left, right in zip(answer_nodes, answer_nodes[1:], strict=False):
        builder.add_conditional_edges(
            left.__name__,
            lambda state: 'stop' if 'outcome' in state else 'continue',
            {'stop': END, 'continue': right.__name__},
        )
    builder.add_edge(answer_nodes[-1].__name__, END)
    for node in (
        prepare_and_call_query_embedding,
        pgvector_retrieval,
        commit_paid_or_prepared_cost_components_and_mark_projection_pending,
        keyword_retrieval_on_pgvector_runtime_failure,
    ):
        builder.add_node(node.__name__, node)
    builder.add_conditional_edges(
        'prepare_and_call_query_embedding',
        lambda state: 'stop' if 'outcome' in state else 'continue',
        {'stop': END, 'continue': 'pgvector_retrieval'},
    )
    builder.add_conditional_edges(
        'pgvector_retrieval',
        lambda state: (
            'stop'
            if 'outcome' in state
            else 'fallback'
            if state.get('fallback_category')
            else 'success'
        ),
        {
            'stop': END,
            'fallback': 'keyword_retrieval_on_pgvector_runtime_failure',
            'success': 'canonical_permission_guard',
        },
    )
    builder.add_conditional_edges(
        'keyword_retrieval_on_pgvector_runtime_failure',
        lambda state: 'stop' if 'outcome' in state else 'continue',
        {'stop': END, 'continue': 'canonical_permission_guard'},
    )
    builder.add_edge(
        'commit_paid_or_prepared_cost_components_and_mark_projection_pending',
        'project_visible_search_results',
    )
    builder.add_edge(START, 'validate_input')
    for left, right in zip(nodes, nodes[1:], strict=False):
        if left is validate_input:
            builder.add_conditional_edges(
                left.__name__,
                lambda state: 'stop' if 'outcome' in state else 'continue',
                {'stop': END, 'continue': right.__name__},
            )
        elif left is preflight_retrieval_paid_cost_ceiling:
            builder.add_conditional_edges(
                left.__name__,
                lambda state: state['configured_backend'],
                {
                    'keyword': right.__name__,
                    'pgvector': 'prepare_and_call_query_embedding',
                },
            )
        elif left is keyword_retrieval:
            builder.add_conditional_edges(
                left.__name__,
                lambda state: 'stop' if 'outcome' in state else 'continue',
                {'stop': END, 'continue': right.__name__},
            )
        elif left is route_surface:
            builder.add_conditional_edges(
                left.__name__,
                lambda state: state['route'],
                {
                    'keyword_search': right.__name__,
                    'provider_free_safe': 'provider_free_safe_outcome_finalizer',
                    'paid_search': 'commit_paid_or_prepared_cost_components_and_mark_projection_pending',
                    'paid_embedding_only_safe': 'paid_embedding_only_safe_outcome_finalizer',
                    'answer': 'prepare_and_preflight_answer_invocation',
                },
            )
        else:
            builder.add_edge(left.__name__, right.__name__)
    builder.add_edge(nodes[-1].__name__, END)
    builder.add_edge('provider_free_safe_outcome_finalizer', END)
    builder.add_edge('paid_embedding_only_safe_outcome_finalizer', END)
    return builder.compile()
