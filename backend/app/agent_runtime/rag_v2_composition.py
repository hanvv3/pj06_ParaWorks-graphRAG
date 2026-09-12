from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from backend.app.agent_runtime.model_router import (
    build_rag_answer_model_config_snapshot_hmac,
)
from backend.app.agent_runtime.rag_application import (
    RagApplicationError,
    RagApplicationFacade,
)
from backend.app.agent_runtime.rag_graph import build_company_memory_rag_answer_v2_graph
from backend.app.agent_runtime.rag_v2_contracts import (
    COMPANY_MEMORY_RAG_GRAPH_VERSION,
    COMPANY_MEMORY_RAG_STATE_SCHEMA_VERSION,
    COMPANY_MEMORY_RAG_WORKFLOW,
)
from backend.app.agent_runtime.rag_v2_registry import (
    RagGraphRegistration,
    RagGraphRegistry,
    build_rag_manifest_registry,
)
from backend.app.agent_runtime.rag_v2_state import RagRequestServices
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    assert_answer_contract_registry_ready,
    build_answer_output_schema_hmac,
    build_answer_prompt_renderer_hmac,
)
from backend.app.core.config import Settings
from backend.app.rag.retrieval import build_query_embedding_model_config_snapshot_hmac


@dataclass(frozen=True, slots=True)
class RagRuntimeDependencies:
    request_factory: Callable[..., AbstractContextManager[RagRequestServices]]


@dataclass(frozen=True, slots=True)
class RagRuntimeBundle:
    compiled_graph: CompiledStateGraph
    graph_registry: RagGraphRegistry
    manifest_registry: AgentRegistry
    facade: RagApplicationFacade


def build_rag_v2_runtime(
    *,
    settings: Settings,
    session_factory: Callable[[], Session],
    dependencies: RagRuntimeDependencies | None = None,
    registration: RagGraphRegistration | None = None,
) -> RagRuntimeBundle:
    settings = settings.model_copy(deep=True)
    try:
        assert_answer_contract_registry_ready()
        build_rag_answer_model_config_snapshot_hmac(
            settings,
            output_schema_hmac=build_answer_output_schema_hmac(settings),
            prompt_renderer_hmac=build_answer_prompt_renderer_hmac(settings),
        )
        build_query_embedding_model_config_snapshot_hmac(settings)
    except (TypeError, ValueError):
        raise RagApplicationError('runtime_version_unavailable') from None
    graph = (
        registration.graph
        if registration is not None
        else build_company_memory_rag_answer_v2_graph()
    )
    registry = RagGraphRegistry()
    registry.register(
        registration
        or RagGraphRegistration(
            COMPANY_MEMORY_RAG_WORKFLOW,
            COMPANY_MEMORY_RAG_GRAPH_VERSION,
            COMPANY_MEMORY_RAG_STATE_SCHEMA_VERSION,
            graph,
        )
    )
    registry.seal()
    return RagRuntimeBundle(
        graph,
        registry,
        build_rag_manifest_registry(),
        RagApplicationFacade(
            settings,
            session_factory,
            (
                dependencies.request_factory
                if dependencies
                else default_rag_request_factory
            ),
            registry,
        ),
    )


@contextmanager
def default_rag_request_factory(
    *, session_factory, settings, actor, surface, assistant_target
):
    from pathlib import Path

    from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
    from backend.app.agent_runtime.rag_sqlite_smoke import SQLiteRagSmokeCoordinator
    from backend.app.agent_runtime.rag_v2_identity import ServerRagSecurityScopeResolver
    from backend.app.agents.rag_orchestrator_agent.v2_answer import (
        StructuredRagAnswerModel,
    )
    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
    from backend.app.rag.retrieval import RagRetrieverRegistry
    from backend.app.rag.search_store import SqlAlchemyKeywordSearchStore
    from backend.app.rag.trusted_evidence import ServingEvidenceResolver

    with session_factory() as inspection:
        engine = inspection.get_bind()
        if engine.dialect.name == 'postgresql':
            with _postgres_request_services(
                db=inspection, settings=settings, session_factory=session_factory
            ) as services:
                yield services
            return
    if engine.dialect.name != 'sqlite':
        raise RagApplicationError('runtime_version_unavailable')
    database = engine.url.database
    coordinator = SQLiteRagSmokeCoordinator(
        engine=engine,
        database_path=Path(database).absolute()
        if database not in {None, '', ':memory:'}
        else None,
        settings=settings,
    )
    with coordinator.request_scope() as scope:
        retrievers = RagRetrieverRegistry()
        retrievers.register(
            'keyword',
            KeywordEvidenceRetriever(
                store=SqlAlchemyKeywordSearchStore(db=scope.db, settings=settings),
                settings=settings,
            ),
        )
        policy = RagCostPolicy(
            settings=settings,
            answer_output_schema_hmac=build_answer_output_schema_hmac(settings),
            answer_prompt_renderer_hmac=build_answer_prompt_renderer_hmac(settings),
        )
        model = StructuredRagAnswerModel(
            routed_model=RoutedRagAnswerModel(
                model=object(),
                provider='openai',
                model_name='gpt-5.4-mini-2026-03-17',
                model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
            ),
            cost_policy=policy,
        )
        yield RagRequestServices(
            db=scope.db,
            security_scope_resolver=ServerRagSecurityScopeResolver(settings),
            retrievers=retrievers,
            evidence_resolver=ServingEvidenceResolver(settings=settings),
            cost_policy=policy,
            cost_ledger=None,
            policy_snapshots=(),
            allocate_run_id=None,
            load_generations=None,
            finalizer_factory=None,
            sqlite_scope=scope,
            answer_model=model,
        )


def _policy_snapshots(policy, settings):
    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
    from backend.app.agent_runtime.rag_runtime_contracts import (
        AuthorizedProviderPolicySnapshot,
    )

    return tuple(
        AuthorizedProviderPolicySnapshot(
            component=component,
            provider='openai',
            model='text-embedding-3-small' if query else 'gpt-5.4-mini-2026-03-17',
            reasoning_or_config_identity='dimensions:1536'
            if query
            else 'reasoning:none',
            authorized_model_config_version='rag-query-embedding-config:v1'
            if query
            else 'rag-answer-model-config:v1',
            authorized_model_config_snapshot_hmac=policy.query_embedding_model_config_snapshot_hmac
            if query
            else policy.answer_model_config_snapshot_hmac,
            authorized_cost_policy_version='rag-query-embedding-cost:v1'
            if query
            else 'rag-answer-cost:v1',
            authorized_token_estimator_version='openai-cl100k-text-embedding-3-small:v1'
            if query
            else 'openai-o200k-rag-answer:v1',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
            authorized_policy_snapshot_hmac=policy.authorized_policy_snapshot_hmac(
                component
            ),
        )
        for component, query in (
            ('query_embedding', True),
            ('answer_generation', False),
        )
    )


@contextmanager
def _postgres_request_services(*, db, settings, session_factory):
    from types import SimpleNamespace

    from sqlalchemy import text

    from backend.app.agent_runtime import rag_provider_transport as transport
    from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
    from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
    from backend.app.agent_runtime.provider_usage import StrictEmbeddingUsageParser
    from backend.app.agent_runtime.rag_advisory_locks import (
        rag_projection_owner_lock_id,
        register_advisory_identity_db,
    )
    from backend.app.agent_runtime.rag_v2_identity import ServerRagSecurityScopeResolver
    from backend.app.agents.rag_orchestrator_agent.v2_answer import (
        StructuredRagAnswerModel,
    )
    from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
        StrictQueryEmbeddingAdapter,
    )
    from backend.app.models import RagServingCorpusGeneration
    from backend.app.rag.index_readiness import RagV2ServingIndexReadinessService
    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
    from backend.app.rag.pgvector_retriever import PgVectorEvidenceRetriever
    from backend.app.rag.retrieval import RagRetrieverRegistry
    from backend.app.rag.search_store import (
        SqlAlchemyKeywordSearchStore,
        build_rag_v2_pgvector_search_store,
    )
    from backend.app.rag.trusted_evidence import ServingEvidenceResolver

    assembly = transport._assemble_rag_request_cost_authority(
        settings=settings, session=db
    )
    policy = assembly.cost_policy
    readiness = RagV2ServingIndexReadinessService(settings)
    retrievers = RagRetrieverRegistry()
    keyword = KeywordEvidenceRetriever(
        store=SqlAlchemyKeywordSearchStore(db=db, settings=settings), settings=settings
    )
    retrievers.register('keyword', keyword)
    if settings.rag_retrieval_backend == 'pgvector':
        store = build_rag_v2_pgvector_search_store(db=db, settings=settings)
        if store is None:
            raise RagApplicationError('retriever_not_configured')
        retrievers.register(
            'pgvector',
            PgVectorEvidenceRetriever(
                store=store,
                readiness=SimpleNamespace(inspect=lambda: readiness.inspect(db=db)),
                keyword_retriever=keyword,
                settings=settings,
            ),
        )
    model = (
        StructuredRagAnswerModel(
            routed_model=RoutedRagAnswerModel(
                model=object(),
                provider='openai',
                model_name='gpt-5.4-mini-2026-03-17',
                model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
            ),
            cost_policy=policy,
        )
        if settings.openai_api_key
        else None
    )
    provider = None
    client = None

    def provider_factory():
        nonlocal provider, client
        if not settings.openai_api_key:
            raise RagApplicationError('model_unavailable')
        if provider is None:
            client = transport._DirectOpenAIProviderClient(settings)
            provider = transport._assemble_rag_provider_dispatch_authority(
                store=assembly.store,
                provider_safety=assembly.provider_safety,
                provider_connection_factory=assembly.provider_connection_factory,
                evidence_barrier=assembly.evidence_barrier,
                identity_secret=fingerprint_secret_bytes(settings)[0],
                timeout_seconds=30,
                provider_client=client,
                settings=settings,
                answer_model=model,
                load_current_readiness=assembly.load_current_readiness,
                runtime_health=assembly.runtime_health,
            )
        return provider

    def allocate_run_id():
        # PostgreSQL's sequence supplies a unique id; register its exact dynamic
        # owner identity in a committed transaction before the ledger uses it.
        with db.get_bind().begin() as connection:
            run_id = connection.scalar(
                text("SELECT nextval(pg_get_serial_sequence('agent_runs', 'id'))")
            )
            register_advisory_identity_db(
                connection,
                rag_projection_owner_lock_id(run_id),
                identity_namespace='dynamic',
            )
        return run_id

    def generations():
        row = db.get(RagServingCorpusGeneration, 1)
        if row is None:
            raise RagApplicationError('serving_index_not_ready')
        return (
            row.corpus_generation,
            row.vector_index_generation
            if settings.rag_retrieval_backend == 'pgvector'
            else None,
        )

    final_sessions = []

    def finalizer_factory(pending, prepared):
        final_db = session_factory()
        final_sessions.append(final_db)
        return _postgres_finalizer(
            db=final_db,
            settings=settings,
            assembly=assembly,
            pending=pending,
            prepared=prepared,
        )

    try:
        yield RagRequestServices(
            db=db,
            security_scope_resolver=ServerRagSecurityScopeResolver(settings),
            retrievers=retrievers,
            evidence_resolver=ServingEvidenceResolver(settings=settings),
            cost_policy=policy,
            cost_ledger=assembly.store,
            policy_snapshots=_policy_snapshots(policy, settings),
            allocate_run_id=allocate_run_id,
            load_generations=generations,
            finalizer_factory=finalizer_factory,
            query_embedding_adapter=(
                StrictQueryEmbeddingAdapter(
                    usage_parser=StrictEmbeddingUsageParser(),
                    cost_policy=policy,
                    transport=object(),
                    settings=settings,
                )
                if settings.openai_api_key
                else None
            ),
            index_readiness=readiness,
            answer_model=model,
            provider_transport_factory=provider_factory,
        )
    finally:
        closers = [final_db.close for final_db in final_sessions]
        if client is not None:
            closers.append(client.close)
        for close in closers:
            try:
                close()
            except Exception:
                # Cleanup cannot replace a primary failure or acknowledged
                # product. Existing trusted health poisons future paid work.
                assembly.runtime_health._poison()


def _postgres_finalizer(*, db, settings, assembly, pending, prepared):
    from types import SimpleNamespace

    from sqlalchemy import select

    from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
    from backend.app.agent_runtime.provider_send_fence import (
        _assemble_rag_evidence_barrier,
    )
    from backend.app.agent_runtime.rag_advisory_locks import (
        RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
        RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
        load_registered_advisory_capability,
        rag_projection_owner_lock_id,
    )
    from backend.app.agent_runtime.rag_finalization import (
        RagFinalizationService,
        SqlAlchemyRagFinalizationBoundary,
        _assemble_paid_rag_phase2_authority,
        _assemble_provider_free_rag_phase2_authority,
    )
    from backend.app.agent_runtime.rag_postgres_binding import (
        _bind_rag_postgres_database,
    )
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService
    from backend.app.assistant.evidence_persistence import AssistantEvidenceWriter
    from backend.app.db.session import RagPostgresDatabaseBootstrap
    from backend.app.models import AgentRun
    from backend.app.rag.index_readiness import RagV2ServingIndexReadinessService
    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
    from backend.app.rag.pgvector_retriever import PgVectorEvidenceRetriever
    from backend.app.rag.search_store import (
        SqlAlchemyKeywordSearchStore,
        build_rag_v2_pgvector_search_store,
    )

    if db is assembly.session or db.get_bind() is not assembly.session.get_bind():
        raise RagApplicationError('runtime_version_unavailable')
    database = _bind_rag_postgres_database(
        db,
        trusted_bootstrap=RagPostgresDatabaseBootstrap,
        bootstrap_capability=assembly.bootstrap_capability,
    )
    try:
        with db.get_bind().connect() as connection:
            evidence_lock = load_registered_advisory_capability(
                connection,
                RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
                identity_namespace='static',
            )

        def capability(run_id):
            with database.connect() as connection:
                return load_registered_advisory_capability(
                    connection,
                    rag_projection_owner_lock_id(run_id),
                    identity_namespace='dynamic',
                )

        def owner_fence(run_id):
            with database.connect() as connection:
                return connection.scalar(
                    select(AgentRun.projection_owner_fence_hmac).where(
                        AgentRun.id == run_id
                    )
                )

        readiness = RagV2ServingIndexReadinessService(settings)

        def current_readiness():
            with (
                database.connect() as connection,
                Session(bind=connection) as observation,
            ):
                return readiness.inspect(db=observation)

        barrier = _assemble_rag_evidence_barrier(
            load_current_identity=lambda: current_readiness().readiness_snapshot_hmac,
            registered_lock=evidence_lock,
            postgres_database=database,
        )
        bindings = tuple(
            assembly.store._terminal_bindings[(pending.parent_agent_run_id, component)]
            for component in ('query_embedding', 'answer_generation')
            if (pending.parent_agent_run_id, component)
            in assembly.store._terminal_bindings
        )
        common = {
            'owner_connection_factory': None,
            'owner_capability_factory': capability,
            'load_current_owner_fence': owner_fence,
            'evidence_barrier': barrier,
            'postgres_database': database,
        }
        phase2_safety = None
        if bindings:
            with db.get_bind().connect() as connection:
                safety_lock = load_registered_advisory_capability(
                    connection,
                    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
                    identity_namespace='static',
                )
            # Phase 2 owns fresh pinned database connections, not phase 1's
            # one-physical-use transport. Its concrete database authority owns
            # lock release/disposal; retain the exact same durable safety root.
            phase2_safety = RagProviderSafetyService(
                latch_path=settings.paraworks_provider_safety_latch_path,
                identity_secret=fingerprint_secret_bytes(settings)[0],
                designated_environment_id=settings.paraworks_env,
                advisory_capability=safety_lock,
            )
        phase2 = (
            _assemble_paid_rag_phase2_authority(
                provider_safety=phase2_safety,
                safety_connection_factory=None,
                safety_requirements=tuple(
                    (binding.policy_snapshot, binding) for binding in bindings
                ),
                load_current_readiness=current_readiness,
                **common,
            )
            if bindings
            else _assemble_provider_free_rag_phase2_authority(**common)
        )
        keyword = KeywordEvidenceRetriever(
            store=SqlAlchemyKeywordSearchStore(db=db, settings=settings),
            settings=settings,
        )
        retriever = keyword
        if prepared.retrieval_result.configured_backend == 'pgvector':
            store = build_rag_v2_pgvector_search_store(db=db, settings=settings)
            if store is None:
                raise RagApplicationError('retriever_not_configured')
            retriever = PgVectorEvidenceRetriever(
                store=store,
                readiness=SimpleNamespace(inspect=lambda: readiness.inspect(db=db)),
                keyword_retriever=keyword,
                settings=settings,
            )
        secret, key_version = fingerprint_secret_bytes(settings)
        writer = AssistantEvidenceWriter(
            fingerprint_secret=secret,
            fingerprint_key_version=key_version,
            settings=settings,
        )
        return RagFinalizationService(
            transaction_boundary=SqlAlchemyRagFinalizationBoundary(
                db=db,
                settings=settings,
                retriever=retriever,
                assistant_writer=writer,
                phase2_authority=phase2,
            ),
            settings=settings,
        )
    except BaseException:
        database.close()
        raise
