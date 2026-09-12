from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from time import monotonic_ns
from typing import TypeVar

from langsmith import tracing_context
from sqlalchemy.engine import Connection

from backend.app.agent_runtime.fingerprints import (
    canonical_json_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.provider_send_fence import (
    EvidenceIdentityChangedBeforeSendError,
    ProviderSendFenceError,
    RagEvidenceSendBarrier,
)
from backend.app.agent_runtime.provider_usage import (
    StrictChatUsageParser,
    StrictEmbeddingUsageParser,
)
from backend.app.agent_runtime.rag_advisory_locks import begin_rag_lock_order
from backend.app.agent_runtime.rag_cost_ledger import (
    RagCostLedger,
    RagCostLedgerError,
    RagCostPersistenceError,
    RagServingEvidenceChangedError,
)
from backend.app.agent_runtime.rag_embedding_delivery import (
    embedding_dispatch_receipt_hmac,
)
from backend.app.agent_runtime.rag_postgres_binding import (
    RagPostgresAdvisoryTransport,
)
from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyService,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
    CommittedRagDispatchGrant,
    RagComponentFinal,
    RagProviderSafetyBinding,
    RagRunTerminal,
    _ClassifiedProviderObservation,
    _issue_classified_provider_observation,
)
from backend.app.agent_runtime.rag_safety_identity import (
    rag_identity_hmac,
    require_lower_hmac,
)
from backend.app.agent_runtime.rag_v2_identity import (
    exact_utf8_bytes,
    fingerprint_secret_bytes,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    PreparedAnswerInvocation,
    StructuredRagAnswerModel,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES,
    ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT,
    RagAnswerOutputValidator,
    ValidatedAnswerBlocks,
)
from backend.app.core.config import Settings
from backend.app.db.initialization import (
    PostgresRuntimeHealthUnavailableError,
    TrustedPostgresEngineBootstrap,
    TrustedPostgresRuntimeHealth,
)
from backend.app.rag.embeddings import validate_query_embedding_vector
from backend.app.rag.retrieval import (
    AnswerGenerationCostInput,
    PreparedQueryEmbedding,
    QueryEmbeddingCallResult,
    QueryEmbeddingCostInput,
    QueryEmbeddingReceipt,
    validate_prepared_query_embedding,
    validate_rag_serving_index_readiness,
)
from backend.app.rag.vector_validation import CosineIndexableVectorValidator

_T = TypeVar('_T')
_PREPARED_SEAL = object()
_DISPATCH_ASSEMBLY_SEAL = object()
_PROVIDER_CLIENT_SEAL = object()


class RagProviderTransportError(RuntimeError):
    """Body-blind provider transport refusal."""


class RagAnswerEvidenceChangedBeforeSendError(RagProviderTransportError):
    """Exact answer grant retired before send; graph must finalize safe projection."""


class RagProviderSafetyRefusalError(RagProviderTransportError):
    """Request-local, acknowledged terminal from an actual pre-send safety refusal."""

    def __init__(self, terminal: RagRunTerminal) -> None:
        super().__init__('provider safety refused dispatch')
        self.terminal = terminal


@dataclass(frozen=True, slots=True)
class RagProviderDelivery:
    """Request-local validated output, released only after durable cost finalization."""

    component_final: RagComponentFinal
    output: QueryEmbeddingCallResult | ValidatedAnswerBlocks | None = field(repr=False)

    def __reduce_ex__(self, _protocol: int):
        raise TypeError('RAG provider delivery cannot be serialized')


@dataclass(frozen=True, slots=True)
class _ClassifiedDelivery:
    observation: _ClassifiedProviderObservation
    output: QueryEmbeddingCallResult | ValidatedAnswerBlocks | None = field(
        default=None, repr=False,
    )


class _DirectOpenAIProviderClient:
    """Exact direct-global OpenAI bindings owned only by runtime assembly."""

    __slots__ = ('_answer_model', '_embedding_client', '_owned_answer_close', '_closed')

    def __init__(self, settings: Settings, *, routed_model: object | None = None) -> None:
        if type(settings) is not Settings or not settings.openai_api_key:
            raise TypeError('direct OpenAI provider settings are unavailable')
        import httpx
        from openai import OpenAI

        from backend.app.agent_runtime.model_router import (
            build_rag_answer_model_route,
        )

        self._owned_answer_close = None
        self._closed = False
        http_client = httpx.Client(trust_env=False)
        try:
            self._embedding_client = OpenAI(
                api_key=settings.openai_api_key,
                base_url='https://api.openai.com/v1',
                timeout=30,
                max_retries=0,
                http_client=http_client,
            )
        except BaseException:
            with suppress(Exception):
                http_client.close()
            raise
        try:
            route = routed_model or build_rag_answer_model_route(settings=settings)
        except BaseException:
            with suppress(Exception):
                self.close()
            raise
        if routed_model is None:
            self._owned_answer_close = getattr(route, 'close_owned_clients', None)
        self._answer_model = route.model

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._embedding_client.close()
        finally:
            if self._owned_answer_close is not None:
                self._owned_answer_close()

    def send(
        self,
        request_bytes: bytes,
        *,
        timeout_seconds: int,
        max_retries: int,
    ) -> object:
        if (
            type(request_bytes) is not bytes
            or timeout_seconds != 30
            or max_retries != 0
        ):
            raise RagProviderTransportError('provider request controls are invalid')
        try:
            body = json.loads(request_bytes.decode('utf-8'))
        except Exception:
            raise RagProviderTransportError('provider request is invalid') from None
        if type(body) is not dict:
            raise RagProviderTransportError('provider request is invalid')
        if set(body) == {'dimensions', 'encoding_format', 'input', 'model'}:
            if (
                body.get('dimensions') != 1536
                or body.get('encoding_format') != 'float'
                or body.get('model') != 'text-embedding-3-small'
                or type(body.get('input')) is not str
                or not body['input']
            ):
                raise RagProviderTransportError('embedding request is invalid')
            response = self._embedding_client.embeddings.create(**body)
            return response.model_dump(mode='json')
        expected_controls = {
            'max_output_tokens': 512,
            'model': 'gpt-5.4-mini-2026-03-17',
            'reasoning': {'effort': 'none'},
            'service_tier': 'default',
            'store': False,
            'stream': False,
            'text': {'format': ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT},
            'tools': [],
        }
        if (
            set(body) != {'input', *expected_controls}
            or any(body.get(key) != value for key, value in expected_controls.items())
            or type(body.get('input')) is not list
        ):
            raise RagProviderTransportError('answer request is invalid')
        return self._answer_model.invoke(body['input'], config={'callbacks': []})


class _StoreOwnedProviderClient:
    __slots__ = ('_client', '_seal')

    def __init__(self, client: object, seal: object) -> None:
        if seal is not _PROVIDER_CLIENT_SEAL or not callable(
            getattr(client, 'send', None)
        ):
            raise TypeError('store-owned provider client is unavailable')
        self._client = client
        self._seal = seal

    def send(
        self,
        request_bytes: bytes,
        *,
        timeout_seconds: int,
        max_retries: int,
        order: object,
        order_capability: object,
    ) -> object:
        if self._seal is not _PROVIDER_CLIENT_SEAL:
            raise RagProviderTransportError('provider client capability is invalid')
        order.require(order_capability, stage='optional_assistant')
        return self._client.send(
            request_bytes,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


@dataclass(frozen=True, slots=True)
class _CanonicalProviderRequest:
    """Assembly-derived request; never accepted from a public caller."""

    component: str
    provider: str
    model: str
    endpoint: str
    service_tier: str
    model_config_snapshot_hmac: str
    rendered_input_utf8: bytes = field(repr=False)
    response_schema_json: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        targets = {
            'query_embedding': ('/v1/embeddings', 'text-embedding-3-small'),
            'answer_generation': (
                '/v1/responses', 'gpt-5.4-mini-2026-03-17'
            ),
        }
        try:
            text = self.rendered_input_utf8.decode('utf-8')
        except (AttributeError, UnicodeDecodeError):
            raise ValueError('rendered provider input is invalid') from None
        if (
            type(self.rendered_input_utf8) is not bytes
            or not text
            or self.component not in targets
            or (self.endpoint, self.model) != targets[self.component]
            or self.provider != 'openai'
            or self.service_tier != 'default'
            or (self.component == 'query_embedding')
            != (self.response_schema_json is None)
        ):
            raise ValueError('provider request target is invalid')
        for value in (
            self.model_config_snapshot_hmac,
        ):
            require_lower_hmac(value)


@dataclass(frozen=True, slots=True)
class _PreparedProviderDispatch:
    request_identity_hmac: str
    evidence_identity_hmac: str
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PreparedState:
    grant: object
    request: _CanonicalProviderRequest
    request_bytes: bytes = field(repr=False)
    snapshot: AuthorizedProviderPolicySnapshot
    binding: RagProviderSafetyBinding
    evidence_identity_hmac: str
    query_identity_hmac: str
    shadow_legacy_not_ready_permit: bool
    domain_prepared: PreparedQueryEmbedding | PreparedAnswerInvocation = field(
        repr=False
    )


@dataclass(frozen=True, slots=True)
class _ProviderFreeSqliteRagSmoke:
    backend: str = 'deterministic_lexical'
    provider_dispatch_enabled: bool = False


def _assemble_sqlite_provider_free_rag_smoke(
    *,
    settings: Settings,
) -> _ProviderFreeSqliteRagSmoke:
    """Explicit local smoke path that cannot construct or dispatch a provider."""
    from backend.app.db.session import engine

    if type(settings) is not Settings or engine.dialect.name != 'sqlite':
        raise TypeError('provider-free RAG smoke requires SQLite')
    return _ProviderFreeSqliteRagSmoke()


@dataclass(frozen=True, slots=True)
class _RagProviderDispatchAssembly:
    store: RagCostLedger = field(repr=False)
    provider_safety: RagProviderSafetyService = field(repr=False)
    provider_connection_factory: Callable[[], Connection] = field(repr=False)
    evidence_barrier: RagEvidenceSendBarrier = field(repr=False)
    identity_secret: bytes = field(repr=False)
    timeout_seconds: int
    provider_client: object = field(repr=False)
    settings: Settings = field(repr=False)
    answer_model: StructuredRagAnswerModel | None = field(repr=False)
    load_current_readiness: Callable[[], object] = field(repr=False)
    runtime_health: TrustedPostgresRuntimeHealth = field(repr=False)
    _seal: object = field(repr=False, compare=False)


def _assemble_rag_provider_dispatch_authority(
    *,
    store: RagCostLedger,
    provider_safety: RagProviderSafetyService,
    provider_connection_factory: Callable[[], Connection],
    evidence_barrier: RagEvidenceSendBarrier,
    identity_secret: bytes,
    timeout_seconds: int,
    provider_client: object,
    settings: Settings,
    answer_model: StructuredRagAnswerModel | None,
    load_current_readiness: Callable[[], object],
    runtime_health: TrustedPostgresRuntimeHealth,
) -> RagProviderDispatchAuthority:
    return RagProviderDispatchAuthority(_RagProviderDispatchAssembly(
        store=store,
        provider_safety=provider_safety,
        provider_connection_factory=provider_connection_factory,
        evidence_barrier=evidence_barrier,
        identity_secret=identity_secret,
        timeout_seconds=timeout_seconds,
        provider_client=provider_client,
        settings=settings,
        answer_model=answer_model,
        load_current_readiness=load_current_readiness,
        runtime_health=runtime_health,
        _seal=_DISPATCH_ASSEMBLY_SEAL,
    ))


@dataclass(frozen=True, slots=True)
class RagRequestCostAuthority:
    session: object
    store: RagCostLedger
    cost_policy: object
    provider_safety: RagProviderSafetyService
    provider_connection_factory: RagPostgresAdvisoryTransport
    evidence_barrier: RagEvidenceSendBarrier
    load_current_readiness: Callable[[], object]
    runtime_health: TrustedPostgresRuntimeHealth
    bootstrap_capability: object


def _assemble_rag_request_cost_authority(
    *,
    settings: Settings,
    session=None,
) -> RagRequestCostAuthority:
    """Assemble request-owned DB/cost authority without constructing providers."""
    import socket

    from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
    from backend.app.agent_runtime.provider_send_fence import (
        _assemble_rag_evidence_barrier,
    )
    from backend.app.agent_runtime.rag_advisory_locks import (
        RAG_AGENT_RUN_COST_AUTHORITY_LOCK_ID,
        RAG_C5_KEY_CORPUS_AUTHORITY_LOCK_ID,
        RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
        RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID,
        RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
        load_registered_advisory_capability,
        rag_projection_owner_lock_id,
    )
    from backend.app.agent_runtime.rag_cost_ledger import _assemble_rag_cost_ledger
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
    from backend.app.agent_runtime.rag_postgres_binding import (
        _bind_rag_postgres_advisory_transport,
    )
    from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
        build_answer_output_schema_hmac,
        build_answer_prompt_renderer_hmac,
    )
    from backend.app.db.session import (
        RagPostgresDatabaseBootstrap,
        SessionLocal,
        engine,
    )
    from backend.app.rag.index_readiness import RagV2ServingIndexReadinessService

    if type(settings) is not Settings:
        raise TypeError('production RAG settings are required')
    if engine.dialect.name != 'postgresql':
        raise RagProviderTransportError(
            'paid RAG dispatch requires PostgreSQL registered capabilities'
        )
    identity_secret, _ = fingerprint_secret_bytes(settings)
    if type(RagPostgresDatabaseBootstrap) is not TrustedPostgresEngineBootstrap:
        raise RagProviderTransportError(
            'paid RAG dispatch runtime health is unavailable'
        )
    runtime_health = RagPostgresDatabaseBootstrap._runtime_effect_authority(engine)

    application_connection_factory = engine.connect

    def load_application_capability(identity: object):
        with application_connection_factory() as connection:
            return load_registered_advisory_capability(
                connection,
                identity,
                identity_namespace='static',
            )

    bootstrap_capability = load_application_capability(
        RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID
    )
    session = SessionLocal() if session is None else session
    provider_connection_factory = _bind_rag_postgres_advisory_transport(
        session,
        trusted_bootstrap=RagPostgresDatabaseBootstrap,
        bootstrap_capability=bootstrap_capability,
    )

    def load_provider_capability(identity: object):
        with provider_connection_factory() as connection:
            return load_registered_advisory_capability(
                connection,
                identity,
                identity_namespace='static',
            )

    paid_capabilities = {
        RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID['lock_name']: bootstrap_capability,
        **{
            identity['lock_name']: load_provider_capability(identity)
            for identity in (
                RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
                RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
                RAG_C5_KEY_CORPUS_AUTHORITY_LOCK_ID,
                RAG_AGENT_RUN_COST_AUTHORITY_LOCK_ID,
            )
        },
    }
    provider_advisory = paid_capabilities['provider_safety_authority']
    evidence_advisory = paid_capabilities['evidence_provider_send']
    provider_safety = RagProviderSafetyService(
        latch_path=settings.paraworks_provider_safety_latch_path,
        identity_secret=identity_secret,
        designated_environment_id=settings.paraworks_env,
        advisory_capability=provider_advisory,
        advisory_transport=provider_connection_factory,
    )
    cost_policy = RagCostPolicy(
        settings=settings,
        answer_output_schema_hmac=build_answer_output_schema_hmac(settings),
        answer_prompt_renderer_hmac=build_answer_prompt_renderer_hmac(settings),
    )

    def load_projection_capability(agent_run_id: int):
        with provider_connection_factory() as connection:
            return load_registered_advisory_capability(
                connection,
                rag_projection_owner_lock_id(agent_run_id),
                identity_namespace='dynamic',
            )

    store = _assemble_rag_cost_ledger(
        session,
        identity_secret=identity_secret,
        cost_policy=cost_policy,
        provider_safety=provider_safety,
        provider_connection_factory=provider_connection_factory,
        designated_environment_id=settings.paraworks_env,
        designated_host_id=socket.gethostname(),
        projection_lock_capability_factory=load_projection_capability,
        runtime_health=runtime_health,
    )

    def load_current_readiness():
        session.expire_all()
        return RagV2ServingIndexReadinessService(settings).inspect(db=session)

    evidence_barrier = _assemble_rag_evidence_barrier(
        load_current_identity=lambda: (
            load_current_readiness().readiness_snapshot_hmac
        ),
        connection_factory=provider_connection_factory,
        registered_lock=evidence_advisory,
        advisory_transport=provider_connection_factory,
    )
    return RagRequestCostAuthority(session, store, cost_policy, provider_safety, provider_connection_factory,
        evidence_barrier, load_current_readiness, runtime_health, bootstrap_capability)


def _assemble_direct_openai_rag_provider_dispatch_authority(*, settings: Settings) -> RagProviderDispatchAuthority:
    """Compatibility entry point; graph assembly uses its request-owned cost slice."""
    from backend.app.agent_runtime.model_router import build_rag_answer_model_route
    assembly = _assemble_rag_request_cost_authority(settings=settings)
    routed_model = build_rag_answer_model_route(settings=settings)
    answer_model = StructuredRagAnswerModel(
        routed_model=routed_model,
        cost_policy=assembly.cost_policy,
    )
    return _assemble_rag_provider_dispatch_authority(
        store=assembly.store,
        provider_safety=assembly.provider_safety,
        provider_connection_factory=assembly.provider_connection_factory,
        evidence_barrier=assembly.evidence_barrier,
        identity_secret=fingerprint_secret_bytes(settings)[0],
        timeout_seconds=30,
        provider_client=_DirectOpenAIProviderClient(
            settings,
            routed_model=routed_model,
        ),
        settings=settings,
        answer_model=answer_model,
        load_current_readiness=assembly.load_current_readiness,
        runtime_health=assembly.runtime_health,
    )


class RagProviderDispatchAuthority:
    """Store-owned request builder and one-shot provider dispatch authority."""

    __slots__ = (
        '_barrier', '_client', '_connection_factory', '_prepared', '_safety',
        '_secret', '_store', '_timeout_seconds', '_settings', '_answer_model',
        '_load_current_readiness',
        '_runtime_health',
    )

    def __init__(self, authority: object) -> None:
        if (
            type(authority) is not _RagProviderDispatchAssembly
            or authority._seal is not _DISPATCH_ASSEMBLY_SEAL
        ):
            raise TypeError('provider dispatch requires assembly-owned authority')
        store = authority.store
        provider_safety = authority.provider_safety
        provider_connection_factory = authority.provider_connection_factory
        evidence_barrier = authority.evidence_barrier
        identity_secret = authority.identity_secret
        timeout_seconds = authority.timeout_seconds
        provider_client = authority.provider_client
        settings = authority.settings
        answer_model = authority.answer_model
        load_current_readiness = authority.load_current_readiness
        runtime_health = authority.runtime_health
        if (
            type(store) is not RagCostLedger
            or type(provider_safety) is not RagProviderSafetyService
            or provider_safety is not store.provider_safety_authority
            or provider_connection_factory is not store.provider_connection_factory
            or type(evidence_barrier) is not RagEvidenceSendBarrier
            or type(identity_secret) is not bytes
            or not identity_secret
            or type(timeout_seconds) is not int
            or timeout_seconds <= 0
            or not callable(getattr(provider_client, 'send', None))
            or type(settings) is not Settings
            or (
                answer_model is not None
                and type(answer_model) is not StructuredRagAnswerModel
            )
            or not callable(load_current_readiness)
            or type(runtime_health) is not TrustedPostgresRuntimeHealth
            or runtime_health is not store.runtime_health_authority
        ):
            raise TypeError('provider dispatch authority is unavailable')
        if type(provider_connection_factory) is RagPostgresAdvisoryTransport and (
            provider_connection_factory.runtime_health_authority
            is not runtime_health
            or provider_safety.advisory_transport_authority
            is not provider_connection_factory
            or evidence_barrier.advisory_transport_authority
            is not provider_connection_factory
        ):
            raise TypeError('provider advisory transport authority changed')
        self._store = store
        self._safety = provider_safety
        self._connection_factory = provider_connection_factory
        self._barrier = evidence_barrier
        self._client = _StoreOwnedProviderClient(
            provider_client, _PROVIDER_CLIENT_SEAL
        )
        self._secret = identity_secret
        self._timeout_seconds = timeout_seconds
        self._settings = settings
        self._answer_model = answer_model
        self._load_current_readiness = load_current_readiness
        self._runtime_health = runtime_health
        self._prepared: dict[int, _PreparedState] = {}

    def prepare(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: PreparedQueryEmbedding | PreparedAnswerInvocation,
    ) -> _PreparedProviderDispatch:
        try:
            with self._runtime_health._effect('rag_provider_prepare'):
                return self._prepare_under_health(
                    grant=grant,
                    prepared=prepared,
                    shadow_legacy_not_ready_permit=False,
                )
        except PostgresRuntimeHealthUnavailableError:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError(
                'provider runtime health refused preparation'
            ) from None

    def prepare_shadow_legacy(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: PreparedQueryEmbedding,
    ) -> _PreparedProviderDispatch:
        """Prepare the exact shadow-owned legacy embedding carrier."""
        try:
            with self._runtime_health._effect('rag_provider_prepare'):
                self._store.require_shadow_legacy_embedding_grant(grant)
                return self._prepare_under_health(
                    grant=grant,
                    prepared=prepared,
                    shadow_legacy_not_ready_permit=True,
                )
        except RagCostLedgerError:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError(
                'shadow legacy provider authority is unavailable'
            ) from None
        except PostgresRuntimeHealthUnavailableError:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError(
                'provider runtime health refused preparation'
            ) from None

    def _prepare_under_health(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: object,
        shadow_legacy_not_ready_permit: bool,
    ) -> _PreparedProviderDispatch:
        domain_prepared = prepared
        if type(prepared) not in {PreparedQueryEmbedding, PreparedAnswerInvocation}:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise TypeError('frozen provider invocation is required')
        try:
            if type(prepared) is PreparedQueryEmbedding:
                readiness = validate_rag_serving_index_readiness(
                    self._load_current_readiness()
                )
                if (
                    readiness[0] is not True
                    and shadow_legacy_not_ready_permit is not True
                ):
                    raise ValueError
                prepared = validate_prepared_query_embedding(
                    prepared,
                    settings=self._settings,
                    expected_readiness=readiness,
                )
            else:
                if self._answer_model is None:
                    raise ValueError
                prepared = self._answer_model.validate_prepared_invocation(prepared)
            domain_prepared = prepared
        except Exception:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError(
                'frozen provider invocation is invalid'
            ) from None
        try:
            snapshot, binding, budget, query_identity_hmac, context_version = (
                self._store._transport_context(grant)
            )
        except RagCostLedgerError as exc:
            raise RagProviderTransportError('committed dispatch grant is unavailable') from exc
        try:
            request, body, expected_budget = self._canonical_request(prepared)
            # Embedding preparation has its own model-policy identity. Bind the
            # exact validated bytes separately to the admission's text context.
            prepared_query_identity = (
                keyed_fingerprint(
                    exact_utf8_bytes(prepared.transient_query_utf8.decode('utf-8')),
                    secret=fingerprint_secret_bytes(self._settings)[0],
                    schema_version='rag-retrieval-query-bytes:v1',
                    policy_version=context_version,
                )
                if type(prepared) is PreparedQueryEmbedding
                else prepared.retrieval_query_hmac
            )
        except (TypeError, ValueError):
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError(
                'provider request cost identity is invalid'
            ) from None
        if (
            request.component != grant.component
            or request.provider != snapshot.provider
            or request.model != snapshot.model
            or not hmac.compare_digest(
                request.model_config_snapshot_hmac,
                snapshot.authorized_model_config_snapshot_hmac,
            )
            or not hmac.compare_digest(
                grant.provider_safety_snapshot_hmac,
                binding.provider_safety_snapshot_hmac,
            )
            or not hmac.compare_digest(
                prepared_query_identity,
                query_identity_hmac,
            )
        ):
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('provider request authority is misaligned')
        if expected_budget != budget:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('provider request cost identity drifted')
        request_bytes = canonical_json_bytes(body)
        try:
            evidence_identity_hmac = self._barrier.snapshot_identity()
        except Exception:
            self._cancel_unconsumed_claim(
                grant,
                outcome='evidence_unavailable',
            )
            raise RagProviderTransportError(
                'evidence identity is unavailable'
            ) from None
        identity = self._request_identity(
            request=request,
            request_bytes=request_bytes,
            grant=grant,
            binding=binding,
            query_identity_hmac=query_identity_hmac,
            evidence_identity_hmac=evidence_identity_hmac,
            shadow_legacy_not_ready_permit=shadow_legacy_not_ready_permit,
        )
        dispatch = _PreparedProviderDispatch(
            request_identity_hmac=identity,
            evidence_identity_hmac=evidence_identity_hmac,
            _seal=_PREPARED_SEAL,
        )
        self._prepared[id(dispatch)] = _PreparedState(
            grant=grant,
            request=request,
            request_bytes=request_bytes,
            snapshot=snapshot,
            binding=binding,
            evidence_identity_hmac=evidence_identity_hmac,
            query_identity_hmac=query_identity_hmac,
            shadow_legacy_not_ready_permit=shadow_legacy_not_ready_permit,
            domain_prepared=domain_prepared,
        )
        return dispatch

    def _canonical_request(
        self,
        prepared: PreparedQueryEmbedding | PreparedAnswerInvocation,
    ) -> tuple[_CanonicalProviderRequest, dict[str, object], object]:
        if type(prepared) is PreparedQueryEmbedding:
            expected_budget = self._store.cost_policy_authority.prepare_query_embedding(
                QueryEmbeddingCostInput(
                    retrieval_query_utf8=prepared.transient_query_utf8,
                    model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
                )
            )
            request = _CanonicalProviderRequest(
                component='query_embedding',
                provider='openai',
                model='text-embedding-3-small',
                endpoint='/v1/embeddings',
                service_tier='default',
                model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
                rendered_input_utf8=prepared.transient_query_utf8,
            )
            body = {
                'dimensions': 1536,
                'encoding_format': 'float',
                'input': prepared.transient_query_utf8.decode('utf-8'),
                'model': 'text-embedding-3-small',
            }
            return request, body, expected_budget
        messages_json = canonical_json_bytes([
            {'content': content, 'ordinal': ordinal, 'role': role}
            for ordinal, (role, content) in enumerate(prepared.messages, start=1)
        ])
        expected_budget = self._store.cost_policy_authority.prepare_answer_generation(
            AnswerGenerationCostInput(
                exact_messages_json=messages_json,
                exact_response_schema_json=ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES,
                model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
            )
        )
        request = _CanonicalProviderRequest(
            component='answer_generation',
            provider='openai',
            model='gpt-5.4-mini-2026-03-17',
            endpoint='/v1/responses',
            service_tier='default',
            model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
            rendered_input_utf8=messages_json,
            response_schema_json=ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES,
        )
        body = {
            'input': [
                {'content': content, 'role': role}
                for role, content in prepared.messages
            ],
            'max_output_tokens': 512,
            'model': 'gpt-5.4-mini-2026-03-17',
            'reasoning': {'effort': 'none'},
            'service_tier': 'default',
            'store': False,
            'stream': False,
            'text': {'format': ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT},
            'tools': [],
        }
        return request, body, expected_budget

    def _cancel_unconsumed_claim(
        self,
        grant: object,
        *,
        outcome: str,
    ) -> None:
        if getattr(grant, 'consumed', None) is not False:
            return
        try:
            self._store.cancel_claim_before_dispatch(
                grant=grant,  # type: ignore[arg-type]
                outcome=outcome,
            )
        except RagCostPersistenceError:
            raise
        except RagCostLedgerError:
            raise RagProviderTransportError(
                'pre-send refusal could not be finalized'
            ) from None

    def dispatch(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: object,
    ) -> _ClassifiedProviderObservation:
        return self._dispatch_with_delivery(grant=grant, prepared=prepared).observation

    def dispatch_and_finalize(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: object,
    ) -> RagProviderDelivery:
        delivery = self._dispatch_with_delivery(grant=grant, prepared=prepared)
        final = self.finalize(grant=grant, observation=delivery.observation)
        output = delivery.output if final.terminal_outcome == 'component_succeeded' else None
        if type(output) is QueryEmbeddingCallResult:
            output = replace(output, committed_dispatch_hmac=embedding_dispatch_receipt_hmac(
                output, agent_run_id=final.agent_run_id,
                dispatch_fence_hmac=final.dispatch_fence_hmac,
                process_instance_hmac=final.process_instance_hmac,
                secret=self._secret,
            ))
        return RagProviderDelivery(
            component_final=final,
            output=output,
        )

    def _dispatch_with_delivery(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: object,
    ) -> _ClassifiedDelivery:
        try:
            with self._runtime_health._effect('rag_provider_dispatch'):
                return self._dispatch_under_health(
                    grant=grant,
                    prepared=prepared,
                )
        except PostgresRuntimeHealthUnavailableError:
            self._prepared.pop(id(prepared), None)
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError(
                'provider runtime health refused dispatch'
            ) from None

    def _dispatch_under_health(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: object,
    ) -> _ClassifiedDelivery:
        state = self._prepared.get(id(prepared))
        if (
            type(prepared) is not _PreparedProviderDispatch
            or prepared._seal is not _PREPARED_SEAL
            or state is None
            or state.grant is not grant
            or not hmac.compare_digest(
                prepared.request_identity_hmac,
                self._request_identity(
                    request=state.request,
                    request_bytes=state.request_bytes,
                    grant=grant,
                    binding=state.binding,
                    query_identity_hmac=state.query_identity_hmac,
                    evidence_identity_hmac=state.evidence_identity_hmac,
                    shadow_legacy_not_ready_permit=(
                        state.shadow_legacy_not_ready_permit
                    ),
                ),
            )
        ):
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('prepared provider dispatch is unavailable')
        self._prepared.pop(id(prepared), None)
        order = begin_rag_lock_order('ordinary')
        sidecar_capability = order.acquire('provider_stable_sidecar')
        safety_capability = order.acquire('provider_safety_rows')
        try:
            def send() -> _ClassifiedDelivery:
                if state.shadow_legacy_not_ready_permit:
                    self._store.require_shadow_legacy_embedding_grant(grant)
                self._store.revalidate_c5_before_send(
                    state.domain_prepared,
                    order=order,
                    order_capability=c5_capability,
                    load_current_readiness=self._load_current_readiness,
                    allow_not_ready_shadow=(
                        state.shadow_legacy_not_ready_permit
                    ),
                )
                cost_capability = order.acquire('agent_run_cost')
                self._store.consume_transport_grant(
                    grant,
                    order=order,
                    order_capability=cost_capability,
                )
                assistant_capability = order.acquire('optional_assistant')
                order.finish()
                started_ns = monotonic_ns()
                with tracing_context(enabled=False):
                    try:
                        result = self._client.send(
                            state.request_bytes,
                            timeout_seconds=self._timeout_seconds,
                            max_retries=0,
                            order=order,
                            order_capability=assistant_capability,
                        )
                    except Exception:
                        return _ClassifiedDelivery(
                            self._response_less_observation(state.request.component)
                        )
                return self._classify_response(
                    state, result,
                    latency_ms=max(0, (monotonic_ns() - started_ns) // 1_000_000),
                )

            with (
                self._connection_factory() as connection,
                self._safety.dispatch_barrier(
                    connection,
                    state.request.component,
                    state.snapshot,
                    expected_binding=state.binding,
                    order=order,
                    sidecar_capability=sidecar_capability,
                    safety_capability=safety_capability,
                ),
            ):
                projection_capability = order.acquire('projection_owner')
                with self._store.projection_owner_barrier(
                    grant.agent_run_id,
                    order=order,
                    order_capability=projection_capability,
                ):
                    evidence_capability = order.acquire('evidence_shared_barrier')
                    c5_capability = order.acquire('c5_key_corpus')
                    return self._barrier._run(
                        expected_identity_hmac=prepared.evidence_identity_hmac,
                        operation=send,
                        order=order,
                        evidence_capability=evidence_capability,
                        c5_capability=c5_capability,
                    )
        except RagCostPersistenceError:
            raise
        except (ProviderSendFenceError, RagCostLedgerError, RagProviderSafetyError) as exc:
            if getattr(grant, 'consumed', None) is False:
                if grant.component == 'answer_generation' and isinstance(
                    exc, (EvidenceIdentityChangedBeforeSendError, RagServingEvidenceChangedError)
                ):
                    self._store._defer_answer_evidence_refusal(grant)
                    raise RagAnswerEvidenceChangedBeforeSendError(
                        'answer evidence changed before provider send'
                    ) from None
                refusal_outcome = (
                    'evidence_unavailable'
                    if isinstance(exc, EvidenceIdentityChangedBeforeSendError)
                    else 'provider_safety_unavailable'
                )
                try:
                    terminal = self._store.cancel_claim_before_dispatch(
                        grant=grant,
                        outcome=refusal_outcome,
                    )
                except RagCostPersistenceError:
                    raise
                except RagCostLedgerError as cancel_exc:
                    raise RagProviderTransportError(
                        'pre-send refusal could not be finalized'
                    ) from cancel_exc
                if isinstance(exc, RagProviderSafetyError):
                    raise RagProviderSafetyRefusalError(terminal) from None
            message = (
                'evidence provider-send fence refused dispatch'
                if isinstance(exc, ProviderSendFenceError)
                else 'provider dispatch authority refused dispatch'
            )
            raise RagProviderTransportError(message) from None
        except Exception:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('provider transport failed') from None

    def finalize(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        observation: object,
    ):
        """Consume one transport-owned classified observation in the ledger."""
        return self._store.finalize_component(
            grant=grant,
            observation=observation,
        )

    @staticmethod
    def _response_less_observation(component: str) -> _ClassifiedProviderObservation:
        return _issue_classified_provider_observation(
            component=component,  # type: ignore[arg-type]
            classification='response_less_failure',
            terminal_outcome=(
                'retriever_unavailable'
                if component == 'query_embedding'
                else 'model_provider_failed'
            ),
            provider_dispatch_started=True,
            provider_response_received=False,
            strict_usage=None,
            actual_cost_usd=None,
            safety_action='unchanged',
        )

    def _classify_response(
        self,
        state: _PreparedState,
        response: object,
        *,
        latency_ms: int,
    ) -> _ClassifiedDelivery:
        if state.request.component == 'query_embedding':
            observation = self._classify_embedding(state, response)
        else:
            observation = self._classify_answer(state, response)
        if observation.classification != 'validated_success':
            return _ClassifiedDelivery(observation)
        if state.request.component == 'query_embedding':
            prepared = state.domain_prepared
            assert type(prepared) is PreparedQueryEmbedding
            assert observation.strict_usage is not None
            assert observation.actual_cost_usd is not None
            receipt = QueryEmbeddingReceipt(
                attempted=True,
                input_tokens=observation.strict_usage.input_tokens,
                actual_cost_usd=observation.actual_cost_usd,
                latency_ms=latency_ms,
                outcome='component_succeeded',
                model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
                provider_policy_snapshot_hmac=prepared.provider_policy_snapshot_hmac,
            )
            output = QueryEmbeddingCallResult(
                prepared=prepared,
                vector=validate_query_embedding_vector(response['data'][0]['embedding']),
                attempted=True,
                validated_input_tokens=observation.strict_usage.input_tokens,
                actual_cost_usd=observation.actual_cost_usd,
                receipt=receipt,
            )
        else:
            validator = RagAnswerOutputValidator(
                signer=self._store.cost_policy_authority.sign_answer_artifact,
            )
            output = validator.validate(
                response['parsed'], slots=state.domain_prepared.evidence_slots,
            )
        return _ClassifiedDelivery(observation, output)

    def _classify_embedding(
        self,
        state: _PreparedState,
        response: object,
    ) -> _ClassifiedProviderObservation:
        usage = None
        actual = None
        usage_invalid = False
        storage_invalid = False
        try:
            if type(response) is not dict:
                raise ValueError
            usage = StrictEmbeddingUsageParser().parse_usage(response.get('usage'))
        except Exception:
            usage_invalid = True
        if usage is not None:
            try:
                actual = self._store.cost_policy_authority.charge_actual(
                    'query_embedding', usage
                )
            except Exception:
                storage_invalid = True
        prepared = state.domain_prepared
        assert type(prepared) is PreparedQueryEmbedding
        if usage is not None and actual is not None and (
            usage.input_tokens > prepared.budget.estimated_input_tokens
            or self._store.cost_policy_authority.usage_exceeds_authorized_cap(
                'query_embedding', usage
            )
        ):
            return self._classified(
                'query_embedding', 'known_overrun', 'provider_usage_overrun',
                usage, actual, 'block_overrun'
            )
        data = response.get('data') if type(response) is dict else None
        item = data[0] if type(data) is list and len(data) == 1 else None
        identity_valid = (
            type(response) is dict
            and type(response.get('object')) is str
            and response.get('object') == 'list'
            and type(response.get('model')) is str
            and response.get('model') == 'text-embedding-3-small'
            and type(data) is list
            and len(data) == 1
            and type(item) is dict
            and type(item.get('object')) is str
            and item.get('object') == 'embedding'
            and type(item.get('index')) is int
            and item.get('index') == 0
        )
        if not identity_valid:
            return self._classified(
                'query_embedding', 'response_identity_invalid',
                'provider_response_identity_invalid', usage, actual,
                'block_remediation',
            )
        vector_valid = False
        try:
            vector = item['embedding']
            if type(vector) is not list:
                raise ValueError
            CosineIndexableVectorValidator().validate(
                vector,
                expected_dimensions=1536,
            )
            vector_valid = True
        except Exception:
            vector_valid = False
        if not vector_valid:
            return self._classified(
                'query_embedding', 'embedding_payload_invalid',
                'provider_embedding_payload_invalid', usage, actual,
                'block_remediation',
            )
        if usage_invalid:
            return self._classified(
                'query_embedding', 'usage_contract_invalid',
                'provider_safety_unavailable', None, None, 'block_remediation'
            )
        if storage_invalid:
            return self._classified(
                'query_embedding', 'usage_storage_invalid',
                'provider_safety_unavailable', None, None, 'block_remediation'
            )
        return self._classified(
            'query_embedding', 'validated_success', 'component_succeeded',
            usage, actual, 'unchanged'
        )

    def _classify_answer(
        self,
        state: _PreparedState,
        response: object,
    ) -> _ClassifiedProviderObservation:
        usage = None
        actual = None
        usage_invalid = False
        storage_invalid = False
        raw = response.get('raw') if type(response) is dict else None
        try:
            usage = StrictChatUsageParser().parse_message(raw)
        except Exception:
            usage_invalid = True
        if usage is not None:
            try:
                actual = self._store.cost_policy_authority.charge_actual(
                    'answer_generation', usage
                )
            except Exception:
                storage_invalid = True
        prepared = state.domain_prepared
        assert type(prepared) is PreparedAnswerInvocation
        if usage is not None and actual is not None and (
            usage.input_tokens > prepared.budget.estimated_input_tokens
            or usage.output_tokens > prepared.budget.maximum_output_tokens
            or self._store.cost_policy_authority.usage_exceeds_authorized_cap(
                'answer_generation', usage
            )
        ):
            return self._classified(
                'answer_generation', 'known_overrun', 'provider_usage_overrun',
                usage, actual, 'block_overrun'
            )
        try:
            metadata = raw.response_metadata
            identity_valid = (
                metadata.get('model') == 'gpt-5.4-mini-2026-03-17'
                and metadata.get('object') == 'response'
                and metadata.get('service_tier') == 'default'
            )
        except Exception:
            identity_valid = False
        if not identity_valid:
            return self._classified(
                'answer_generation', 'response_identity_invalid',
                'provider_response_identity_invalid', usage, actual,
                'block_remediation',
            )
        if usage_invalid:
            return self._classified(
                'answer_generation', 'usage_contract_invalid',
                'provider_safety_unavailable', None, None, 'block_remediation'
            )
        if storage_invalid:
            return self._classified(
                'answer_generation', 'usage_storage_invalid',
                'provider_safety_unavailable', None, None, 'block_remediation'
            )
        schema_valid = False
        validation_failure = 'structured_output_invalid'
        validator = RagAnswerOutputValidator(
            signer=self._store.cost_policy_authority.sign_answer_artifact
        )
        try:
            if response['parsing_error'] is not None:
                raise ValueError
            validator.validate(response['parsed'], slots=prepared.evidence_slots)
            schema_valid = True
        except Exception:
            if type(response) is dict and response.get('parsing_error') is None:
                validation_failure = validator.failure_classification(
                    response.get('parsed'), slots=prepared.evidence_slots
                )
        if not schema_valid:
            return self._classified(
                'answer_generation', validation_failure,
                validation_failure, usage, actual, 'unchanged'
            )
        return self._classified(
            'answer_generation', 'validated_success', 'component_succeeded',
            usage, actual, 'unchanged'
        )

    @staticmethod
    def _classified(
        component: str,
        classification: str,
        terminal_outcome: str,
        usage: object,
        actual: object,
        safety_action: str,
    ) -> _ClassifiedProviderObservation:
        return _issue_classified_provider_observation(
            component=component,  # type: ignore[arg-type]
            classification=classification,  # type: ignore[arg-type]
            terminal_outcome=terminal_outcome,  # type: ignore[arg-type]
            provider_dispatch_started=True,
            provider_response_received=True,
            strict_usage=usage,  # type: ignore[arg-type]
            actual_cost_usd=actual,  # type: ignore[arg-type]
            safety_action=safety_action,  # type: ignore[arg-type]
        )

    def _request_identity(
        self,
        *,
        request: _CanonicalProviderRequest,
        request_bytes: bytes,
        grant: CommittedRagDispatchGrant,
        binding: RagProviderSafetyBinding,
        query_identity_hmac: str,
        evidence_identity_hmac: str,
        shadow_legacy_not_ready_permit: bool,
    ) -> str:
        return rag_identity_hmac(
            {
                'component': request.component,
                'dispatch_fence_hmac': grant.dispatch_fence_hmac,
                'endpoint': request.endpoint,
                'evidence_identity_hmac': evidence_identity_hmac,
                'model': request.model,
                'model_config_snapshot_hmac': request.model_config_snapshot_hmac,
                'provider': request.provider,
                'provider_safety_snapshot_hmac': (
                    binding.provider_safety_snapshot_hmac
                ),
                'query_identity_hmac': query_identity_hmac,
                'rendered_input_sha256': hashlib.sha256(
                    request.rendered_input_utf8
                ).hexdigest(),
                'request_bytes_sha256': hashlib.sha256(request_bytes).hexdigest(),
                'service_tier': request.service_tier,
                'shadow_legacy_not_ready_permit': (
                    shadow_legacy_not_ready_permit
                ),
            },
            secret=self._secret,
            schema_version='rag-provider-dispatch-request:v1',
            policy_version='rag-provider-transport:v1',
        )
