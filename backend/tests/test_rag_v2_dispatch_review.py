from __future__ import annotations

import inspect
import typing
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine

from backend.app.agent_runtime.rag_cost_ledger import RagCostLedger
from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderDispatchAuthority,
    RagProviderTransportError,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    PreparedAnswerInvocation,
)
from backend.app.core.config import Settings
from backend.app.rag.retrieval import PreparedQueryEmbedding


def test_public_dispatch_boundary_has_no_raw_client_bytes_or_factory_hooks():
    import backend.app.agent_runtime.rag_provider_transport as transport_module
    import backend.app.agent_runtime.rag_runtime_contracts as contracts_module

    assert tuple(inspect.signature(RagCostLedger.__init__).parameters) == (
        'self',
        'authority',
    )
    assert tuple(
        inspect.signature(RagProviderDispatchAuthority.__init__).parameters
    ) == ('self', 'authority')
    production_assembler = transport_module._assemble_direct_openai_rag_provider_dispatch_authority
    assert 'provider_client' not in inspect.signature(
        production_assembler
    ).parameters
    assert tuple(inspect.signature(production_assembler).parameters) == ('settings',)
    source = inspect.getsource(production_assembler)
    for owned_authority in (
        'SessionLocal',
        'RagProviderSafetyService',
        '_assemble_rag_cost_ledger',
        '_assemble_rag_evidence_barrier',
        'RagV2ServingIndexReadinessService',
        'load_registered_advisory_capability',
        '_DirectOpenAIProviderClient',
        'RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID',
        'RAG_C5_KEY_CORPUS_AUTHORITY_LOCK_ID',
        'RAG_AGENT_RUN_COST_AUTHORITY_LOCK_ID',
    ):
        assert owned_authority in source
    assert not hasattr(transport_module, 'RagProviderRequest')
    assert not hasattr(contracts_module, 'StrictProviderOutcome')
    assert not hasattr(RagCostLedger, '_transport_provider_client')
    assert tuple(
        inspect.signature(RagProviderDispatchAuthority.prepare).parameters
    ) == ('self', 'grant', 'prepared')


def test_provider_prepare_accepts_only_frozen_domain_invocations():
    annotation = typing.get_type_hints(
        RagProviderDispatchAuthority.prepare
    )['prepared']
    assert annotation == PreparedQueryEmbedding | PreparedAnswerInvocation


def test_dispatch_authority_exposes_no_caller_send_or_raw_serialization_hooks():
    import backend.app.agent_runtime.provider_send_fence as fence_module

    assert not hasattr(fence_module, 'run_shared_advisory_send_fence')
    for method_name in ('prepare', 'dispatch', 'finalize'):
        method = getattr(RagProviderDispatchAuthority, method_name)
        parameters = set(inspect.signature(method).parameters)
        assert not parameters & {
            'provider_client',
            'send',
            'request_bytes',
            'rendered_input_utf8',
            'response_schema_json',
            'connection_factory',
            'freshness',
            'operation',
        }


def test_paid_assembler_rejects_sqlite_before_artifact_or_provider_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    import backend.app.agent_runtime.rag_provider_transport as transport_module
    import backend.app.db.session as db_session

    sqlite_engine = create_engine('sqlite+pysqlite:///:memory:')
    monkeypatch.setattr(db_session, 'engine', sqlite_engine)
    provider_clients = []

    class ForbiddenProviderClient:
        def __init__(self, *_args, **_kwargs):
            provider_clients.append('created')

    monkeypatch.setattr(
        transport_module,
        '_DirectOpenAIProviderClient',
        ForbiddenProviderClient,
    )
    latch = tmp_path / 'paid-provider-safety.json'
    settings = Settings(
        _env_file=None,
        paraworks_provider_safety_latch_path=str(latch),
        openai_api_key='test-key-never-used',
    )

    with pytest.raises(RagProviderTransportError, match='PostgreSQL'):
        transport_module._assemble_direct_openai_rag_provider_dispatch_authority(
            settings=settings
        )

    assert provider_clients == []
    assert not latch.exists()


def test_direct_paid_assembler_shares_one_exact_provider_connection_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import backend.app.agent_runtime.model_router as router_module
    import backend.app.agent_runtime.provider_send_fence as fence_module
    import backend.app.agent_runtime.rag_advisory_locks as lock_module
    import backend.app.agent_runtime.rag_cost_ledger as ledger_module
    import backend.app.agent_runtime.rag_postgres_binding as binding_module
    import backend.app.agent_runtime.rag_provider_transport as transport_module
    import backend.app.db.session as db_session

    trusted_bootstrap = db_session.RagPostgresDatabaseBootstrap
    assert trusted_bootstrap is not None
    runtime_health = trusted_bootstrap._runtime_effect_authority(db_session.engine)
    class FakeEngine:
        dialect = SimpleNamespace(name='postgresql')

        def connect(self):
            return nullcontext(object())

    fake_engine = FakeEngine()
    monkeypatch.setattr(db_session, 'engine', fake_engine)
    monkeypatch.setattr(db_session, 'SessionLocal', lambda: object())
    monkeypatch.setattr(
        type(trusted_bootstrap),
        '_runtime_effect_authority',
        lambda self, application_engine: runtime_health,
    )
    monkeypatch.setattr(
        lock_module,
        'load_registered_advisory_capability',
        lambda *_args, **_kwargs: object(),
    )
    class ProviderTransport:
        def __call__(self):
            return nullcontext(object())

    provider_transport = ProviderTransport()
    monkeypatch.setattr(
        binding_module,
        '_bind_rag_postgres_advisory_transport',
        lambda *_args, **_kwargs: provider_transport,
        raising=False,
    )
    safety = object()
    safety_assemblies = []

    def assemble_safety(**kwargs):
        safety_assemblies.append(kwargs)
        return safety

    monkeypatch.setattr(
        transport_module,
        'RagProviderSafetyService',
        assemble_safety,
    )
    provider_factories = []
    store = object()

    def assemble_ledger(_session, **kwargs):
        provider_factories.append(kwargs['provider_connection_factory'])
        assert kwargs['provider_connection_factory'] is provider_transport
        assert kwargs['runtime_health'] is runtime_health
        return store

    monkeypatch.setattr(ledger_module, '_assemble_rag_cost_ledger', assemble_ledger)
    barrier = object()
    barrier_assemblies = []

    def assemble_barrier(**kwargs):
        barrier_assemblies.append(kwargs)
        return barrier

    monkeypatch.setattr(
        fence_module,
        '_assemble_rag_evidence_barrier',
        assemble_barrier,
    )
    monkeypatch.setattr(router_module, 'build_rag_answer_model_route', lambda **_kwargs: object())
    monkeypatch.setattr(transport_module, 'StructuredRagAnswerModel', lambda **_kwargs: None)
    provider_client = SimpleNamespace(send=lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        transport_module,
        '_DirectOpenAIProviderClient',
        lambda *_args, **_kwargs: provider_client,
    )
    assembled = object()

    def assemble_dispatch(**kwargs):
        assert kwargs['store'] is store
        assert kwargs['provider_safety'] is safety
        assert kwargs['evidence_barrier'] is barrier
        assert kwargs['runtime_health'] is runtime_health
        assert kwargs['provider_connection_factory'] is provider_factories[0]
        return assembled

    monkeypatch.setattr(
        transport_module,
        '_assemble_rag_provider_dispatch_authority',
        assemble_dispatch,
    )

    result = transport_module._assemble_direct_openai_rag_provider_dispatch_authority(
        settings=Settings(
            _env_file=None,
            paraworks_provider_safety_latch_path=str(tmp_path / 'provider-safety.json'),
            openai_api_key='test-key-never-used',
        )
    )

    assert result is assembled
    assert len(provider_factories) == 1
    assert safety_assemblies[0]['advisory_transport'] is provider_transport
    assert barrier_assemblies[0]['advisory_transport'] is provider_transport


def test_sqlite_smoke_assembly_is_explicit_and_provider_free(
    monkeypatch: pytest.MonkeyPatch,
):
    import backend.app.agent_runtime.rag_provider_transport as transport_module
    import backend.app.db.session as db_session

    monkeypatch.setattr(
        db_session,
        'engine',
        create_engine('sqlite+pysqlite:///:memory:'),
    )

    class ForbiddenProviderClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError('provider client must not exist in SQLite smoke')

    monkeypatch.setattr(
        transport_module,
        '_DirectOpenAIProviderClient',
        ForbiddenProviderClient,
    )

    smoke = transport_module._assemble_sqlite_provider_free_rag_smoke(
        settings=Settings(_env_file=None)
    )

    assert smoke.backend == 'deterministic_lexical'
    assert smoke.provider_dispatch_enabled is False
    assert not hasattr(smoke, 'dispatch')


@pytest.mark.parametrize(
    'missing_lock_name',
    (
        'provider_safety_authority',
        'evidence_provider_send',
        'projection_owner_registry',
        'c5_key_corpus_authority',
        'agent_run_cost_authority',
    ),
)
def test_paid_assembler_requires_every_committed_static_capability_before_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    missing_lock_name: str,
):
    import backend.app.agent_runtime.rag_advisory_locks as lock_module
    import backend.app.agent_runtime.rag_postgres_binding as binding_module
    import backend.app.agent_runtime.rag_provider_transport as transport_module
    import backend.app.db.session as db_session

    trusted_bootstrap = db_session.RagPostgresDatabaseBootstrap
    assert trusted_bootstrap is not None
    runtime_health = trusted_bootstrap._runtime_effect_authority(db_session.engine)
    fake_engine = SimpleNamespace(
        dialect=SimpleNamespace(name='postgresql'),
        connect=lambda: nullcontext(object()),
    )
    monkeypatch.setattr(db_session, 'engine', fake_engine)
    monkeypatch.setattr(
        type(trusted_bootstrap),
        '_runtime_effect_authority',
        lambda self, application_engine: runtime_health,
    )

    def load_registered(_connection, identity, *, identity_namespace):
        assert identity_namespace == 'static'
        if identity['lock_name'] == missing_lock_name:
            raise RuntimeError(f'missing {missing_lock_name}')
        return object()

    monkeypatch.setattr(
        lock_module,
        'load_registered_advisory_capability',
        load_registered,
    )

    class ProviderTransport:
        def __call__(self):
            return nullcontext(object())

    monkeypatch.setattr(
        binding_module,
        '_bind_rag_postgres_advisory_transport',
        lambda *_args, **_kwargs: ProviderTransport(),
    )
    provider_clients = []

    class ForbiddenProviderClient:
        def __init__(self, *_args, **_kwargs):
            provider_clients.append('created')

    monkeypatch.setattr(
        transport_module,
        '_DirectOpenAIProviderClient',
        ForbiddenProviderClient,
    )
    latch = tmp_path / f'{missing_lock_name}.json'
    settings = Settings(
        _env_file=None,
        paraworks_provider_safety_latch_path=str(latch),
        openai_api_key='test-key-never-used',
    )

    with pytest.raises(RuntimeError, match=missing_lock_name):
        transport_module._assemble_direct_openai_rag_provider_dispatch_authority(
            settings=settings
        )

    assert provider_clients == []
    assert not latch.exists()
