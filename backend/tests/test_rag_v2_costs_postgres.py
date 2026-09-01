from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    lock_runtime_state,
)
from backend.app.agent_runtime.provider_send_fence import (
    _assemble_rag_evidence_barrier,
)
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
    RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID,
    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
    acquire_advisory_lock,
    load_registered_advisory_capability,
    rag_projection_owner_lock_id,
    register_advisory_identity_db,
    release_advisory_lock,
)
from backend.app.agent_runtime.rag_cost_ledger import _assemble_rag_cost_ledger
from backend.app.agent_runtime.rag_finalization import (
    RagFinalizationError,
    SqlAlchemyRagFinalizationBoundary,
    _assemble_paid_rag_phase2_authority,
    _assemble_projection_owner_recovery_authority,
    _assemble_provider_free_rag_phase2_authority,
)
from backend.app.agent_runtime.rag_postgres_binding import (
    _bind_rag_postgres_database,
)
from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService
from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderTransportError,
    _assemble_rag_provider_dispatch_authority,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    _issue_classified_provider_observation as StrictProviderOutcome,
)
from backend.app.core.config import get_settings
from backend.app.models import AgentRun, AgentRunCostComponent
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import StrictProviderUsage
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    ServingProjectionReadCoordinator,
    build_serving_lock_plan,
)
from backend.tests.test_rag_v2_costs import (
    _TEST_COST_POLICY,
    _admit,
    _budget,
    _snapshot,
)
from backend.tests.test_rag_v2_provider_transport import (
    _TEST_SETTINGS,
    _admit_transport,
    _prepared_query,
)
from backend.tests.test_rag_v2_serving_locks import _seed_lock_prefix


@pytest.fixture
def postgres_cost_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[tuple[Engine, RagProviderSafetyService]]:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip('disposable PostgreSQL cost-authority gate is not configured')
    parsed = make_url(database_url)
    if parsed.host != '127.0.0.1' or parsed.port != 55432:
        pytest.fail('cost-authority PostgreSQL gate requires 127.0.0.1:55432')
    admin = create_engine(database_url)
    schema_name = f'rag_task12_cost_{uuid4().hex}'
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA {schema_name}'))
    query = dict(parsed.query)
    query['options'] = f'-csearch_path={schema_name},public'
    isolated_url = parsed.set(query=query).render_as_string(hide_password=False)
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', isolated_url)
    get_settings.cache_clear()
    command.upgrade(Config('alembic.ini'), 'head')
    engine = create_engine(isolated_url)
    try:
        with engine.begin() as connection:
            register_advisory_identity_db(
                connection,
                RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
                identity_namespace='static',
            )
            register_advisory_identity_db(
                connection,
                RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID,
                identity_namespace='static',
            )
        with engine.connect() as connection:
            capability = load_registered_advisory_capability(
                connection,
                RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
                identity_namespace='static',
            )
        service = RagProviderSafetyService(
            latch_path=tmp_path / 'provider-safety.json',
            identity_secret=b'task-12-test-identity-secret',
            designated_environment_id='test',
            advisory_capability=capability,
        )
        with engine.connect() as connection:
            service.bootstrap(
                connection,
                (
                    _snapshot('query_embedding', _TEST_COST_POLICY),
                    _snapshot('answer_generation', _TEST_COST_POLICY),
                ),
                reviewed_transition_reference_hmac='9' * 64,
            )
        yield engine, service
    finally:
        engine.dispose()
        get_settings.cache_clear()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA {schema_name} CASCADE'))
        admin.dispose()


def _database_bootstrap_capability(engine: Engine):
    with engine.connect() as connection:
        return load_registered_advisory_capability(
            connection,
            RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID,
            identity_namespace='static',
        )


def _database_authority(engine: Engine, session: Session):
    bootstrap = _database_bootstrap_capability(engine)
    dedicated_engine = create_engine(engine.url, poolclass=NullPool)
    return _bind_rag_postgres_database(
        session,
        dedicated_engine=dedicated_engine,
        bootstrap_capability=bootstrap,
    )


def test_postgres_failed_component_closes_exact_sibling_and_parent(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, service = postgres_cost_authority
    ledger = _assemble_rag_cost_ledger(
        Session(engine),
        identity_secret=b'task-12-test-identity-secret',
        cost_policy=_TEST_COST_POLICY,
        provider_safety=service,
        provider_connection_factory=engine.connect,
        designated_environment_id='test',
        designated_host_id='pytest-postgres-host',
    )
    _admit(ledger, 501)
    grant = ledger.claim_component(
        run_id=501,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    ledger.consume_committed_grant(grant)
    final = ledger.finalize_component(
        grant=grant,
        observation=StrictProviderOutcome(
            component='query_embedding',
            classification='response_less_failure',
            terminal_outcome='retriever_unavailable',
            provider_dispatch_started=True,
            provider_response_received=False,
            strict_usage=None,
            actual_cost_usd=None,
            safety_action='unchanged',
        ),
    )
    assert final.parent_status == 'failed'
    assert final.parent_total_charged_cost_usd == Decimal('0.000010')


def test_postgres_reviewed_intercomponent_recovery_accepts_terminal_zero_sibling(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, service = postgres_cost_authority
    ledger = _assemble_rag_cost_ledger(
        Session(engine),
        identity_secret=b'task-12-test-identity-secret',
        cost_policy=_TEST_COST_POLICY,
        provider_safety=service,
        provider_connection_factory=engine.connect,
        designated_environment_id='test',
        designated_host_id='pytest-postgres-host',
    )
    _admit(ledger, 502)
    grant = ledger.claim_component(
        run_id=502,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    ledger.consume_committed_grant(grant)
    ledger.finalize_component(
        grant=grant,
        observation=StrictProviderOutcome(
            component='query_embedding',
            classification='validated_success',
            terminal_outcome='component_succeeded',
            provider_dispatch_started=True,
            provider_response_received=True,
            strict_usage=StrictProviderUsage(20, 0, 20),
            actual_cost_usd=Decimal('0.000001'),
            safety_action='unchanged',
        ),
    )
    attestation = ledger._expected_dead_process_attestation_hmac(
        run_id=502,
        dead_process_instance_hmac=ledger.process_instance_hmac,
    )
    terminal = ledger.recover_incomplete_run(
        run_id=502,
        dead_process_attestation_hmac=attestation,
    )
    assert terminal.run_record_phase == 'admission_only'
    assert terminal.total_charged_cost_usd == Decimal('0.000001')


def test_postgres_projection_recovery_waits_for_owner_then_mutates_parent_once(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, service = postgres_cost_authority
    run_id = (uuid4().int % (2**31 - 1)) + 1
    with engine.begin() as connection:
        register_advisory_identity_db(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
        register_advisory_identity_db(
            connection,
            rag_projection_owner_lock_id(run_id),
            identity_namespace='dynamic',
        )
    with engine.connect() as connection:
        evidence_capability = load_registered_advisory_capability(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
    with engine.connect() as connection:
        owner_capability = load_registered_advisory_capability(
            connection,
            rag_projection_owner_lock_id(run_id),
            identity_namespace='dynamic',
        )
    ledger = _assemble_rag_cost_ledger(
        Session(engine),
        identity_secret=b'task-12-test-identity-secret',
        cost_policy=_TEST_COST_POLICY,
        provider_safety=service,
        provider_connection_factory=engine.connect,
        designated_environment_id='test',
        designated_host_id='pytest-postgres-recovery',
        projection_lock_capability_factory=lambda _run_id: owner_capability,
    )
    _admit(ledger, run_id)
    safety_requirements = []
    final = None
    for component, reserve, usage, actual in (
        (
            'query_embedding',
            '0.000010',
            StrictProviderUsage(20, 0, 20),
            Decimal('0.000001'),
        ),
        (
            'answer_generation',
            '0.002000',
            StrictProviderUsage(1, 10, 11),
            Decimal('0.000046'),
        ),
    ):
        grant = ledger.claim_component(
            run_id=run_id,
            component=component,
            prepared=_budget(component, reserve),
        )
        safety_requirements.append(
            (
                ledger._admission_snapshots[(run_id, component)],
                ledger._grant_bindings[(run_id, component)],
            )
        )
        ledger.consume_committed_grant(grant)
        final = ledger.finalize_component(
            grant=grant,
            observation=StrictProviderOutcome(
                component=component,
                classification='validated_success',
                terminal_outcome='component_succeeded',
                provider_dispatch_started=True,
                provider_response_received=True,
                strict_usage=usage,
                actual_cost_usd=actual,
                safety_action='unchanged',
            ),
        )
    assert final is not None
    assert final.projection_owner_fence_hmac is not None

    def load_fence(_run_id: int) -> str:
        with Session(engine) as session:
            parent = session.get(AgentRun, run_id)
            assert parent is not None
            assert parent.projection_owner_fence_hmac is not None
            return parent.projection_owner_fence_hmac

    projection_settings = get_settings()
    _seed_lock_prefix(ledger._session, projection_settings)
    postgres_database = _database_authority(engine, ledger._session)
    evidence_barrier = _assemble_rag_evidence_barrier(
        load_current_identity=lambda: 'a' * 64,
        registered_lock=evidence_capability,
        postgres_database=postgres_database,
    )
    provider_free = _assemble_provider_free_rag_phase2_authority(
        owner_connection_factory=None,
        owner_capability_factory=lambda _run_id: owner_capability,
        load_current_owner_fence=load_fence,
        evidence_barrier=evidence_barrier,
        postgres_database=postgres_database,
    )
    paid = _assemble_paid_rag_phase2_authority(
        provider_safety=service,
        safety_connection_factory=None,
        safety_requirements=tuple(safety_requirements),
        owner_connection_factory=None,
        owner_capability_factory=lambda _run_id: owner_capability,
        load_current_owner_fence=load_fence,
        evidence_barrier=evidence_barrier,
        load_current_readiness=lambda: RagServingIndexReadiness(
            ready=True,
            corpus_generation=1,
            vector_index_generation=1,
            expected_document_count=0,
            live_vector_count=0,
            tombstone_count=0,
            mismatch_count_capped_at_20=0,
            embedding_model='text-embedding-3-small',
            embedding_dimensions=1536,
            index_policy_version='rag-v2-serving-index:v1',
            readiness_snapshot_hmac='4' * 64,
        ),
        postgres_database=postgres_database,
    )
    recovery = _assemble_projection_owner_recovery_authority(
        ledger=ledger,
        provider_free=provider_free,
        paid=paid,
        projection_read=ServingProjectionReadCoordinator(
            db=ledger._session,
            settings=projection_settings,
        ),
        postgres_database=postgres_database,
    )
    live_owner = engine.connect()
    try:
        acquire_advisory_lock(live_owner, owner_capability, shared=False)
        with pytest.raises(RagFinalizationError, match='still live'):
            recovery.recover(run_id)
        with Session(engine) as probe:
            parent = probe.get(AgentRun, run_id)
            assert parent is not None and parent.status == 'running'
        release_advisory_lock(live_owner, owner_capability, shared=False)
    finally:
        live_owner.close()

    ledger._session.close()
    recovery_started = threading.Event()
    recovery_finished = threading.Event()
    recovery_pid: list[int] = []
    recovery_result: list[object] = []
    recovery_errors: list[BaseException] = []

    def recover_after_mutation_lock() -> None:
        recovery_session = Session(engine)
        try:
            recovery_ledger = _assemble_rag_cost_ledger(
                recovery_session,
                identity_secret=b'task-12-test-identity-secret',
                cost_policy=_TEST_COST_POLICY,
                provider_safety=service,
                provider_connection_factory=engine.connect,
                designated_environment_id='test',
                designated_host_id='pytest-postgres-recovery-worker',
                projection_lock_capability_factory=lambda _run_id: owner_capability,
            )
            worker_database = _database_authority(engine, recovery_session)
            worker_barrier = _assemble_rag_evidence_barrier(
                load_current_identity=lambda: 'a' * 64,
                registered_lock=evidence_capability,
                postgres_database=worker_database,
            )
            worker_provider_free = _assemble_provider_free_rag_phase2_authority(
                owner_connection_factory=None,
                owner_capability_factory=lambda _run_id: owner_capability,
                load_current_owner_fence=load_fence,
                evidence_barrier=worker_barrier,
                postgres_database=worker_database,
            )
            worker_paid = _assemble_paid_rag_phase2_authority(
                provider_safety=service,
                safety_connection_factory=None,
                safety_requirements=tuple(safety_requirements),
                owner_connection_factory=None,
                owner_capability_factory=lambda _run_id: owner_capability,
                load_current_owner_fence=load_fence,
                evidence_barrier=worker_barrier,
                load_current_readiness=lambda: RagServingIndexReadiness(
                    ready=True,
                    corpus_generation=1,
                    vector_index_generation=1,
                    expected_document_count=0,
                    live_vector_count=0,
                    tombstone_count=0,
                    mismatch_count_capped_at_20=0,
                    embedding_model='text-embedding-3-small',
                    embedding_dimensions=1536,
                    index_policy_version='rag-v2-serving-index:v1',
                    readiness_snapshot_hmac='4' * 64,
                ),
                postgres_database=worker_database,
            )
            worker_recovery = _assemble_projection_owner_recovery_authority(
                ledger=recovery_ledger,
                provider_free=worker_provider_free,
                paid=worker_paid,
                projection_read=ServingProjectionReadCoordinator(
                    db=recovery_session,
                    settings=projection_settings,
                ),
                postgres_database=worker_database,
            )
            recovery_pid.append(
                recovery_session.scalar(text('SELECT pg_backend_pid()'))
            )
            recovery_started.set()
            recovery_result.append(worker_recovery.recover(run_id))
        except BaseException as exc:  # pragma: no cover - surfaced below
            recovery_errors.append(exc)
        finally:
            recovery_finished.set()
            if 'worker_database' in locals():
                worker_database.close()
            recovery_session.close()

    worker: threading.Thread | None = None
    with Session(engine) as mutation_session, mutation_session.begin():
        plan = build_serving_lock_plan(
            mutation_session,
            ['history_event:999998'],
        )
        with KeyedMutationGuard.generation_barrier(mutation_session):
            key_context = lock_runtime_state(mutation_session, mode='share')
            ServingMutationLockCoordinator(
                db=mutation_session,
                settings=projection_settings,
            ).acquire(key_context=key_context, plan=plan)
            worker = threading.Thread(
                target=recover_after_mutation_lock,
                daemon=True,
            )
            worker.start()
            assert recovery_started.wait(5)
            deadline = time.monotonic() + 5
            observed_database_lock = False
            while time.monotonic() < deadline:
                with engine.connect() as observer:
                    wait_type = observer.scalar(
                        text(
                            'SELECT wait_event_type FROM pg_stat_activity '
                            'WHERE pid = :pid'
                        ),
                        {'pid': recovery_pid[0]},
                    )
                if wait_type == 'Lock':
                    observed_database_lock = True
                    break
                time.sleep(0.05)
            assert observed_database_lock is True
            assert recovery_finished.is_set() is False
            with Session(engine) as probe:
                parent = probe.get(AgentRun, run_id)
                assert parent is not None
                assert parent.status == 'running'
                assert parent.run_record_phase == 'cost_finalized_pending_projection'

    assert worker is not None
    worker.join(timeout=10)
    assert worker.is_alive() is False
    assert recovery_errors == []
    assert len(recovery_result) == 1
    terminal = recovery_result[0]
    assert terminal.outcome == 'persistence_failed'
    with Session(engine) as probe:
        parent = probe.get(AgentRun, run_id)
        assert parent is not None and parent.status == 'failed'
        assert parent.run_record_phase == 'final'
        children = tuple(
            probe.scalars(
                select(AgentRunCostComponent)
                .where(AgentRunCostComponent.agent_run_id == run_id)
                .order_by(AgentRunCostComponent.component_ordinal)
            )
        )
        assert tuple(value.dispatch_count for value in children) == (1, 1)
        assert tuple(value.actual_input_tokens for value in children) == (20, 1)
        assert tuple(value.actual_output_tokens for value in children) == (0, 10)
    postgres_database.close()


def test_postgres_database_authority_uses_dedicated_nullpool_without_app_reuse(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, _ = postgres_cost_authority
    session = Session(engine)
    dedicated_engine = create_engine(engine.url, poolclass=NullPool)
    disposed = 0

    def record_dispose(*_args: object) -> None:
        nonlocal disposed
        disposed += 1

    event.listen(dedicated_engine, 'engine_disposed', record_dispose)
    authority = _bind_rag_postgres_database(
        session,
        dedicated_engine=dedicated_engine,
        bootstrap_capability=_database_bootstrap_capability(engine),
    )
    assert type(dedicated_engine.pool) is NullPool
    application_checkouts = 0

    def checkout(*_args: object) -> None:
        nonlocal application_checkouts
        application_checkouts += 1

    event.listen(engine, 'checkout', checkout)
    try:
        with authority.connect() as first:
            assert first.scalar(text('SELECT current_database()'))
        with authority.connect() as second:
            assert second.scalar(text('SELECT current_database()'))
        assert application_checkouts == 0
    finally:
        event.remove(engine, 'checkout', checkout)
        authority.close()
        authority.close()
        assert disposed == 1
        with pytest.raises(TypeError, match='closed'):
            authority.connect()
        session.close()


def test_postgres_database_authority_preserves_injected_custom_creator_contract(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, _ = postgres_cost_authority
    session = Session(engine)
    args, kwargs = engine.dialect.create_connect_args(engine.url)
    application_name = f'rag-authority-{uuid4().hex}'
    creator_calls = 0
    creator_kwargs = dict(kwargs)
    creator_kwargs['application_name'] = application_name

    def creator():
        nonlocal creator_calls
        creator_calls += 1
        return engine.dialect.loaded_dbapi.connect(
            *args,
            **creator_kwargs,
        )

    invalid_url = engine.url.set(host='127.0.0.2', port=1)
    dedicated_engine = create_engine(
        invalid_url,
        poolclass=NullPool,
        creator=creator,
    )
    authority = _bind_rag_postgres_database(
        session,
        dedicated_engine=dedicated_engine,
        bootstrap_capability=_database_bootstrap_capability(engine),
    )
    try:
        with authority.connect() as connection:
            assert connection.scalar(
                text("SELECT current_setting('application_name')")
            ) == application_name
        assert creator_calls >= 2
    finally:
        authority.close()
        session.close()


def test_postgres_database_authority_preserves_injected_connect_args(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, _ = postgres_cost_authority
    session = Session(engine)
    application_name = f'rag-connect-args-{uuid4().hex}'
    dedicated_engine = create_engine(
        engine.url,
        poolclass=NullPool,
        connect_args={'application_name': application_name},
    )
    authority = _bind_rag_postgres_database(
        session,
        dedicated_engine=dedicated_engine,
        bootstrap_capability=_database_bootstrap_capability(engine),
    )
    try:
        with authority.connect() as connection:
            assert connection.scalar(
                text("SELECT current_setting('application_name')")
            ) == application_name
    finally:
        authority.close()
        session.close()


def test_postgres_database_authority_rejects_schema_split_during_binding(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, _ = postgres_cost_authority
    session = Session(engine)
    query = dict(engine.url.query)
    query['options'] = '-csearch_path=public'
    dedicated_engine = create_engine(
        engine.url.set(query=query),
        poolclass=NullPool,
    )
    disposed = 0

    def record_dispose(*_args: object) -> None:
        nonlocal disposed
        disposed += 1

    event.listen(dedicated_engine, 'engine_disposed', record_dispose)
    try:
        with pytest.raises(TypeError, match='does not match session'):
            _bind_rag_postgres_database(
                session,
                dedicated_engine=dedicated_engine,
                bootstrap_capability=_database_bootstrap_capability(engine),
            )
        assert disposed == 1
    finally:
        session.close()


def test_postgres_boundary_rejects_same_engine_search_path_drift(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, _ = postgres_cost_authority
    with engine.begin() as connection:
        register_advisory_identity_db(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
    with engine.connect() as connection:
        evidence_capability = load_registered_advisory_capability(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
    application_connection = engine.connect()
    session = Session(bind=application_connection)
    authority = _database_authority(engine, session)
    barrier = _assemble_rag_evidence_barrier(
        load_current_identity=lambda: 'a' * 64,
        registered_lock=evidence_capability,
        postgres_database=authority,
    )
    phase2 = _assemble_provider_free_rag_phase2_authority(
        owner_connection_factory=None,
        owner_capability_factory=lambda _run_id: (_ for _ in ()).throw(
            AssertionError('owner lock must not be reached')
        ),
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=barrier,
        postgres_database=authority,
    )
    try:
        session.execute(text('SET SESSION search_path TO public'))
        session.commit()
        with pytest.raises(TypeError, match='phase-2 database authority'):
            SqlAlchemyRagFinalizationBoundary(
                db=session,
                settings=get_settings(),
                retriever=SimpleNamespace(invoke=lambda request: request),
                phase2_authority=phase2,
            )
    finally:
        authority.close()
        session.close()
        application_connection.invalidate()
        application_connection.close()


def test_postgres_transport_rechecks_locked_cost_row_before_zero_call_send(
    postgres_cost_authority: tuple[Engine, RagProviderSafetyService],
):
    engine, service = postgres_cost_authority
    run_id = 503
    with engine.begin() as connection:
        register_advisory_identity_db(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
        register_advisory_identity_db(
            connection,
            rag_projection_owner_lock_id(run_id),
            identity_namespace='dynamic',
        )
    with engine.connect() as connection:
        evidence_lock = load_registered_advisory_capability(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
    with engine.connect() as connection:
        projection_lock = load_registered_advisory_capability(
            connection,
            rag_projection_owner_lock_id(run_id),
            identity_namespace='dynamic',
        )
    with engine.connect() as connection:
        provider_lock = load_registered_advisory_capability(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
    recovery_before_commit = threading.Event()
    allow_recovery_commit = threading.Event()
    sender_finished = threading.Event()
    recovery_finished = threading.Event()
    seen: list[bytes] = []

    class ForbiddenClient:
        def send(self, body: bytes, *, timeout_seconds: int, max_retries: int):
            seen.append(body)
            raise AssertionError('recovery must win before provider send')

    sender_session = Session(engine)
    recovery_session = Session(engine)
    ledger = _assemble_rag_cost_ledger(
        sender_session,
        identity_secret=b'task-12-test-identity-secret',
        cost_policy=_TEST_COST_POLICY,
        provider_safety=service,
        provider_connection_factory=engine.connect,
        designated_environment_id='test',
        designated_host_id='pytest-postgres-host',
        projection_lock_capability_factory=lambda value: (
            projection_lock if value == run_id else None
        ),
    )
    recovery_ledger = _assemble_rag_cost_ledger(
        recovery_session,
        identity_secret=b'task-12-test-identity-secret',
        cost_policy=_TEST_COST_POLICY,
        provider_safety=service,
        provider_connection_factory=engine.connect,
        designated_environment_id='test',
        designated_host_id='pytest-postgres-recovery-host',
        projection_lock_capability_factory=lambda value: (
            projection_lock if value == run_id else None
        ),
    )
    query_budget = _admit_transport(ledger, run_id)
    grant = ledger.claim_component(
        run_id=run_id,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _assemble_rag_provider_dispatch_authority(
        store=ledger,
        provider_safety=service,
        provider_connection_factory=engine.connect,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: '2' * 64,
            connection_factory=engine.connect,
            registered_lock=evidence_lock,
        ),
        identity_secret=b'task-12-test-identity-secret',
        timeout_seconds=30,
        provider_client=ForbiddenClient(),
        settings=_TEST_SETTINGS,
        answer_model=None,
        load_current_readiness=lambda: RagServingIndexReadiness(
            ready=True,
            corpus_generation=1,
            vector_index_generation=1,
            expected_document_count=0,
            live_vector_count=0,
            tombstone_count=0,
            mismatch_count_capped_at_20=0,
            embedding_model='text-embedding-3-small',
            embedding_dimensions=1536,
            index_policy_version='rag-v2-serving-index:v1',
            readiness_snapshot_hmac='4' * 64,
        ),
    )
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )
    sender_result: list[object] = []
    recovery_result: list[object] = []
    thread_errors: list[BaseException] = []

    @event.listens_for(recovery_session, 'before_commit')
    def hold_recovery_commit(_session: Session) -> None:
        recovery_before_commit.set()
        assert allow_recovery_commit.wait(timeout=10)

    def send() -> None:
        try:
            sender_result.append(
                authority.dispatch(grant=grant, prepared=prepared)
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            sender_result.append(exc)
        finally:
            sender_finished.set()

    def recover() -> None:
        try:
            recovery_result.append(recovery_ledger.recover_incomplete_run(
                run_id=run_id,
                dead_process_attestation_hmac=(
                    recovery_ledger._expected_dead_process_attestation_hmac(
                        run_id=run_id,
                        dead_process_instance_hmac=ledger.process_instance_hmac,
                    )
                ),
            ))
        except BaseException as exc:  # pragma: no cover - surfaced below
            thread_errors.append(exc)
        finally:
            recovery_finished.set()

    recovery = threading.Thread(target=recover, name='rag-recovery-worker')
    recovery.start()
    assert recovery_before_commit.wait(timeout=10)

    sender = threading.Thread(target=send, name='rag-provider-sender')
    sender.start()
    assert not sender_finished.wait(timeout=0.25)
    allow_recovery_commit.set()
    sender.join(timeout=10)
    recovery.join(timeout=10)

    assert not sender.is_alive()
    assert not recovery.is_alive()
    assert thread_errors == []
    assert len(sender_result) == 1
    assert isinstance(sender_result[0], RagProviderTransportError)
    assert len(recovery_result) == 1
    assert seen == []
    assert recovery_result[0].run_record_phase == 'admission_only'

    event.remove(recovery_session, 'before_commit', hold_recovery_commit)
    sender_session.close()
    recovery_session.close()
    probe_engine = create_engine(engine.url, poolclass=NullPool)
    with probe_engine.connect() as connection:
        for capability in (provider_lock, evidence_lock, projection_lock):
            acquired = connection.scalar(
                text('SELECT pg_try_advisory_lock(:key1, :key2)'),
                {'key1': capability.key[0], 'key2': capability.key[1]},
            )
            assert acquired is True
            assert connection.scalar(
                text('SELECT pg_advisory_unlock(:key1, :key2)'),
                {'key1': capability.key[0], 'key2': capability.key[1]},
            ) is True
    probe_engine.dispose()
    assert engine.pool.checkedout() == 0
