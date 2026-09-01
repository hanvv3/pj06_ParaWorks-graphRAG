from __future__ import annotations

import os
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session

from backend.app.agent_runtime.provider_send_fence import (
    _assemble_rag_evidence_barrier,
)
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
    load_registered_advisory_capability,
    rag_projection_owner_lock_id,
    register_advisory_identity_db,
)
from backend.app.agent_runtime.rag_cost_ledger import _assemble_rag_cost_ledger
from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService
from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderTransportError,
    _assemble_rag_provider_dispatch_authority,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    _issue_classified_provider_observation as StrictProviderOutcome,
)
from backend.app.core.config import get_settings
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import StrictProviderUsage
from backend.tests.test_rag_v2_costs import (
    _TEST_COST_POLICY,
    _admit,
    _budget,
    _snapshot,
)
from backend.tests.test_rag_v2_provider_transport import (
    _TEST_SETTINGS,
    _admit_transport,
    _Client,
    _prepared_query,
)


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
    seen = []
    ledger = _assemble_rag_cost_ledger(
        Session(engine),
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
        provider_client=_Client(seen),
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
    ledger.recover_incomplete_run(
        run_id=run_id,
        dead_process_attestation_hmac=(
            ledger._expected_dead_process_attestation_hmac(
                run_id=run_id,
                dead_process_instance_hmac=ledger.process_instance_hmac,
            )
        ),
    )

    with pytest.raises(RagProviderTransportError):
        authority.dispatch(grant=grant, prepared=prepared)
    assert seen == []
