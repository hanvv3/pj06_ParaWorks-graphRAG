import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

import backend.app.models  # noqa: F401
from backend.app.agent_runtime.contracts import (
    AgentManifest,
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_preflight import (
    V21PreparedReviewConfig,
    create_or_reuse_review_thread,
    prepare_review_request,
)
from backend.app.agent_runtime.review_v21_extraction import (
    ExtractionCallStateError,
    ExtractionCallStore,
    ExtractionProviderSafetySnapshot,
    ProviderUsage,
    build_prepared_extraction_plan_set,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import (
    AgentRun,
    AutoReviewExtractionCall,
    AutoReviewProviderSafetyState,
    ReviewItem,
)
from backend.app.models.source import Source
from backend.app.schemas.review_workflow import ReviewWorkflowRunRequest


@pytest.fixture
def postgres_runtime(monkeypatch: pytest.MonkeyPatch):
    original_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not original_url:
        pytest.fail('PARAWORKS_TEST_POSTGRES_URL is required for Task 3 PostgreSQL store tests')
    admin = create_engine(original_url)
    if inspect(admin).get_table_names():
        pytest.fail('Task 3 PostgreSQL URL must name a freshly recreated empty database')
    schema = f'task3_{uuid4().hex}'
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA {schema}'))
        connection.execute(
            text(
                f'CREATE TABLE {schema}.alembic_version '
                '(version_num VARCHAR(32) NOT NULL PRIMARY KEY)'
            )
        )
    parsed = make_url(original_url)
    query = dict(parsed.query)
    query['options'] = f'-csearch_path={schema},public'
    isolated_url = parsed.set(query=query).render_as_string(hide_password=False)
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', isolated_url)
    get_settings.cache_clear()
    command.upgrade(Config('alembic.ini'), 'head')
    engine = create_engine(isolated_url)
    try:
        yield engine, isolated_url
    finally:
        engine.dispose()
        get_settings.cache_clear()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        admin.dispose()


def _settings(database_url: str) -> Settings:
    return Settings(
        paraworks_database_url=database_url,
        database_url=database_url,
        auto_review_mode='shadow',
        agent_runtime_security_scope_id='scope-task3',
        agent_runtime_fingerprint_secret='task3-postgres-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='task3-key-v1',
        auto_review_validator_input_cost_per_1m_tokens=Decimal('2.000000'),
        auto_review_validator_output_cost_per_1m_tokens=Decimal('12.000000'),
        auto_review_extraction_input_cost_per_1m_tokens=Decimal('0.750000'),
        auto_review_extraction_output_cost_per_1m_tokens=Decimal('4.500000'),
    )


def _actor() -> DemoUser:
    return DemoUser(
        id='task3-owner',
        email='task3-owner@example.test',
        role='employee',
        permission_levels={'internal'},
        name='Task 3 Owner',
        title='Tester',
        department='Quality',
    )


def _insert_provider_state(
    connection,
    *,
    purpose: str,
    model: str,
    reasoning: str,
    cost_policy: str,
    estimator: str,
    input_price: str,
    output_price: str,
) -> None:
    state_id = connection.scalar(
        text(
            'INSERT INTO auto_review_provider_safety_states '
            '(purpose, provider, model, reasoning_effort, state_version, '
            'authorized_cost_policy_version, token_estimator_version, '
            'tokenizer_encoding, reply_priming_tokens, framing_safety_tokens, '
            'input_usd_per_1m, output_usd_per_1m, breaker_open, overrun_count, '
            'authorized_at, last_event_sequence, created_at, updated_at) VALUES '
            '(:purpose, \'openai\', :model, :reasoning, 1, :cost_policy, '
            ':estimator, \'o200k_base\', 16, 512, :input_price, :output_price, '
            'false, 0, CURRENT_TIMESTAMP, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) '
            'RETURNING id'
        ),
        {
            'purpose': purpose,
            'model': model,
            'reasoning': reasoning,
            'cost_policy': cost_policy,
            'estimator': estimator,
            'input_price': Decimal(input_price),
            'output_price': Decimal(output_price),
        },
    )
    event_id = connection.scalar(
        text(
            'INSERT INTO auto_review_provider_safety_events '
            '(provider_safety_state_id, purpose, provider, model, reasoning_effort, '
            'event_sequence, event_kind, prior_state_version, new_state_version, '
            'cost_policy_version, token_estimator_version, tokenizer_encoding, '
            'reply_priming_tokens, framing_safety_tokens, input_usd_per_1m, '
            'output_usd_per_1m, prior_breaker_open, new_breaker_open, '
            'actor_subject_hmac, fingerprint_key_version, '
            'fingerprint_key_material_verifier, created_at) VALUES '
            '(:state_id, :purpose, \'openai\', :model, :reasoning, 1, '
            '\'initial_authorized\', 0, 1, :cost_policy, :estimator, '
            '\'o200k_base\', 16, 512, :input_price, :output_price, false, false, '
            ':actor_hmac, \'task3-key-v1\', :verifier, CURRENT_TIMESTAMP) '
            'RETURNING id'
        ),
        {
            'state_id': state_id,
            'purpose': purpose,
            'model': model,
            'reasoning': reasoning,
            'cost_policy': cost_policy,
            'estimator': estimator,
            'input_price': Decimal(input_price),
            'output_price': Decimal(output_price),
            'actor_hmac': 'a' * 64,
            'verifier': 'b' * 64,
        },
    )
    connection.execute(
        text(
            'UPDATE auto_review_provider_safety_states SET '
            'last_event_sequence=1, last_event_id=:event_id WHERE id=:state_id'
        ),
        {'event_id': event_id, 'state_id': state_id},
    )


def _seed_runtime(engine, database_url: str):
    settings = _settings(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO auto_review_runtime_key_states '
                '(component, fingerprint_key_version, '
                'fingerprint_key_material_verifier, generation, ready, updated_at) '
                "VALUES ('auto_review_trust_promotion', 'task3-key-v1', :verifier, "
                '1, true, clock_timestamp())'
            ),
            {'verifier': 'b' * 64},
        )
        _insert_provider_state(
            connection,
            purpose='extraction',
            model='gpt-5.4-mini-2026-03-17',
            reasoning='none',
            cost_policy='auto-review-extraction-cost:v1',
            estimator='openai-o200k-extraction:v1',
            input_price='0.750000',
            output_price='4.500000',
        )
        _insert_provider_state(
            connection,
            purpose='validation',
            model='gpt-5.6-terra',
            reasoning='medium',
            cost_policy='auto-review-cost:v1',
            estimator='openai-o200k-chat:v1',
            input_price='2.000000',
            output_price='12.000000',
        )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        source = Source(
            source_type='gmail',
            source_id=f'gmail:task3:{uuid4().hex}',
            source_url='https://private.example/task3',
            title='private title',
            permission_level='internal',
            raw_metadata={
                'content_signature': 'c' * 64,
                'source_snippet': '검토 가능한 근거 문장',
            },
            server_content_signature_schema='server-source-content:v1',
            server_content_signature='c' * 64,
        )
        db.add(source)
        db.flush()
        db.commit()
        request = ReviewWorkflowRunRequest(
            source_refs=[
                {
                    'source_type': 'gmail',
                    'source_id': source.source_id,
                    'version_or_signature': 'c' * 64,
                }
            ],
            agent_names=['history_agent'],
            client_request_id=f'task3-{uuid4().hex}',
        )
        registry = AgentRegistry()
        registry.register(
            AgentManifest(
                name='history_agent',
                owner='Developer C',
                input_contract='EvidencePacket',
                output_contract='AgentRunResult',
                prompt_versions=('history-extraction:v1',),
                supported_permissions=('internal',),
                capabilities=('history_generation',),
            )
        )
        prepare_review_request(
            db,
            request=request,
            actor=_actor(),
            registry=registry,
            settings=settings,
        )
        packet = EvidencePacket(
            source_type='company_memory',
            source_window='task3-postgres',
            messages=[
                EvidenceMessage(
                    source_id=source.source_id,
                    source_url=source.source_url,
                    text='검토 가능한 근거 문장',
                    author='owner@example.test',
                    timestamp='2026-08-28T00:00:00Z',
                    permission_level='internal',
                    source_snippet_override='검토 가능한 근거 문장',
                )
            ],
            permission_context=PermissionContext('task3-owner', 'employee'),
        )
        safety = ExtractionProviderSafetySnapshot(
            purpose='extraction',
            provider='openai',
            model='gpt-5.4-mini-2026-03-17',
            reasoning_effort='none',
            state_version=1,
            cost_policy_version='auto-review-extraction-cost:v1',
            token_estimator_version='openai-o200k-extraction:v1',
            tokenizer_encoding='o200k_base',
            reply_priming_tokens=16,
            framing_safety_tokens=512,
            input_usd_per_1m=Decimal('0.750000'),
            output_usd_per_1m=Decimal('4.500000'),
            breaker_open=False,
        )
        plans = build_prepared_extraction_plan_set(
            packet=packet,
            selected_agent_names=('history_agent',),
            settings=settings,
            fingerprint_key_material_verifier='b' * 64,
            safety_snapshots=(safety,),
        )
        plan = plans.plans[0]
        config = V21PreparedReviewConfig(
            configured_auto_review_mode='shadow',
            validator_provider='openai',
            validator_model='gpt-5.6-terra',
            validator_reasoning_effort='medium',
            validator_prompt_version='auto-review-validation:v1',
            validator_output_contract_version='candidate-validation-batch:v1',
            policy_version='auto-review-policy:v1',
            cost_policy_version='auto-review-cost:v1',
            fingerprint_key_version='task3-key-v1',
            fingerprint_key_material_verifier='b' * 64,
            token_estimator_version='openai-o200k-chat:v1',
            tokenizer_encoding='o200k_base',
            max_input_tokens_per_batch=6000,
            max_output_tokens_per_batch=3072,
            reply_priming_tokens=16,
            framing_safety_tokens=512,
            max_validation_batches_per_workflow=2,
            max_validation_candidates_per_batch=4,
            max_validation_candidates_per_workflow=5,
            max_provider_attempts=1,
            provider_timeout_seconds=60,
            provider_send_start_window_seconds=5,
            provider_attempt_lease_seconds=120,
            provider_commit_grace_seconds=30,
            validator_input_usd_per_1m=Decimal('2.000000'),
            validator_output_usd_per_1m=Decimal('12.000000'),
            enforce_percentage=0,
            authorized_percentage_at_launch=0,
            rollout_authorization_generation=1,
            validation_provider_safety_state_version=1,
            rollout_control_epoch=1,
            extraction_plan_set_hmac=plans.plan_set_hmac,
            extraction_provider_safety_snapshot_set_hmac=(
                plans.provider_safety_snapshot_set_hmac
            ),
            confirmed_extraction_cost_ceiling_usd=plans.confirmed_cost_ceiling_usd,
            confirmed_validation_cost_ceiling_usd=Decimal('0.097728'),
            confirmed_total_cost_ceiling_usd=Decimal('0.114444'),
            total_budget_limit_usd=Decimal('0.200000'),
            extraction_plan_identities=(plan.identity_payload(),),
        )
        prepared = prepare_review_request(
            db,
            request=request,
            actor=_actor(),
            registry=registry,
            settings=settings,
            v21_config=config,
        )
        preflight = create_or_reuse_review_thread(
            db,
            prepared=prepared,
            request=request,
            actor=_actor(),
            settings=settings,
        )
    store = ExtractionCallStore(
        session_factory=factory,
        settings=settings,
        prepared_plan_set=plans,
    )
    return factory, store, preflight.thread.thread_id, plan


def test_postgresql_store_concurrent_claim_marker_and_atomic_empty_completion(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)

    def claim():
        return store.claim_or_replay(
            thread_id,
            plan,
            actor_subject_id='task3-owner',
            allowed_permission_levels=('internal',),
            validation_obligation=Decimal('0.097728'),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        contexts = list(pool.map(lambda _: claim(), range(2)))
    assert contexts[0] == contexts[1]
    grant = store.mark_attempt_started(contexts[0])
    assert grant.authoritative_lease_expires_at > datetime.now(UTC)
    store.complete(
        contexts[0],
        result={
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        },
        usage=ProviderUsage(10, 10),
    )
    with factory() as db:
        calls = tuple(db.scalars(select(AutoReviewExtractionCall)).all())
        runs = tuple(db.scalars(select(AgentRun)).all())
        assert len(calls) == len(runs) == 1
        assert calls[0].status == 'completed'
        assert calls[0].result_kind == 'no_candidate'
        assert calls[0].result_candidate_count == 0
        assert len(calls[0].result_candidate_set_hmac or '') == 64
        assert runs[0].status == 'complete'


def test_postgresql_migration_cycles_task2_to_hardened_head(
    postgres_runtime,
) -> None:
    engine, _ = postgres_runtime
    config = Config('alembic.ini')
    command.downgrade(config, '7c5a2e9f4b10')
    command.upgrade(config, 'head')
    with engine.connect() as connection:
        definition = connection.scalar(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname='ck_auto_review_extraction_calls_terminal_charge'"
            )
        )
        revision = connection.scalar(text('SELECT version_num FROM alembic_version'))
    assert definition is not None
    assert 'provider_attempt_count = 1' in definition
    assert 'attempt_started_at IS NOT NULL' in definition
    assert revision == '9d7f3a1c6e20'


def test_postgresql_store_cancel_before_marker_is_zero_charge_terminal(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    context = store.claim_or_replay(
        thread_id,
        plan,
        actor_subject_id='task3-owner',
        allowed_permission_levels=('internal',),
    )
    with factory() as db, db.begin():
        db.execute(
            text(
                'UPDATE agent_workflow_threads SET cancelled_at=clock_timestamp(), '
                "cancelled_by_subject_id='task3-owner' WHERE thread_id=:thread_id"
            ),
            {'thread_id': thread_id},
        )
    with pytest.raises(ExtractionCallStateError, match='cancel'):
        store.mark_attempt_started(context)
    with factory() as db:
        call = db.scalar(select(AutoReviewExtractionCall))
        assert call is not None
        assert call.status == 'failed'
        assert call.provider_attempt_count == 0
        assert call.charged_cost_usd == Decimal('0.000000')


def test_postgresql_store_post_attempt_expiry_recovers_once_at_full_reserve(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    context = store.claim_or_replay(
        thread_id,
        plan,
        actor_subject_id='task3-owner',
        allowed_permission_levels=('internal',),
    )
    store.mark_attempt_started(context)
    with factory() as db, db.begin():
        db.execute(
            text(
                "UPDATE auto_review_extraction_calls SET lease_expires_at="
                "clock_timestamp() - interval '1 second' WHERE workflow_thread_id=:id"
            ),
            {'id': thread_id},
        )
    assert store.recover_expired(context) is None
    with factory() as db:
        call = db.scalar(select(AutoReviewExtractionCall))
        assert call is not None
        assert call.status == 'failed'
        assert call.provider_attempt_count == 1
        assert call.charged_cost_usd == call.reserved_cost_usd
        with pytest.raises(ExtractionCallStateError):
            store.complete(
                context,
                result={
                    'result_kind': 'no_candidate',
                    'candidate': None,
                    'no_candidate_reason': 'no_relevant_evidence',
                },
                usage=ProviderUsage(1, 1),
            )


def test_postgresql_store_rejects_owner_and_permission_before_reserve(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    with pytest.raises(ExtractionCallStateError, match='owner'):
        store.claim_or_replay(
            thread_id,
            plan,
            actor_subject_id='another-owner',
            allowed_permission_levels=('internal',),
        )
    with pytest.raises(ExtractionCallStateError, match='permission'):
        store.claim_or_replay(
            thread_id,
            plan,
            actor_subject_id='task3-owner',
            allowed_permission_levels=('public',),
        )
    with factory() as db:
        assert db.scalar(select(AutoReviewExtractionCall)) is None


@pytest.mark.parametrize('drift_kind', ['runtime_key', 'source_permission'])
def test_postgresql_store_e2_drift_terminalizes_zero_charge(
    postgres_runtime,
    drift_kind: str,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    context = store.claim_or_replay(
        thread_id,
        plan,
        actor_subject_id='task3-owner',
        allowed_permission_levels=('internal',),
    )
    with factory() as db, db.begin():
        if drift_kind == 'runtime_key':
            db.execute(
                text(
                    'UPDATE auto_review_runtime_key_states SET '
                    "fingerprint_key_version='task3-key-v2'"
                )
            )
        else:
            db.execute(text("UPDATE sources SET permission_level='restricted'"))
    with pytest.raises(ExtractionCallStateError):
        store.mark_attempt_started(context)
    with factory() as db:
        call = db.scalar(select(AutoReviewExtractionCall))
        assert call is not None
        assert call.status == 'failed'
        assert call.provider_attempt_count == 0
        assert call.charged_cost_usd == Decimal('0.000000')


def test_postgresql_store_cancel_after_marker_discards_and_charges_once(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    context = store.claim_or_replay(
        thread_id,
        plan,
        actor_subject_id='task3-owner',
        allowed_permission_levels=('internal',),
    )
    store.mark_attempt_started(context)
    with factory() as db, db.begin():
        db.execute(
            text(
                'UPDATE agent_workflow_threads SET cancelled_at=clock_timestamp(), '
                "cancelled_by_subject_id='task3-owner' WHERE thread_id=:thread_id"
            ),
            {'thread_id': thread_id},
        )
    store.complete(
        context,
        result={
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        },
        usage=ProviderUsage(11, 7),
    )
    with factory() as db:
        call = db.scalar(select(AutoReviewExtractionCall))
        assert call is not None
        assert call.status == 'failed'
        assert call.provider_attempt_count == 1
        assert call.charged_input_tokens == 11
        assert call.charged_output_tokens == 7
        assert db.scalar(select(ReviewItem)) is None
    with pytest.raises(ExtractionCallStateError):
        store.complete(
            context,
            result={
                'result_kind': 'no_candidate',
                'candidate': None,
                'no_candidate_reason': 'no_relevant_evidence',
            },
            usage=ProviderUsage(11, 7),
        )


def test_postgresql_store_known_overrun_opens_breaker_and_fails(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    context = store.claim_or_replay(
        thread_id,
        plan,
        actor_subject_id='task3-owner',
        allowed_permission_levels=('internal',),
    )
    store.mark_attempt_started(context)
    store.complete(
        context,
        result={
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        },
        usage=ProviderUsage(plan.max_input_tokens + 1, plan.max_output_tokens),
    )
    with factory() as db:
        call = db.scalar(select(AutoReviewExtractionCall))
        safety = db.scalar(
            select(AutoReviewProviderSafetyState).where(
                AutoReviewProviderSafetyState.purpose == 'extraction'
            )
        )
        assert call is not None and safety is not None
        assert call.status == 'failed'
        assert call.budget_overrun is True
        assert safety.breaker_open is True
        assert safety.overrun_count == 1


def test_postgresql_store_replay_rederives_empty_terminal_marker(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    context = store.claim_or_replay(
        thread_id,
        plan,
        actor_subject_id='task3-owner',
        allowed_permission_levels=('internal',),
    )
    store.mark_attempt_started(context)
    store.complete(
        context,
        result={
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        },
        usage=ProviderUsage(1, 1),
    )
    assert (
        store.claim_or_replay(
            thread_id,
            plan,
            actor_subject_id='task3-owner',
            allowed_permission_levels=('internal',),
        ).lease_token
        == context.lease_token
    )
    with factory() as db, db.begin():
        db.execute(
            text(
                "UPDATE auto_review_extraction_calls SET "
                "result_candidate_set_hmac=:corrupt"
            ),
            {'corrupt': 'd' * 64},
        )
    with pytest.raises(ExtractionCallStateError, match='corrupt'):
        store.claim_or_replay(
            thread_id,
            plan,
            actor_subject_id='task3-owner',
            allowed_permission_levels=('internal',),
        )


def test_postgresql_candidate_callback_requires_committed_exact_child_proof(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    factory, store, thread_id, plan = _seed_runtime(engine, database_url)
    context = store.claim_or_replay(
        thread_id,
        plan,
        actor_subject_id='task3-owner',
        allowed_permission_levels=('internal',),
    )
    store.mark_attempt_started(context)

    def incomplete_writer(db, run, _plan, _parsed):
        db.add(
            ReviewItem(
                item_type='history_event',
                payload={'bounded': True},
                source_links=['https://private.example/task3'],
                source_snippets=['검토 가능한 근거 문장'],
                confidence_score=0.98,
                permission_level='internal',
                status='pending_review',
                workflow_thread_id=thread_id,
                candidate_key='e' * 64,
                agent_run_id=run.id,
                candidate_contract_version='c5-v1',
            )
        )
        return 'e' * 64, 'f' * 64

    store.complete(
        context,
        result={
            'result_kind': 'candidate',
            'candidate': {
                'item_type': 'history_event',
                'title': '제목',
                'summary': '요약',
                'reason': '직접 근거',
                'confidence_score': '0.9800',
                'uncertainty_reason': None,
                'field_evidence_bindings': [
                    {'field_key': 'title', 'evidence_slot_id': 'S01'},
                    {'field_key': 'summary', 'evidence_slot_id': 'S02'},
                    {'field_key': 'reason', 'evidence_slot_id': 'S03'},
                ],
            },
            'no_candidate_reason': None,
        },
        usage=ProviderUsage(12, 12),
        candidate_writer=incomplete_writer,
    )
    with factory() as db:
        call = db.scalar(select(AutoReviewExtractionCall))
        run = db.scalar(select(AgentRun))
        assert call is not None and run is not None
        assert call.status == run.status == 'failed'
        assert call.result_kind is None
        assert run.metadata_['failure_reason_code'] == 'evidence_binding_mismatch'
        assert db.scalar(select(ReviewItem)) is None
