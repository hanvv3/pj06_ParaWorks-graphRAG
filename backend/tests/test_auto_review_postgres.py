import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

import backend.app.agent_runtime.auto_review_validation_store as validation_store_module
import backend.app.models  # noqa: F401
from backend.app.agent_runtime.auto_review_validation_store import (
    AutoReviewValidationIdentity,
    AutoReviewValidationStore,
    ValidationAttemptAdmission,
    ValidationBatchClaimRequest,
    ValidationBatchCompletion,
    ValidationBatchFailure,
    ValidationCandidateClaim,
    ValidationCandidateCompletion,
    ValidationLockedContext,
    ValidationSourceState,
    ValidationStoreError,
    derive_validation_batch_fingerprint,
    derive_validation_key,
    derive_validation_owner_permission_hmac,
    derive_validation_source_state_hmac,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models import (
    AutoReviewProviderSafetyEvent,
    AutoReviewProviderSafetyState,
    AutoReviewRuntimeKeyState,
    AutoReviewValidation,
    AutoReviewValidationCall,
)
from backend.app.schemas.auto_review import CandidateValidationResult


@pytest.fixture
def postgres_runtime(monkeypatch: pytest.MonkeyPatch):
    original_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not original_url:
        pytest.fail('PARAWORKS_TEST_POSTGRES_URL is required for Task 9 tests')
    admin = create_engine(original_url)
    if inspect(admin).get_table_names():
        pytest.fail('Task 9 PostgreSQL URL must name a fresh empty database')
    schema = f'task9_{uuid4().hex}'
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
        _env_file=None,
        paraworks_database_url=database_url,
        database_url=database_url,
        agent_runtime_fingerprint_secret='task9-secret-at-least-32-bytes-long',
        agent_runtime_fingerprint_key_version='task9-key-v1',
    )


def _store(factory, settings: Settings, **kwargs) -> AutoReviewValidationStore:
    resolver = kwargs.pop(
        'current_permission_resolver',
        lambda subject_id: ('internal',) if subject_id == 'task9-owner' else (),
    )
    return AutoReviewValidationStore(
        session_factory=factory,
        settings=settings,
        current_permission_resolver=resolver,
        **kwargs,
    )


def _identity(*, workflow_hmac: str = '1' * 64) -> AutoReviewValidationIdentity:
    return AutoReviewValidationIdentity(
        workflow_execution_identity_hmac=workflow_hmac,
        security_scope_hmac='2' * 64,
        candidate_key='3' * 64,
        evidence_version_hash='4' * 64,
        normalized_claim_fingerprint='5' * 64,
        candidate_generation_fingerprint='6' * 64,
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v2',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version='auto-review-policy:v1',
        fingerprint_key_version='task9-key-v1',
        fingerprint_key_material_verifier='7' * 64,
        token_estimator_version='openai-o200k-chat:v1',
        tokenizer_encoding='o200k_base',
        max_input_tokens=6000,
        max_output_tokens=3072,
        max_candidates_per_batch=4,
        max_batches_per_workflow=2,
        max_candidates_per_workflow=5,
        max_provider_attempts=1,
        provider_timeout_seconds=60,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=120,
        provider_commit_grace_seconds=30,
        input_cost_per_1m_tokens=Decimal('2.000000'),
        output_cost_per_1m_tokens=Decimal('12.000000'),
        cost_policy_version='auto-review-cost:v1',
        provider_safety_state_version=1,
        rollout_control_epoch=2,
        authorized_percentage_at_launch=10,
        rollout_authorization_generation=3,
        confirmed_extraction_cost_ceiling_usd=Decimal('0.083580'),
        confirmed_validation_cost_ceiling_usd=Decimal('0.097728'),
        confirmed_total_cost_ceiling_usd=Decimal('0.181308'),
    )


def _request(
    settings: Settings,
    *,
    workflow_thread_id: str,
    review_item_ids: tuple[int, ...],
    candidate_seed: int = 1,
    prepared_content_hmac: str = 'a' * 64,
    validation_ceiling: Decimal = Decimal('0.097728'),
) -> ValidationBatchClaimRequest:
    identity = replace(
        _identity(workflow_hmac=(workflow_thread_id[-1] * 64)),
        confirmed_validation_cost_ceiling_usd=validation_ceiling,
    )
    candidates = tuple(
        ValidationCandidateClaim(
            review_item_id=review_item_id,
            identity=replace(
                identity, candidate_key=f'{candidate_seed + index - 1:x}' * 64
            ),
            validation_key=derive_validation_key(
                replace(
                    identity,
                    candidate_key=f'{candidate_seed + index - 1:x}' * 64,
                ),
                settings=settings,
            ),
        )
        for index, review_item_id in enumerate(review_item_ids, start=1)
    )
    values = {
        'workflow_thread_id': workflow_thread_id,
        'candidates': candidates,
        'prepared_content_hmac': prepared_content_hmac,
        'serialized_character_count': 900,
        'framed_input_tokens': 1200,
        'reserved_input_tokens': 1200,
        'reserved_output_tokens': 3072,
        'reserved_cost_usd': Decimal('0.039264'),
    }
    return ValidationBatchClaimRequest(
        batch_fingerprint=derive_validation_batch_fingerprint(
            settings=settings, **values
        ),
        max_provider_attempts=1,
        **values,
    )


def _seed_review_items(engine, requests: tuple[ValidationBatchClaimRequest, ...]) -> None:
    seeded_workflows: dict[str, int] = {}
    with engine.begin() as connection:
        identity = requests[0].candidates[0].identity
        connection.execute(
            text(
                'INSERT INTO auto_review_runtime_key_states '
                '(component, fingerprint_key_version, '
                'fingerprint_key_material_verifier, generation, ready, updated_at) '
                "VALUES ('auto_review_trust_promotion', :key_version, :verifier, "
                '1, true, CURRENT_TIMESTAMP)'
            ),
            {
                'key_version': identity.fingerprint_key_version,
                'verifier': identity.fingerprint_key_material_verifier,
            },
        )
        safety_id = connection.scalar(
            text(
                'INSERT INTO auto_review_provider_safety_states '
                '(purpose, provider, model, reasoning_effort, state_version, '
                'authorized_cost_policy_version, token_estimator_version, '
                'tokenizer_encoding, reply_priming_tokens, framing_safety_tokens, '
                'input_usd_per_1m, output_usd_per_1m, breaker_open, overrun_count, '
                'authorized_at, last_event_sequence, created_at, updated_at) VALUES '
                '(:purpose, :provider, :model, :reasoning, 1, :cost_policy, '
                ':estimator, :encoding, 16, 512, :input_price, :output_price, '
                'false, 0, CURRENT_TIMESTAMP, 0, CURRENT_TIMESTAMP, '
                'CURRENT_TIMESTAMP) RETURNING id'
            ),
            {
                'purpose': 'validation',
                'provider': identity.validator_provider,
                'model': identity.validator_model,
                'reasoning': identity.reasoning_effort,
                'cost_policy': identity.cost_policy_version,
                'estimator': identity.token_estimator_version,
                'encoding': identity.tokenizer_encoding,
                'input_price': identity.input_cost_per_1m_tokens,
                'output_price': identity.output_cost_per_1m_tokens,
            },
        )
        event_id = connection.scalar(
            text(
                'INSERT INTO auto_review_provider_safety_events '
                '(provider_safety_state_id, purpose, provider, model, '
                'reasoning_effort, event_sequence, event_kind, '
                'prior_state_version, new_state_version, cost_policy_version, '
                'token_estimator_version, tokenizer_encoding, '
                'reply_priming_tokens, framing_safety_tokens, input_usd_per_1m, '
                'output_usd_per_1m, prior_breaker_open, new_breaker_open, '
                'actor_subject_hmac, fingerprint_key_version, '
                'fingerprint_key_material_verifier, created_at) VALUES '
                '(:state_id, :purpose, :provider, :model, :reasoning, 1, '
                ':kind, 0, 1, :cost_policy, :estimator, :encoding, 16, 512, '
                ':input_price, :output_price, false, false, :actor, '
                ':key_version, :verifier, CURRENT_TIMESTAMP) RETURNING id'
            ),
            {
                'state_id': safety_id,
                'purpose': 'validation',
                'provider': identity.validator_provider,
                'model': identity.validator_model,
                'reasoning': identity.reasoning_effort,
                'kind': 'initial_authorized',
                'cost_policy': identity.cost_policy_version,
                'estimator': identity.token_estimator_version,
                'encoding': identity.tokenizer_encoding,
                'input_price': identity.input_cost_per_1m_tokens,
                'output_price': identity.output_cost_per_1m_tokens,
                'actor': 'b' * 64,
                'key_version': identity.fingerprint_key_version,
                'verifier': identity.fingerprint_key_material_verifier,
            },
        )
        connection.execute(
            text(
                'UPDATE auto_review_provider_safety_states SET '
                'last_event_sequence=1, last_event_id=:event_id WHERE id=:state_id'
            ),
            {'event_id': event_id, 'state_id': safety_id},
        )
        for run_id, request in enumerate(requests, start=1):
            identity = request.candidates[0].identity
            if request.workflow_thread_id not in seeded_workflows:
                connection.execute(
                    text(
                        'INSERT INTO agent_workflow_threads '
                        '(thread_id, workflow_name, graph_version, '
                        'checkpoint_thread_id, checkpoint_store, owner_subject_id, '
                        'security_scope_id, input_hash, evidence_version_hash, '
                        'status, state_version, created_at, updated_at) VALUES '
                        '(:workflow, :name, :graph, :checkpoint, :store, :owner, '
                        ':scope, :input_hash, :evidence_hash, :status, 0, '
                        'clock_timestamp(), clock_timestamp())'
                    ),
                    {
                        'workflow': request.workflow_thread_id,
                        'name': 'company_memory_review',
                        'graph': 'company-memory-review-v2.1-auto-review',
                        'checkpoint': f'checkpoint:{request.workflow_thread_id}',
                        'store': 'postgres',
                        'owner': 'task9-owner',
                        'scope': 'scope-task9',
                        'input_hash': 'f' * 64,
                        'evidence_hash': identity.evidence_version_hash,
                        'status': 'drafting',
                    },
                )
                connection.execute(
                    text(
                    'INSERT INTO agent_workflow_requests '
                    '(workflow_thread_id, input_schema_version, request_kind, '
                    'agent_names, selection_policy_version, input_hash, '
                    'fingerprint_key_version, fingerprint_key_material_verifier, '
                    'auto_review_mode, auto_review_validator_provider, '
                    'auto_review_validator_model, auto_review_reasoning_effort, '
                    'auto_review_validator_prompt_version, '
                    'auto_review_validator_output_contract_version, '
                    'auto_review_policy_version, auto_review_cost_policy_version, '
                    'auto_review_token_estimator_version, '
                    'auto_review_tokenizer_encoding, auto_review_reply_priming_tokens, '
                    'auto_review_framing_safety_tokens, auto_review_max_input_tokens, '
                    'auto_review_max_output_tokens, '
                    'auto_review_max_candidates_per_batch, '
                    'auto_review_max_batches_per_workflow, '
                    'auto_review_max_candidates_per_workflow, '
                    'auto_review_max_provider_attempts, '
                    'auto_review_provider_timeout_seconds, '
                    'auto_review_provider_send_start_window_seconds, '
                    'auto_review_provider_attempt_lease_seconds, '
                    'auto_review_provider_commit_grace_seconds, '
                    'auto_review_validator_input_usd_per_1m, '
                    'auto_review_validator_output_usd_per_1m, '
                    'authorized_percentage_at_launch, '
                    'rollout_authorization_generation, '
                    'validation_provider_safety_state_version, '
                    'rollout_control_epoch, confirmed_extraction_cost_ceiling_usd, '
                    'confirmed_validation_cost_ceiling_usd, '
                    'confirmed_total_cost_ceiling_usd, auto_review_budget_limit_usd) '
                    'VALUES (:workflow, :schema, :kind, :agents, :selection, '
                    ':input_hash, :key_version, :verifier, :mode, :provider, '
                    ':model, :reasoning, :prompt, :output_contract, :policy, '
                    ':cost_policy, :estimator, :encoding, 16, 512, :max_input, '
                    ':max_output, :max_batch_candidates, :max_batches, '
                    ':max_workflow_candidates, 1, :timeout, :send_window, '
                    ':lease_seconds, :grace_seconds, :input_price, :output_price, '
                    ':percentage, :generation, :safety_version, :epoch, '
                    ':extraction_ceiling, :validation_ceiling, :total_ceiling, '
                    ':total_ceiling)'
                    ),
                    {
                    'workflow': request.workflow_thread_id,
                    'schema': 'review-v2.1:v1',
                    'kind': 'review_source_versions',
                    'agents': '[]',
                    'selection': 'review-workflow-batch:v2.1',
                    'input_hash': 'f' * 64,
                    'key_version': identity.fingerprint_key_version,
                    'verifier': identity.fingerprint_key_material_verifier,
                    'mode': 'shadow',
                    'provider': identity.validator_provider,
                    'model': identity.validator_model,
                    'reasoning': identity.reasoning_effort,
                    'prompt': identity.validator_prompt_version,
                    'output_contract': identity.validator_output_contract_version,
                    'policy': identity.policy_version,
                    'cost_policy': identity.cost_policy_version,
                    'estimator': identity.token_estimator_version,
                    'encoding': identity.tokenizer_encoding,
                    'max_input': identity.max_input_tokens,
                    'max_output': identity.max_output_tokens,
                    'max_batch_candidates': identity.max_candidates_per_batch,
                    'max_batches': identity.max_batches_per_workflow,
                    'max_workflow_candidates': identity.max_candidates_per_workflow,
                    'timeout': identity.provider_timeout_seconds,
                    'send_window': identity.provider_send_start_window_seconds,
                    'lease_seconds': identity.provider_attempt_lease_seconds,
                    'grace_seconds': identity.provider_commit_grace_seconds,
                    'input_price': identity.input_cost_per_1m_tokens,
                    'output_price': identity.output_cost_per_1m_tokens,
                    'percentage': identity.authorized_percentage_at_launch,
                    'generation': identity.rollout_authorization_generation,
                    'safety_version': identity.provider_safety_state_version,
                    'epoch': identity.rollout_control_epoch,
                    'extraction_ceiling': (
                        identity.confirmed_extraction_cost_ceiling_usd
                    ),
                    'validation_ceiling': (
                        identity.confirmed_validation_cost_ceiling_usd
                    ),
                    'total_ceiling': identity.confirmed_total_cost_ceiling_usd,
                    },
                )
                workflow_ref_id = 1000 + run_id
                seeded_workflows[request.workflow_thread_id] = workflow_ref_id
                connection.execute(
                    text(
                        'INSERT INTO sources '
                        '(id, source_type, source_id, source_url, title, '
                        'permission_level, raw_metadata, '
                        'server_content_signature_schema, '
                        'server_content_signature, created_at) VALUES '
                        '(:id, :kind, :source_id, :url, :title, :permission, '
                        ':metadata, :signature_schema, :signature, '
                        'clock_timestamp())'
                    ),
                    {
                        'id': workflow_ref_id,
                        'kind': 'gmail',
                        'source_id': f'gmail:task9:{workflow_ref_id}',
                        'url': 'https://private.example/task9',
                        'title': 'bounded source',
                        'permission': 'internal',
                        'metadata': '{}',
                        'signature_schema': 'server-source-content:v1',
                        'signature': '8' * 64,
                    },
                )
                connection.execute(
                    text(
                    'INSERT INTO agent_workflow_evidence_refs '
                    '(id, workflow_thread_id, ordinal, canonical_source_type, '
                    'canonical_table, canonical_row_id, document_version_id, '
                    'external_revision, content_signature, '
                    'permission_level_snapshot, content_fingerprint) VALUES '
                    '(:id, :workflow, 1, :source_type, :source_table, :row_id, '
                    'NULL, :revision, :signature, :permission, :fingerprint)'
                    ),
                    {
                    'id': workflow_ref_id,
                    'workflow': request.workflow_thread_id,
                    'source_type': 'gmail',
                    'source_table': 'sources',
                    'row_id': workflow_ref_id,
                    'revision': 'v1',
                    'signature': '8' * 64,
                    'permission': 'internal',
                    'fingerprint': '9' * 64,
                    },
                )
            workflow_ref_id = seeded_workflows[request.workflow_thread_id]
            connection.execute(
                text(
                    'INSERT INTO agent_runs '
                    '(id, agent_name, prompt_version, status, source_window, cache_key, '
                    'model_name, input_tokens, output_tokens, total_tokens, '
                    'estimated_cost_usd, permission_level, metadata, '
                    'workflow_thread_id, effect_key, started_at) VALUES '
                    '(:id, :agent, :prompt, :status, :window, :cache, :model, '
                    '0, 0, 0, 0, :permission, :metadata, :workflow, :effect, '
                    'clock_timestamp())'
                ),
                {
                    'id': run_id,
                    'agent': 'mail_document_agent',
                    'prompt': 'task9:v1',
                    'status': 'complete',
                    'window': 'task9',
                    'cache': f'cache-{run_id}',
                    'model': 'fake',
                    'permission': 'internal',
                    'metadata': '{}',
                    'workflow': request.workflow_thread_id,
                    'effect': f'effect-{run_id}',
                },
            )
            for candidate in request.candidates:
                connection.execute(
                    text(
                        'INSERT INTO review_items '
                        '(id, item_type, payload, source_links, source_snippets, '
                        'confidence_score, permission_level, status, '
                        'workflow_thread_id, candidate_key, agent_run_id, '
                        'candidate_contract_version, created_at) VALUES '
                        '(:id, :kind, :payload, :links, :snippets, 0.99, '
                        ':permission, :status, :workflow, :candidate_key, :run_id, '
                        ':contract, clock_timestamp())'
                    ),
                    {
                        'id': candidate.review_item_id,
                        'kind': 'timeline_event',
                        'payload': '{}',
                        'links': '[]',
                        'snippets': '[]',
                        'permission': 'internal',
                        'status': 'pending_review',
                        'workflow': request.workflow_thread_id,
                        'candidate_key': candidate.identity.candidate_key,
                        'run_id': run_id,
                        'contract': 'c5-v1',
                    },
                )
                connection.execute(
                    text(
                        'INSERT INTO review_item_evidence_refs '
                        '(review_item_id, workflow_thread_id, '
                        'workflow_evidence_ref_id, candidate_slot_ordinal, '
                        'message_content_fingerprint, fingerprint_key_version, '
                        'fingerprint_key_material_verifier, created_at) VALUES '
                        '(:review_item, :workflow, :workflow_ref, 1, '
                        ':message_hmac, :key_version, :verifier, clock_timestamp())'
                    ),
                    {
                        'review_item': candidate.review_item_id,
                        'workflow': request.workflow_thread_id,
                        'workflow_ref': workflow_ref_id,
                        'message_hmac': 'a' * 64,
                        'key_version': 'task9-key-v1',
                        'verifier': '7' * 64,
                    },
                )


def _admission(request: ValidationBatchClaimRequest, claim) -> ValidationAttemptAdmission:
    identity = request.candidates[0].identity
    settings = _settings('sqlite:///unused-task9.db')
    return ValidationAttemptAdmission(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token,
        prepared_content_hmac=request.prepared_content_hmac,
        serialized_character_count=request.serialized_character_count,
        framed_input_tokens=request.framed_input_tokens,
        max_output_tokens=identity.max_output_tokens,
        recomputed_reserved_cost_usd=request.reserved_cost_usd,
        expected_owner_permission_hmac=derive_validation_owner_permission_hmac(
            owner_subject_id='task9-owner',
            permission_levels=('internal',),
            settings=settings,
        ),
        expected_source_state_hmac=derive_validation_source_state_hmac(
            workflow_thread_id=request.workflow_thread_id,
            sources=(
                ValidationSourceState(
                    canonical_row_id=1001,
                    canonical_source_type='gmail',
                    content_signature='8' * 64,
                    content_fingerprint='9' * 64,
                    permission_level='internal',
                ),
            ),
            settings=settings,
        ),
        fingerprint_key_version=identity.fingerprint_key_version,
        fingerprint_key_material_verifier=identity.fingerprint_key_material_verifier,
        token_estimator_version=identity.token_estimator_version,
        cost_policy_version=identity.cost_policy_version,
        expected_effective_mode='shadow',
        provider_timeout_seconds=identity.provider_timeout_seconds,
        provider_send_start_window_seconds=(
            identity.provider_send_start_window_seconds
        ),
        provider_attempt_lease_seconds=identity.provider_attempt_lease_seconds,
        provider_commit_grace_seconds=identity.provider_commit_grace_seconds,
    )


def _validation_result(slot: str) -> CandidateValidationResult:
    return CandidateValidationResult.model_validate(
        {
            'candidate_slot_id': slot,
            'claim_results': [
                {
                    'field_key': 'title',
                    'verdict': 'supported',
                    'claim_scope': 'direct_fact',
                    'entailment_score': '0.9900',
                    'evidence_slot_ids': ['E01'],
                },
                {
                    'field_key': 'result_summary',
                    'verdict': 'supported',
                    'claim_scope': 'direct_fact',
                    'entailment_score': '0.9900',
                    'evidence_slot_ids': ['E01'],
                },
            ],
            'uncertainty_codes': [],
            'conflict_codes': [],
        }
    )


def test_concurrent_claims_produce_one_canonical_result(postgres_runtime) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-concurrency-1',
        review_item_ids=(901, 902),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    barrier = Barrier(2)

    def claim_once():
        store = _store(factory, settings)
        barrier.wait()
        return store.claim_or_replay(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: claim_once(), range(2)))

    assert sorted(result.disposition for result in results) == ['busy', 'claimed']
    assert len({result.validation_call_id for result in results}) == 1
    with factory() as db:
        assert len(tuple(db.scalars(select(AutoReviewValidationCall)))) == 1
        assert len(tuple(db.scalars(select(AutoReviewValidation)))) == 2


def test_workflow_budget_reservation_is_atomic_across_concurrent_batches(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    workflow = 'workflow-task9-budget-3'
    first = _request(
        settings,
        workflow_thread_id=workflow,
        review_item_ids=(921,),
        candidate_seed=1,
        prepared_content_hmac='a' * 64,
        validation_ceiling=Decimal('0.050000'),
    )
    second = _request(
        settings,
        workflow_thread_id=workflow,
        review_item_ids=(922,),
        candidate_seed=2,
        prepared_content_hmac='b' * 64,
        validation_ceiling=Decimal('0.050000'),
    )
    _seed_review_items(engine, (first, second))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    barrier = Barrier(2)

    def claim_once(request):
        store = _store(factory, settings)
        barrier.wait()
        try:
            return store.claim_or_replay(request).disposition
        except ValidationStoreError:
            return 'budget_rejected'

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(claim_once, (first, second)))

    assert sorted(outcomes) == ['budget_rejected', 'claimed']
    with factory() as db:
        calls = tuple(db.scalars(select(AutoReviewValidationCall)))
        assert len(calls) == 1
        assert Decimal(calls[0].reserved_cost_usd) == Decimal('0.039264')


def test_restart_reuses_completed_validation_and_cost_metadata(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-restart-2',
        review_item_ids=(911, 912),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    owner = _store(factory, settings)
    claim = owner.claim_or_replay(request)
    grant = owner.mark_attempt_started(_admission(request, claim))

    class Invocation:
        prepared_content_hmac = request.prepared_content_hmac
        character_count = request.serialized_character_count
        framed_input_tokens = request.framed_input_tokens
        max_output_tokens = 3072

    owner.dispatch_prepared_validation(
        invocation=Invocation(),
        provider=lambda invocation, **kwargs: None,
        grant=grant,
    )
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token,
    )
    projections = owner.complete(
        context,
        ValidationBatchCompletion(
            validation_call_id=claim.validation_call_id,
            workflow_thread_id=request.workflow_thread_id,
            batch_fingerprint=request.batch_fingerprint,
            lease_token=claim.lease_token,
            candidates=tuple(
                ValidationCandidateCompletion(
                    review_item_id=candidate.review_item_id,
                    validation_key=candidate.validation_key,
                    result=_validation_result(f'C{index:02d}'),
                    policy_decision='auto_approve',
                    policy_reason_codes=('eligible',),
                )
                for index, candidate in enumerate(
                    sorted(request.candidates, key=lambda item: item.validation_key),
                    start=1,
                )
            ),
            input_tokens=101,
            output_tokens=51,
        ),
    )
    restarted = _store(factory, settings)
    replay = restarted.claim_or_replay(request)

    assert replay.disposition == 'replayed'
    assert replay.terminal_status == 'completed'
    assert all(item.cache_hit for item in replay.completed)
    assert sum(item.input_tokens for item in projections) == 101
    assert sum(item.output_tokens for item in projections) == 51
    assert sum(item.estimated_cost_usd for item in projections) == Decimal(
        '0.000814'
    )
    assert [item.result for item in replay.completed] == [
        item.result for item in projections
    ]


def test_persisted_validation_contains_no_raw_evidence_url_or_exception(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-raw-hygiene',
        review_item_ids=(915,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _store(factory, settings)
    claim = store.claim_or_replay(request)
    grant = store.mark_attempt_started(_admission(request, claim))
    raw_marker = 'https://private.example/raw-do-not-persist provider-secret-detail'

    class Invocation:
        prepared_content_hmac = request.prepared_content_hmac
        character_count = request.serialized_character_count
        framed_input_tokens = request.framed_input_tokens
        max_output_tokens = 3072

    def provider(*_args, **_kwargs):
        raise RuntimeError(raw_marker)

    with pytest.raises(RuntimeError, match='raw-do-not-persist'):
        store.dispatch_prepared_validation(
            invocation=Invocation(),
            provider=provider,
            grant=grant,
        )
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token,
    )
    store.fail(
        context,
        ValidationBatchFailure(
            validation_call_id=claim.validation_call_id,
            workflow_thread_id=request.workflow_thread_id,
            batch_fingerprint=request.batch_fingerprint,
            lease_token=claim.lease_token,
            reason_code='validator_unavailable',
            usage_known=False,
            input_tokens=None,
            output_tokens=None,
        ),
    )

    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        children = tuple(
            db.scalars(
                select(AutoReviewValidation).where(
                    AutoReviewValidation.validation_call_id
                    == claim.validation_call_id
                )
            )
        )
        assert call is not None
        persisted = json.dumps(
            {
                'call': {
                    column.name: getattr(call, column.name)
                    for column in AutoReviewValidationCall.__table__.columns
                },
                'children': [
                    {
                        column.name: getattr(child, column.name)
                        for column in AutoReviewValidation.__table__.columns
                    }
                    for child in children
                ],
            },
            default=str,
            sort_keys=True,
        )
    assert raw_marker not in persisted
    assert 'https://private.example/task9' not in persisted


def test_cancel_before_attempt_is_zero_charge_and_releases_reserve(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    workflow = 'workflow-task9-cancel-4'
    first = _request(
        settings,
        workflow_thread_id=workflow,
        review_item_ids=(931,),
        candidate_seed=1,
        validation_ceiling=Decimal('0.050000'),
    )
    second = _request(
        settings,
        workflow_thread_id=workflow,
        review_item_ids=(932,),
        candidate_seed=2,
        prepared_content_hmac='b' * 64,
        validation_ceiling=Decimal('0.050000'),
    )
    _seed_review_items(engine, (first, second))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _store(factory, settings)
    claim = store.claim_or_replay(first)
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=workflow,
        batch_fingerprint=first.batch_fingerprint,
        lease_token=claim.lease_token,
    )

    store.cancel(context)
    replay = store.claim_or_replay(first)
    next_claim = store.claim_or_replay(second)

    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'post_validation_drift'
    assert next_claim.disposition == 'claimed'
    with factory() as db:
        first_call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        assert first_call is not None
        assert first_call.charged_input_tokens == 0
        assert first_call.charged_output_tokens == 0
        assert Decimal(first_call.charged_cost_usd) == Decimal('0')


def test_expired_started_attempt_never_retries_and_keeps_reserve_charged(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-expired-5',
        review_item_ids=(941,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    now = [datetime(2026, 8, 30, 12, 0, tzinfo=UTC)]
    store = _store(
        factory,
        settings,
        db_clock=lambda db: now[0],
    )
    claim = store.claim_or_replay(request)
    store.mark_attempt_started(_admission(request, claim))
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token,
    )
    now[0] += timedelta(seconds=121)

    store.recover_expired(context)
    replay = store.claim_or_replay(request)

    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'validator_unavailable'
    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        assert call is not None
        assert call.provider_attempt_count == 1
        assert Decimal(call.charged_cost_usd) == request.reserved_cost_usd


def test_owner_permission_loss_before_marker_is_zero_call_zero_charge(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-permission-6',
        review_item_ids=(951,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    permission_levels = ['internal']
    store = _store(
        factory,
        settings,
        current_permission_resolver=lambda subject_id: tuple(permission_levels),
    )
    claim = store.claim_or_replay(request)
    permission_levels.clear()

    with pytest.raises(ValidationStoreError, match='guard changed'):
        store.mark_attempt_started(_admission(request, claim))
    replay = store.claim_or_replay(request)

    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'post_validation_drift'
    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        assert call is not None
        assert call.provider_attempt_count == 0
        assert Decimal(call.charged_cost_usd) == Decimal('0')


def test_runtime_key_drift_before_marker_is_zero_call_zero_charge(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-runtime-drift',
        review_item_ids=(956,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _store(factory, settings)
    claim = store.claim_or_replay(request)

    with factory.begin() as db:
        runtime = db.scalar(select(AutoReviewRuntimeKeyState).with_for_update())
        assert runtime is not None
        runtime.ready = False
        runtime.generation += 1

    with pytest.raises(ValidationStoreError, match='runtime key'):
        store.mark_attempt_started(_admission(request, claim))

    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        assert call is not None
        assert call.status == 'failed'
        assert call.provider_attempt_count == 0
        assert Decimal(call.charged_cost_usd) == Decimal('0')


def test_provider_price_drift_before_marker_is_zero_call_zero_charge(
    postgres_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-price-drift',
        review_item_ids=(957,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _store(factory, settings)
    claim = store.claim_or_replay(request)
    monkeypatch.setattr(
        validation_store_module,
        'AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M',
        Decimal('3.000000'),
    )

    with pytest.raises(ValidationStoreError, match='provider safety'):
        store.mark_attempt_started(_admission(request, claim))

    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        assert call is not None
        assert call.status == 'failed'
        assert call.provider_attempt_count == 0
        assert Decimal(call.charged_cost_usd) == Decimal('0')


def test_missing_rollout_authority_demotes_enforce_to_zero_call_shadow(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-rollout-missing',
        review_item_ids=(958,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _store(factory, settings)
    claim = store.claim_or_replay(request)
    enforce_admission = replace(
        _admission(request, claim), expected_effective_mode='enforce'
    )

    with pytest.raises(ValidationStoreError, match='rollout authority'):
        store.mark_attempt_started(enforce_admission)

    replay = store.claim_or_replay(request)
    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'post_validation_drift'
    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        assert call is not None
        assert call.provider_attempt_count == 0
        assert Decimal(call.charged_cost_usd) == Decimal('0')


def test_overrun_opens_one_call_attributed_breaker_and_replay_is_idempotent(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-overrun-7',
        review_item_ids=(961,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _store(factory, settings)
    claim = store.claim_or_replay(request)
    grant = store.mark_attempt_started(_admission(request, claim))

    class Invocation:
        prepared_content_hmac = request.prepared_content_hmac
        character_count = request.serialized_character_count
        framed_input_tokens = request.framed_input_tokens
        max_output_tokens = 3072

    store.dispatch_prepared_validation(
        invocation=Invocation(),
        provider=lambda invocation, **kwargs: None,
        grant=grant,
    )
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token,
    )
    candidate = request.candidates[0]
    completed = store.complete(
        context,
        ValidationBatchCompletion(
            validation_call_id=claim.validation_call_id,
            workflow_thread_id=request.workflow_thread_id,
            batch_fingerprint=request.batch_fingerprint,
            lease_token=claim.lease_token,
            candidates=(
                ValidationCandidateCompletion(
                    review_item_id=candidate.review_item_id,
                    validation_key=candidate.validation_key,
                    result=_validation_result('C01'),
                    policy_decision='auto_approve',
                    policy_reason_codes=('eligible',),
                ),
            ),
            input_tokens=request.framed_input_tokens + 1,
            output_tokens=10,
        ),
    )
    replay = store.claim_or_replay(request)

    assert completed == ()
    assert replay.failure_reason_code == 'budget_exceeded'
    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        safety = db.scalar(select(AutoReviewProviderSafetyState))
        events = tuple(db.scalars(select(AutoReviewProviderSafetyEvent)))
        assert call is not None and call.budget_overrun is True
        assert safety is not None and safety.breaker_open is True
        assert safety.state_version == 2
        assert safety.overrun_count == 1
        assert len(events) == 2
        assert events[-1].event_kind == 'budget_overrun'
        assert events[-1].call_hmac is not None


def test_source_change_before_marker_is_zero_call_zero_charge(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-source-drift-8',
        review_item_ids=(971,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _store(factory, settings)
    claim = store.claim_or_replay(request)
    with engine.begin() as connection:
        connection.execute(
            text(
                'UPDATE sources SET server_content_signature=:signature '
                'WHERE id=1001'
            ),
            {'signature': 'a' * 64},
        )

    with pytest.raises(ValidationStoreError, match='guard changed'):
        store.mark_attempt_started(_admission(request, claim))
    replay = store.claim_or_replay(request)

    assert replay.failure_reason_code == 'post_validation_drift'
    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        assert call is not None and call.provider_attempt_count == 0
        assert Decimal(call.charged_cost_usd) == Decimal('0')


def test_owner_permission_loss_during_provider_discards_result_and_charges_once(
    postgres_runtime,
) -> None:
    engine, database_url = postgres_runtime
    settings = _settings(database_url)
    request = _request(
        settings,
        workflow_thread_id='workflow-task9-permission-call-9',
        review_item_ids=(981,),
    )
    _seed_review_items(engine, (request,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    permission_levels = ['internal']
    store = _store(
        factory,
        settings,
        current_permission_resolver=lambda subject_id: tuple(permission_levels),
    )
    claim = store.claim_or_replay(request)
    grant = store.mark_attempt_started(_admission(request, claim))

    class Invocation:
        prepared_content_hmac = request.prepared_content_hmac
        character_count = request.serialized_character_count
        framed_input_tokens = request.framed_input_tokens
        max_output_tokens = 3072

    store.dispatch_prepared_validation(
        invocation=Invocation(),
        provider=lambda invocation, **kwargs: permission_levels.clear(),
        grant=grant,
    )
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token,
    )
    candidate = request.candidates[0]
    completed = store.complete(
        context,
        ValidationBatchCompletion(
            validation_call_id=claim.validation_call_id,
            workflow_thread_id=request.workflow_thread_id,
            batch_fingerprint=request.batch_fingerprint,
            lease_token=claim.lease_token,
            candidates=(
                ValidationCandidateCompletion(
                    review_item_id=candidate.review_item_id,
                    validation_key=candidate.validation_key,
                    result=_validation_result('C01'),
                    policy_decision='auto_approve',
                    policy_reason_codes=('eligible',),
                ),
            ),
            input_tokens=101,
            output_tokens=51,
        ),
    )
    replay = store.claim_or_replay(request)

    assert completed == ()
    assert replay.failure_reason_code == 'post_validation_drift'
    with factory() as db:
        call = db.get(AutoReviewValidationCall, claim.validation_call_id)
        child = db.scalar(select(AutoReviewValidation))
        assert call is not None
        assert Decimal(call.charged_cost_usd) == Decimal('0.000814')
        assert child is not None and child.claim_results == []
