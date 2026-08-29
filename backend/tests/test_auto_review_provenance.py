from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Generator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import create_engine, event, insert, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import make_url
from sqlalchemy.exc import (
    DataError,
    DBAPIError,
    DisconnectionError,
    IntegrityError,
    InterfaceError,
    InvalidRequestError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
    StatementError,
)
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import (
    AutoReviewKeyAdminError,
    AutoReviewKeyAdminService,
    FingerprintKeyRing,
    fingerprint_key_material_verifier,
)
from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.agent_runtime.keyed_mutation_guard import (
    AUTO_REVIEW_KEY_GENERATION_LOCK_ID,
    TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID,
    KeyedMutationGuard,
    KeyGenerationLockedContext,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import USERS
from backend.app.db.base import Base
from backend.app.db.initialization import (
    DatabaseConfigurationError,
    DatabaseInitializationError,
)
from backend.app.knowledge.claim_fingerprints import (
    normalized_claim_fingerprint,
    promoted_effect_fingerprint,
    trusted_title_collision_bucket,
)
from backend.app.knowledge.trusted_fingerprint_projection import (
    ProjectionRowSnapshot,
    ProjectionSummary,
    ProjectionSummaryDelta,
    build_hidden_collision_exists_statement,
    build_missing_active_projection_exists_statement,
    build_nonterminal_provider_call_lock_statement,
    projection_identity_ready,
    projection_row_digest,
    rebuild_trusted_fingerprint_projection,
    source_projection_snapshots,
)
from backend.app.knowledge.trusted_provenance import (
    TrustedProvenanceMismatch,
    find_explicit_promotion,
    has_legacy_human_base,
    validate_reuse_bundle,
)
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AutoReviewExtractionCall,
    AutoReviewProviderSafetyEvent,
    AutoReviewProviderSafetyState,
    AutoReviewRolloutControlEvent,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    AutoReviewValidation,
    AutoReviewValidationCall,
    Document,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    TimelineEvent,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    TrustedKnowledgeFingerprint,
    TrustedKnowledgeFingerprintProjectionState,
)
from backend.app.review.actors import ReuseExistingPromotion, human_review_actor
from backend.app.review.auto_review_resolution import (
    AutoReviewHumanOnly,
    AutoReviewResolutionService,
)
from backend.app.review.transitions import ReviewTransitionService
from backend.app.schemas.auto_review import AUTO_REVIEW_POLICY_VERSION
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
)


def _settings(*, version: str = 'v1', secret: str = 't' * 48) -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=True,
        database_url='sqlite:///:memory:',
        agent_runtime_fingerprint_key_version=version,
        agent_runtime_fingerprint_secret=secret,
        auto_review_mode='disabled',
    )


@pytest.fixture(scope='module')
def auto_review_postgres() -> Generator[
    tuple[sessionmaker[Session], Settings],
    None,
    None,
]:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip('exact auto reaffirmation requires disposable PostgreSQL')
    parsed = make_url(database_url)
    if (
        parsed.get_backend_name() != 'postgresql'
        or parsed.host != '127.0.0.1'
        or parsed.port != 55432
        or not (parsed.database or '').endswith('_test')
        or not (parsed.username or '').endswith('_test')
    ):
        raise ValueError('isolated PostgreSQL test database and role are required')
    database_url = parsed.set(drivername='postgresql+psycopg').render_as_string(
        hide_password=False
    )
    prior_mode = os.environ.get('PARAWORKS_DEMO_MODE')
    prior_url = os.environ.get('PARAWORKS_DATABASE_URL')
    os.environ['PARAWORKS_DEMO_MODE'] = 'false'
    os.environ['PARAWORKS_DATABASE_URL'] = database_url
    get_settings.cache_clear()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
    finally:
        get_settings.cache_clear()
        if prior_mode is None:
            os.environ.pop('PARAWORKS_DEMO_MODE', None)
        else:
            os.environ['PARAWORKS_DEMO_MODE'] = prior_mode
        if prior_url is None:
            os.environ.pop('PARAWORKS_DATABASE_URL', None)
        else:
            os.environ['PARAWORKS_DATABASE_URL'] = prior_url
    engine = create_engine(database_url)
    with engine.connect() as connection:
        database_name, user_name = connection.execute(
            text('SELECT current_database(), current_user')
        ).one()
    if database_name != parsed.database or user_name != parsed.username:
        raise ValueError('isolated PostgreSQL test database and role are required')
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=database_url,
        database_url=database_url,
        agent_runtime_fingerprint_key_version='v1',
        agent_runtime_fingerprint_secret='t' * 48,
        auto_review_mode='enforce',
        auto_review_enforce_percentage=100,
    )
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    with factory() as db:
        runtime = db.scalar(select(AutoReviewRuntimeKeyState))
        projection = db.scalar(select(TrustedKnowledgeFingerprintProjectionState))
        if runtime is None:
            runtime = AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier=verifier,
                generation=1,
                ready=True,
            )
            db.add(runtime)
        elif (
            runtime.fingerprint_key_version != 'v1'
            or runtime.fingerprint_key_material_verifier != verifier
        ):
            raise ValueError('disposable runtime key identity is not isolated')
        if projection is None:
            db.add(
                TrustedKnowledgeFingerprintProjectionState(
                    component='trusted_knowledge_fingerprints',
                    projection_schema_version='trusted-fingerprint-projection:v1',
                    fingerprint_key_version='v1',
                    fingerprint_key_material_verifier=verifier,
                    generation=runtime.generation,
                    ready=False,
                    rebuild_required=True,
                )
            )
        db.commit()
    rebuilt = rebuild_trusted_fingerprint_projection(
        session_factory=factory,
        settings=settings,
    )
    if not rebuilt.ready:
        raise AssertionError('disposable PostgreSQL projection did not become ready')
    try:
        yield factory, settings
    finally:
        engine.dispose()


@pytest.fixture
def auto_review_db_session(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> Generator[tuple[Session, Settings], None, None]:
    factory, settings = auto_review_postgres
    rebuilt = rebuild_trusted_fingerprint_projection(
        session_factory=factory,
        settings=settings,
    )
    if not rebuilt.ready:
        raise AssertionError('disposable PostgreSQL projection did not become ready')
    with factory() as db:
        yield db, settings


def _item(item_type: str, payload: dict[str, object] | None = None) -> ReviewItem:
    defaults: dict[str, dict[str, object]] = {
        'timeline_event': {
            'title': '  출시\t완료  ',
            'result_summary': ' 고객 배포가\n완료되었습니다. ',
        },
        'history_event': {
            'title': '  범위 변경  ',
            'reason': ' 고급 diff는\t다음 단계로 이동했습니다. ',
        },
        'decision_record': {
            'title': 'PostgreSQL 유지',
            'decision_summary': 'PostgreSQL을 기록 원장으로 유지합니다.',
        },
        'todo': {
            'title': '소유자 확인',
            'assignee': '김하나',
            'due_date': '2026-08-30',
            'priority': 'high',
            'priority_reason': '출시 전에 확인해야 합니다.',
        },
    }
    return ReviewItem(
        item_type=item_type,
        payload=payload or defaults[item_type],
        source_links=['https://example.test/evidence'],
        source_snippets=['exact evidence'],
        confidence_score=0.99,
        permission_level='internal',
        status='pending_review',
    )


def _effect_fingerprint(
    knowledge_type: str,
    fields: dict[str, str | None],
    *,
    project_key: str | None = 'project-alpha',
    scope: str = 'scope-a',
    settings: Settings | None = None,
) -> str:
    return promoted_effect_fingerprint(
        knowledge_type=knowledge_type,
        normalized_persisted_fields=fields,
        project_key=project_key,
        security_scope_id=scope,
        settings=settings or _settings(),
    )


def _snapshot(*, knowledge_id: int = 1, claim: str = 'a' * 64) -> ProjectionRowSnapshot:
    return ProjectionRowSnapshot(
        knowledge_type='timeline_event',
        knowledge_id=knowledge_id,
        scope_resolution='exact',
        security_scope_id='scope-a',
        project_scope_hmac='b' * 64,
        normalized_title_bucket_hmac='c' * 64,
        normalized_claim_fingerprint=claim,
        permission_level='internal',
        review_status='approved',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='d' * 64,
    )


def _current_permission_resolver(
    *,
    levels: tuple[str, ...] = ('public', 'internal'),
    resolved_subject_id: str | None = None,
):
    def resolve(requested_subject_id: str):
        return (
            resolved_subject_id or requested_subject_id,
            levels,
        )

    return resolve


def _auto_resolution_service(settings: Settings) -> AutoReviewResolutionService:
    return AutoReviewResolutionService(
        settings=settings,
        current_permission_resolver=_current_permission_resolver(),
    )


def _corrupt_projection_identity(
    db: Session,
    *,
    drift_kind: str,
    knowledge_type: str,
    knowledge_id: int,
) -> None:
    row = db.scalar(
        select(TrustedKnowledgeFingerprint).where(
            TrustedKnowledgeFingerprint.knowledge_type == knowledge_type,
            TrustedKnowledgeFingerprint.knowledge_id == knowledge_id,
        )
    )
    assert row is not None
    if drift_kind == 'title':
        row.normalized_title_bucket_hmac = '9' * 64
    elif drift_kind == 'claim':
        row.normalized_claim_fingerprint = '8' * 64
    elif drift_kind == 'project':
        row.project_scope_hmac = '7' * 64
    elif drift_kind == 'exact_scope':
        row.scope_resolution = 'legacy_unknown'
        row.security_scope_id = None
        row.normalized_claim_fingerprint = None
    elif drift_kind == 'legacy_scope':
        row.scope_resolution = 'exact'
        row.security_scope_id = 'scope-a'
        row.normalized_claim_fingerprint = '6' * 64
    elif drift_kind == 'extra':
        db.add(
            TrustedKnowledgeFingerprint(
                knowledge_type=row.knowledge_type,
                knowledge_id=2_000_000 + row.id,
                scope_resolution=row.scope_resolution,
                security_scope_id=row.security_scope_id,
                project_scope_hmac=row.project_scope_hmac,
                normalized_title_bucket_hmac=row.normalized_title_bucket_hmac,
                normalized_claim_fingerprint=row.normalized_claim_fingerprint,
                permission_level=row.permission_level,
                review_status=row.review_status,
                fingerprint_key_version=row.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    row.fingerprint_key_material_verifier
                ),
            )
        )
    else:
        raise AssertionError(f'unsupported projection drift: {drift_kind}')
    db.flush()


def _seed_c5_item(
    db: Session,
    *,
    item_type: str = 'history_event',
    graph_version: str = 'company-memory-review-v2.0',
    evidence_count: int = 2,
    payload: dict[str, object] | None = None,
    permission_level: str = 'internal',
    settings: Settings | None = None,
) -> ReviewItem:
    active_settings = settings or _settings()
    key = uuid4().hex
    thread = AgentWorkflowThread(
        thread_id=f'c5-task5:{key}',
        workflow_name='company-memory-review',
        graph_version=graph_version,
        checkpoint_thread_id=f'checkpoint:{key}',
        checkpoint_store='memory',
        owner_subject_id=USERS['admin'].id,
        security_scope_id='scope-a',
        input_hash='1' * 64,
        evidence_version_hash='2' * 64,
        status='awaiting_review',
    )
    run = AgentRun(
        agent_name='history_agent',
        prompt_version='history-extraction:c5-v1',
        status='complete',
        source_window='bounded',
        cache_key=f'cache:{key}',
        model_name='gpt-5.4-mini-2026-03-17',
        generation_provider='openai',
        generation_reasoning_effort='none',
        generation_route_version='auto-review-extraction-route:v1',
        generation_output_contract_version='history-candidate:c5-v1',
        permission_level='internal',
        workflow_thread_id=thread.thread_id,
    )
    db.add_all([thread, run])
    db.flush()
    sources = [
        Source(
            source_type='drive',
            source_id=f'drive:{key}:{ordinal}',
            source_url=f'https://example.test/evidence/{key}/{ordinal}',
            title=f'Exact evidence {ordinal}',
            permission_level=permission_level,
            raw_metadata={},
            server_content_signature_schema='server-source-content:v1',
            server_content_signature=f'{ordinal:x}'.zfill(64),
            connector_content_signature=f'fixture-{ordinal}',
        )
        for ordinal in range(1, evidence_count + 1)
    ]
    db.add_all(sources)
    db.flush()
    documents = [
        Document(
            source_id=source.id,
            title=source.title,
            current_version='v1',
            current_document_version_id=None,
        )
        for source in sources
    ]
    db.add_all(documents)
    db.flush()
    versions = [
        DocumentVersion(
            document_id=document.id,
            version='v1',
            body=f'Exact evidence body {ordinal}',
        )
        for ordinal, document in enumerate(documents, start=1)
    ]
    db.add_all(versions)
    db.flush()
    for document, version in zip(documents, versions, strict=True):
        document.current_document_version_id = version.id
    workflow_refs = []
    for ordinal, (source, version) in enumerate(
        zip(sources, versions, strict=True),
        start=1,
    ):
        workflow_refs.append(
            AgentWorkflowEvidenceRef(
                workflow_thread_id=thread.thread_id,
                ordinal=ordinal,
                canonical_source_type='drive',
                canonical_table='sources',
                canonical_row_id=source.id,
                document_version_id=version.id,
                external_revision=f'revision-{ordinal}',
                content_signature=f'{ordinal:x}'.zfill(64),
            permission_level_snapshot=permission_level,
                content_fingerprint=f'{ordinal + 10:x}'.zfill(64),
            )
        )
    db.add_all(workflow_refs)
    db.flush()
    item = _item(item_type, payload)
    if payload is None:
        item.payload = {
            **item.payload,
            'title': f"{item.payload['title']} {key}",
        }
    item.permission_level = permission_level
    item.workflow_thread_id = thread.thread_id
    item.candidate_key = f'candidate:{key}'
    item.agent_run_id = run.id
    item.candidate_contract_version = 'c5-v1'
    db.add(item)
    db.flush()
    db.add_all([
        ReviewItemEvidenceRef(
            review_item_id=item.id,
            workflow_thread_id=thread.thread_id,
            workflow_evidence_ref_id=ref.id,
            candidate_slot_ordinal=ordinal,
            message_content_fingerprint=f'{ordinal + 20:x}'.zfill(64),
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='f' * 64,
        )
        for ordinal, ref in enumerate(workflow_refs, start=1)
    ])
    thread.evidence_version_hash = build_keyed_fingerprint(
        [
            {
                'source_type': ref.canonical_source_type,
                'canonical_table': ref.canonical_table,
                'canonical_row_id': ref.canonical_row_id,
                'document_version_id': ref.document_version_id,
                'external_revision': ref.external_revision,
                'content_signature': ref.content_signature,
                'permission_level': ref.permission_level_snapshot,
                'content_fingerprint': ref.content_fingerprint,
            }
            for ref in sorted(workflow_refs, key=lambda row: row.ordinal)
        ],
        settings=active_settings,
        schema_version='review-evidence-versions:v1',
        policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    )
    db.commit()
    db.refresh(item)
    return item


def _seed_completed_validation(
    db: Session,
    item: ReviewItem,
    *,
    settings: Settings | None = None,
) -> AutoReviewValidation:
    now = datetime.now(UTC)
    active_settings = settings or _settings()
    verifier = fingerprint_key_material_verifier(
        active_settings.agent_runtime_fingerprint_secret
    )
    workflow = db.get(AgentWorkflowThread, item.workflow_thread_id)
    assert workflow is not None
    run = db.get(AgentRun, item.agent_run_id)
    assert run is not None
    extraction_model = f'task5-extraction-{item.id}'
    validation_model = f'task5-validation-{item.id}'
    rollout = db.scalar(
        select(AutoReviewRolloutState).where(
            AutoReviewRolloutState.security_scope_id == workflow.security_scope_id,
            AutoReviewRolloutState.policy_version == AUTO_REVIEW_POLICY_VERSION,
        )
    )
    if rollout is None:
        rollout = AutoReviewRolloutState(
            security_scope_id=workflow.security_scope_id,
            policy_version=AUTO_REVIEW_POLICY_VERSION,
        )
        db.add(rollout)
        db.commit()
    if rollout.max_authorized_percentage != 100:
        percentages = (
            (10, 100) if rollout.max_authorized_percentage == 0 else (100,)
        )
        for percentage in percentages:
            prior_state_version = rollout.state_version
            prior_control_epoch = rollout.control_epoch
            prior_percentage = rollout.max_authorized_percentage
            prior_generation = rollout.authorization_generation
            authorized_at = datetime.now(UTC)
            rollout_event = AutoReviewRolloutControlEvent(
                rollout_state_id=rollout.id,
                security_scope_id=rollout.security_scope_id,
                policy_version=rollout.policy_version,
                event_sequence=rollout.last_event_sequence + 1,
                event_kind='percentage_authorized',
                prior_state_version=prior_state_version,
                new_state_version=prior_state_version + 1,
                prior_control_epoch=prior_control_epoch,
                new_control_epoch=prior_control_epoch + 1,
                prior_max_authorized_percentage=prior_percentage,
                new_max_authorized_percentage=percentage,
                prior_breaker_open=False,
                new_breaker_open=False,
                prior_authorization_generation=prior_generation,
                new_authorization_generation=prior_generation + 1,
                regression_gate_reference=f'task5-{percentage}',
                actor_subject_hmac='a' * 64,
                fingerprint_key_version=active_settings.agent_runtime_fingerprint_key_version,
                fingerprint_key_material_verifier=verifier,
                created_at=authorized_at,
            )
            db.add(rollout_event)
            db.flush()
            rollout.state_version = prior_state_version + 1
            rollout.control_epoch = prior_control_epoch + 1
            rollout.max_authorized_percentage = percentage
            rollout.authorization_generation = prior_generation + 1
            rollout.authorization_at = authorized_at
            rollout.regression_gate_reference = f'task5-{percentage}'
            rollout.last_event_sequence += 1
            rollout.last_event_id = rollout_event.id
            db.commit()

    for purpose, model, effort in (
        ('extraction', extraction_model, 'none'),
        ('validation', validation_model, 'medium'),
    ):
        safety = AutoReviewProviderSafetyState(
            purpose=purpose,
            provider='openai',
            model=model,
            reasoning_effort=effort,
            state_version=1,
            authorized_cost_policy_version='auto-review-cost:v1',
            token_estimator_version='openai-o200k-chat:v1',
            tokenizer_encoding='o200k_base',
            reply_priming_tokens=16,
            framing_safety_tokens=512,
            input_usd_per_1m=Decimal('2.000000'),
            output_usd_per_1m=Decimal('12.000000'),
            breaker_open=False,
            authorized_at=now,
        )
        db.add(safety)
        db.flush()
        safety_event = AutoReviewProviderSafetyEvent(
            provider_safety_state_id=safety.id,
            purpose=purpose,
            provider='openai',
            model=model,
            reasoning_effort=effort,
            event_sequence=1,
            event_kind='initial_authorized',
            prior_state_version=0,
            new_state_version=1,
            cost_policy_version='auto-review-cost:v1',
            token_estimator_version='openai-o200k-chat:v1',
            tokenizer_encoding='o200k_base',
            reply_priming_tokens=16,
            framing_safety_tokens=512,
            input_usd_per_1m=Decimal('2.000000'),
            output_usd_per_1m=Decimal('12.000000'),
            prior_breaker_open=False,
            new_breaker_open=False,
            actor_subject_hmac='b' * 64,
            fingerprint_key_version=active_settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            created_at=now,
        )
        db.add(safety_event)
        db.flush()
        safety.last_event_sequence = 1
        safety.last_event_id = safety_event.id

    db.add(
        AgentWorkflowRequest(
            workflow_thread_id=workflow.thread_id,
            input_schema_version='review-workflow:v2.1',
            request_kind='review_source_versions',
            agent_names=[run.agent_name],
            selection_policy_version='selection:v1',
            input_hash=workflow.input_hash,
            fingerprint_key_version=active_settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            auto_review_mode='enforce',
            auto_review_validator_provider='openai',
            auto_review_validator_model=validation_model,
            auto_review_reasoning_effort='medium',
            auto_review_policy_version=AUTO_REVIEW_POLICY_VERSION,
            auto_review_cost_policy_version='auto-review-cost:v1',
            auto_review_extraction_cost_policy_version='auto-review-cost:v1',
            auto_review_token_estimator_version='openai-o200k-chat:v1',
            auto_review_extraction_token_estimator_version='openai-o200k-chat:v1',
            auto_review_tokenizer_encoding='o200k_base',
            auto_review_reply_priming_tokens=16,
            auto_review_framing_safety_tokens=512,
            auto_review_validator_input_usd_per_1m=Decimal('2.000000'),
            auto_review_validator_output_usd_per_1m=Decimal('12.000000'),
            auto_review_extraction_input_usd_per_1m=Decimal('2.000000'),
            auto_review_extraction_output_usd_per_1m=Decimal('12.000000'),
            auto_review_extraction_provider='openai',
            auto_review_extraction_model=extraction_model,
            auto_review_extraction_reasoning_effort='none',
            auto_review_extraction_route_version='auto-review-extraction-route:v1',
            auto_review_enforce_percentage=100,
            authorized_percentage_at_launch=rollout.max_authorized_percentage,
            rollout_authorization_generation=rollout.authorization_generation,
            validation_provider_safety_state_version=1,
            rollout_control_epoch=rollout.control_epoch,
        )
    )
    db.add(
        AutoReviewExtractionCall(
            agent_run_id=run.id,
            workflow_thread_id=workflow.thread_id,
            agent_name=run.agent_name,
            extraction_plan_hmac='d' * 64,
            provider='openai',
            model=extraction_model,
            reasoning_effort='none',
            route_version='auto-review-extraction-route:v1',
            prompt_version=run.prompt_version,
            output_contract_version='history-candidate:c5-v1',
            extraction_registry_version='auto-review-extraction-registry:v1',
            cost_policy_version='auto-review-cost:v1',
            provider_safety_state_version=1,
            token_estimator_version='openai-o200k-chat:v1',
            tokenizer_encoding='o200k_base',
            reply_priming_tokens=16,
            framing_safety_tokens=512,
            prepared_content_hmac='e' * 64,
            prepared_character_count=10,
            framed_input_token_cap=10,
            total_output_token_cap=10,
            max_candidates_per_agent=1,
            input_usd_per_1m=Decimal('2.000000'),
            output_usd_per_1m=Decimal('12.000000'),
            fingerprint_key_version=active_settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            provider_timeout_seconds=60,
            provider_send_start_window_seconds=5,
            provider_attempt_lease_seconds=120,
            provider_commit_grace_seconds=30,
            workflow_extraction_cost_ceiling_usd=Decimal('0.016716'),
            workflow_total_cost_ceiling_usd=Decimal('0.065580'),
            status='completed',
            max_provider_attempts=1,
            provider_attempt_count=1,
            attempt_started_at=now,
            reserved_input_tokens=10,
            reserved_output_tokens=10,
            reserved_cost_usd=Decimal('0.000100'),
            charged_input_tokens=10,
            charged_output_tokens=10,
            charged_cost_usd=Decimal('0.000100'),
            result_kind='candidate',
            result_candidate_count=1,
            result_candidate_set_hmac='f' * 64,
            terminal_at=now,
        )
    )
    call = AutoReviewValidationCall(
        workflow_thread_id=item.workflow_thread_id,
        batch_fingerprint=uuid4().hex.ljust(64, '0'),
        status='completed',
        candidate_count=1,
        max_provider_attempts=1,
        provider_attempt_count=1,
        attempt_started_at=now,
        reserved_input_tokens=10,
        reserved_output_tokens=10,
        reserved_cost_usd=Decimal('0.000100'),
        charged_input_tokens=10,
        charged_output_tokens=10,
        charged_cost_usd=Decimal('0.000100'),
        prepared_content_hmac='a' * 64,
        serialized_character_count=10,
        framed_input_token_count=10,
        max_output_tokens=10,
        token_estimator_version='openai-o200k-chat:v1',
        tokenizer_encoding='o200k_base',
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        input_usd_per_1m=Decimal('2.000000'),
        output_usd_per_1m=Decimal('12.000000'),
        cost_policy_version='auto-review-cost:v1',
        provider_safety_state_version=1,
        fingerprint_key_version=active_settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=verifier,
        workflow_extraction_cost_ceiling_usd=Decimal('0.016716'),
        workflow_validation_cost_ceiling_usd=Decimal('0.048864'),
        workflow_total_cost_ceiling_usd=Decimal('0.065580'),
        provider_timeout_seconds=60,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=120,
        provider_commit_grace_seconds=30,
        terminal_at=now,
    )
    db.add(call)
    db.flush()
    validation = AutoReviewValidation(
        review_item_id=item.id,
        validation_call_id=call.id,
        workflow_thread_id=item.workflow_thread_id,
        validation_key=uuid4().hex.ljust(64, '0'),
        evidence_version_hash=workflow.evidence_version_hash,
        candidate_generation_fingerprint='c' * 64,
        status='completed',
        validator_provider='openai',
        validator_model=validation_model,
        reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v1',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version=AUTO_REVIEW_POLICY_VERSION,
        fingerprint_key_version=active_settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=verifier,
        cost_policy_version='auto-review-cost:v1',
        confirmed_validation_cost_ceiling_usd=Decimal('0.048864'),
        completed_at=now,
    )
    db.add(validation)
    db.commit()
    db.refresh(validation)
    return validation


def _seed_claimed_validation_call(
    db: Session,
    *,
    provider_attempt_count: int,
    settings: Settings,
) -> tuple[AutoReviewValidationCall, AutoReviewValidation]:
    item = _seed_c5_item(db)
    now = datetime.now(UTC)
    call = AutoReviewValidationCall(
        workflow_thread_id=item.workflow_thread_id,
        batch_fingerprint=uuid4().hex.ljust(64, '0'),
        status='claimed',
        candidate_count=1,
        max_provider_attempts=1,
        provider_attempt_count=provider_attempt_count,
        attempt_started_at=now if provider_attempt_count == 1 else None,
        reserved_input_tokens=10,
        reserved_output_tokens=10,
        reserved_cost_usd=Decimal('0.000100'),
        charged_input_tokens=None,
        charged_output_tokens=None,
        charged_cost_usd=Decimal('0'),
        prepared_content_hmac='a' * 64,
        serialized_character_count=10,
        framed_input_token_count=10,
        max_output_tokens=10,
        token_estimator_version='openai-o200k-chat:v1',
        tokenizer_encoding='o200k_base',
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        input_usd_per_1m=Decimal('2.000000'),
        output_usd_per_1m=Decimal('12.000000'),
        cost_policy_version='auto-review-cost:v1',
        provider_safety_state_version=1,
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ),
        workflow_extraction_cost_ceiling_usd=Decimal('0.016716'),
        workflow_validation_cost_ceiling_usd=Decimal('0.048864'),
        workflow_total_cost_ceiling_usd=Decimal('0.065580'),
        provider_timeout_seconds=60,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=120,
        provider_commit_grace_seconds=30,
        terminal_at=None,
    )
    db.add(call)
    db.flush()
    validation = AutoReviewValidation(
        review_item_id=item.id,
        validation_call_id=call.id,
        workflow_thread_id=item.workflow_thread_id,
        validation_key=uuid4().hex.ljust(64, '0'),
        evidence_version_hash='2' * 64,
        candidate_generation_fingerprint='3' * 64,
        status='claimed',
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v1',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version=AUTO_REVIEW_POLICY_VERSION,
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ),
        cost_policy_version='auto-review-cost:v1',
        confirmed_validation_cost_ceiling_usd=Decimal('0.048864'),
        claim_results=[],
        uncertainty_codes=[],
        conflict_codes=[],
        policy_reason_codes=[],
    )
    db.add(validation)
    db.commit()
    db.refresh(call)
    db.refresh(validation)
    return call, validation


def _open_provider_breaker(
    db: Session,
    state: AutoReviewProviderSafetyState,
    *,
    settings: Settings,
) -> None:
    event_at = datetime.now(UTC)
    event = AutoReviewProviderSafetyEvent(
        provider_safety_state_id=state.id,
        purpose=state.purpose,
        provider=state.provider,
        model=state.model,
        reasoning_effort=state.reasoning_effort,
        event_sequence=state.last_event_sequence + 1,
        event_kind='budget_overrun',
        prior_state_version=state.state_version,
        new_state_version=state.state_version + 1,
        cost_policy_version=state.authorized_cost_policy_version,
        token_estimator_version=state.token_estimator_version,
        tokenizer_encoding=state.tokenizer_encoding,
        reply_priming_tokens=state.reply_priming_tokens,
        framing_safety_tokens=state.framing_safety_tokens,
        input_usd_per_1m=state.input_usd_per_1m,
        output_usd_per_1m=state.output_usd_per_1m,
        prior_breaker_open=state.breaker_open,
        new_breaker_open=True,
        reason_code='budget_overrun',
        call_hmac='9' * 64,
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ),
        created_at=event_at,
    )
    db.add(event)
    db.flush()
    state.state_version += 1
    state.breaker_open = True
    state.breaker_reason_code = 'budget_overrun'
    state.overrun_count += 1
    state.last_overrun_cost_usd = Decimal('0.000100')
    state.last_overrun_at = event_at
    state.last_event_sequence += 1
    state.last_event_id = event.id
    db.flush()


def _invalidate_rollout(
    db: Session,
    state: AutoReviewRolloutState,
    *,
    settings: Settings,
) -> None:
    event = AutoReviewRolloutControlEvent(
        rollout_state_id=state.id,
        security_scope_id=state.security_scope_id,
        policy_version=state.policy_version,
        event_sequence=state.last_event_sequence + 1,
        event_kind='generation_invalidated',
        prior_state_version=state.state_version,
        new_state_version=state.state_version + 1,
        prior_control_epoch=state.control_epoch,
        new_control_epoch=state.control_epoch + 1,
        prior_max_authorized_percentage=state.max_authorized_percentage,
        new_max_authorized_percentage=0,
        prior_breaker_open=state.breaker_open,
        new_breaker_open=state.breaker_open,
        prior_authorization_generation=state.authorization_generation,
        new_authorization_generation=state.authorization_generation + 1,
        reason_code=state.breaker_reason_code,
        regression_gate_reference=state.regression_gate_reference,
        actor_subject_hmac='8' * 64,
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ),
    )
    db.add(event)
    db.flush()
    state.state_version += 1
    state.control_epoch += 1
    state.max_authorized_percentage = 0
    state.authorization_generation += 1
    state.authorization_at = None
    state.last_event_sequence += 1
    state.last_event_id = event.id
    db.flush()


def _seed_additional_completed_validation(
    db: Session,
    validation: AutoReviewValidation,
) -> AutoReviewValidation:
    original_call = db.get(AutoReviewValidationCall, validation.validation_call_id)
    assert original_call is not None
    call_values = {
        column.name: getattr(original_call, column.name)
        for column in AutoReviewValidationCall.__table__.columns
        if not column.primary_key
    }
    call_values['batch_fingerprint'] = uuid4().hex.ljust(64, '0')
    call = AutoReviewValidationCall(**call_values)
    db.add(call)
    db.flush()
    validation_values = {
        column.name: getattr(validation, column.name)
        for column in AutoReviewValidation.__table__.columns
        if not column.primary_key
    }
    validation_values['validation_call_id'] = call.id
    validation_values['validation_key'] = uuid4().hex.ljust(64, '0')
    additional = AutoReviewValidation(**validation_values)
    db.add(additional)
    db.commit()
    db.refresh(additional)
    return additional


def _mark_claimed_validation_failed(
    db: Session,
    call: AutoReviewValidationCall,
    validation: AutoReviewValidation,
) -> None:
    call.status = 'failed'
    call.charged_input_tokens = (
        call.reserved_input_tokens if call.provider_attempt_count == 1 else 0
    )
    call.charged_output_tokens = (
        call.reserved_output_tokens if call.provider_attempt_count == 1 else 0
    )
    call.charged_cost_usd = (
        call.reserved_cost_usd if call.provider_attempt_count == 1 else Decimal('0')
    )
    call.terminal_at = datetime.now(UTC)
    validation.status = 'failed'
    validation.completed_at = datetime.now(UTC)
    db.commit()


class _StaticKeyRingSource:
    def __init__(self, ring: FingerprintKeyRing) -> None:
        self._ring = ring

    def load(self) -> FingerprintKeyRing:
        return self._ring


def _rotation_service(
    factory: sessionmaker[Session],
    *,
    settings: Settings,
    current_version: str,
    current_secret: str,
    next_version: str,
    next_secret: str,
) -> AutoReviewKeyAdminService:
    return AutoReviewKeyAdminService(
        session_factory=factory,
        settings=settings.model_copy(
            update={
                'auto_review_mode': 'disabled',
                'agent_runtime_fingerprint_key_version': current_version,
                'agent_runtime_fingerprint_secret': current_secret,
            }
        ),
        key_ring_source=_StaticKeyRingSource(
            FingerprintKeyRing(
                current_version=current_version,
                current_secret=SecretStr(current_secret),
                next_version=next_version,
                next_secret=SecretStr(next_secret),
            )
        ),
    )


def _reuse_directive_for_item(
    db: Session,
    item: ReviewItem,
) -> ReuseExistingPromotion:
    links = tuple(
        db.scalars(
            select(TrustedKnowledgeApprovalLink)
            .where(TrustedKnowledgeApprovalLink.review_item_id == item.id)
            .order_by(TrustedKnowledgeApprovalLink.promotion_effect_kind.desc())
        ).all()
    )
    assert len(links) == 2
    primary, companion = links
    return ReuseExistingPromotion(
        expected_type='history_event',
        expected_id=primary.knowledge_id,
        expected_claim_fingerprint=primary.claim_fingerprint,
        expected_companion_id=companion.knowledge_id,
        expected_companion_claim_fingerprint=companion.claim_fingerprint,
    )


def _prepare_exact_auto_reaffirmation(
    db: Session,
    *,
    settings: Settings,
) -> tuple[ReviewItem, ReviewItem, ReuseExistingPromotion]:
    canonical = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=canonical.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    directive = _reuse_directive_for_item(db, canonical)
    candidate = _seed_c5_item(
        db,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(canonical.payload),
    )
    _seed_completed_validation(db, candidate, settings=settings)
    return canonical, candidate, directive


def test_timeline_fingerprint_uses_only_title_and_result_summary() -> None:
    settings = _settings()
    left = _item('timeline_event')
    right = _item('timeline_event', {**left.payload, 'metadata': {'rank': 1}})
    assert normalized_claim_fingerprint(
        item=left, security_scope_id='scope-a', settings=settings
    ) == normalized_claim_fingerprint(
        item=right, security_scope_id='scope-a', settings=settings
    )


def test_history_fingerprint_uses_only_title_and_reason() -> None:
    settings = _settings()
    left = _item('history_event')
    right = _item('history_event', {**left.payload, 'summary': 'ignored metadata'})
    assert normalized_claim_fingerprint(
        item=left, security_scope_id='scope-a', settings=settings
    ) == normalized_claim_fingerprint(
        item=right, security_scope_id='scope-a', settings=settings
    )


def test_human_decision_and_todo_effects_receive_target_specific_fingerprints() -> None:
    decision = _effect_fingerprint(
        'decision_record',
        {'title': 'A', 'decision_summary': 'B'},
    )
    todo = _effect_fingerprint(
        'todo',
        {
            'title': 'A',
            'assignee': None,
            'due_date': None,
            'priority': 'high',
            'priority_reason': 'B',
        },
    )
    assert len(decision) == len(todo) == 64
    assert decision != todo


def test_companion_timeline_uses_its_own_written_target_fingerprint() -> None:
    primary = _effect_fingerprint(
        'decision_record', {'title': 'Use Postgres', 'decision_summary': 'Durable'}
    )
    companion = _effect_fingerprint(
        'timeline_event',
        {'title': '[결정] Use Postgres', 'result_summary': 'Durable'},
    )
    assert primary != companion


def test_metadata_order_unicode_nfc_and_whitespace_are_canonical() -> None:
    composed = _effect_fingerprint(
        'timeline_event',
        {'title': 'Café  출시', 'result_summary': '완료\t됨'},
    )
    decomposed = _effect_fingerprint(
        'timeline_event',
        {'result_summary': ' 완료   됨 ', 'title': 'Cafe\u0301 출시'},
    )
    assert composed == decomposed


def test_different_target_project_scope_or_substantive_value_changes_hmac() -> None:
    base = _effect_fingerprint(
        'timeline_event', {'title': 'A', 'result_summary': 'B'}
    )
    variants = {
        _effect_fingerprint('history_event', {'title': 'A', 'reason': 'B'}),
        _effect_fingerprint(
            'timeline_event', {'title': 'A', 'result_summary': 'B'}, project_key='p2'
        ),
        _effect_fingerprint(
            'timeline_event', {'title': 'A', 'result_summary': 'B'}, scope='scope-b'
        ),
        _effect_fingerprint(
            'timeline_event', {'title': 'A', 'result_summary': 'C'}
        ),
    }
    assert base not in variants
    assert len(variants) == 4


def test_create_new_writes_primary_companion_and_all_evidence_links_once(
    db_session: Session,
) -> None:
    item = _seed_c5_item(db_session)
    refs = tuple(
        db_session.scalars(
            select(AgentWorkflowEvidenceRef)
            .where(AgentWorkflowEvidenceRef.workflow_thread_id == item.workflow_thread_id)
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).all()
    )
    bindings = tuple(
        db_session.scalars(
            select(ReviewItemEvidenceRef)
            .where(ReviewItemEvidenceRef.review_item_id == item.id)
            .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
        ).all()
    )
    refs[1].canonical_row_id = refs[0].canonical_row_id
    refs[1].document_version_id = refs[0].document_version_id
    refs[1].external_revision = refs[0].external_revision
    refs[1].content_signature = refs[0].content_signature
    refs[1].content_fingerprint = refs[0].content_fingerprint
    bindings[1].message_content_fingerprint = bindings[0].message_content_fingerprint
    workflow = db_session.get(AgentWorkflowThread, item.workflow_thread_id)
    assert workflow is not None
    workflow.evidence_version_hash = build_keyed_fingerprint(
        [
            {
                'source_type': ref.canonical_source_type,
                'canonical_table': ref.canonical_table,
                'canonical_row_id': ref.canonical_row_id,
                'document_version_id': ref.document_version_id,
                'external_revision': ref.external_revision,
                'content_signature': ref.content_signature,
                'permission_level': ref.permission_level_snapshot,
                'content_fingerprint': ref.content_fingerprint,
            }
            for ref in sorted(refs, key=lambda row: row.ordinal)
        ],
        settings=_settings(),
        schema_version='review-evidence-versions:v1',
        policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    )
    db_session.commit()
    result = ReviewTransitionService(settings=_settings()).transition(
        db=db_session,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    links = tuple(
        db_session.scalars(
            select(TrustedKnowledgeApprovalLink)
            .where(TrustedKnowledgeApprovalLink.review_item_id == item.id)
            .order_by(TrustedKnowledgeApprovalLink.promotion_effect_kind)
        ).all()
    )
    children = tuple(db_session.scalars(select(TrustedKnowledgeEvidenceLink)).all())
    assert result.promotion is not None
    assert result.promotion.effect == 'created'
    assert [link.promotion_effect_kind for link in links] == ['companion', 'primary']
    assert len(children) == 4
    assert {child.approval_link_id for child in children} == {link.id for link in links}
    assert all(
        len({child.evidence_hash for child in children if child.approval_link_id == link.id})
        == 2
        for link in links
    )


def test_create_new_replay_keeps_created_effect_and_does_not_append_links(
    db_session: Session,
) -> None:
    item = _seed_c5_item(db_session)
    service = ReviewTransitionService(settings=_settings())
    created = service.transition(
        db=db_session,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    replay = service.transition(
        db=db_session,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    assert created.promotion is not None and created.promotion.effect == 'created'
    assert replay.replayed is True
    assert replay.promotion == created.promotion
    assert len(
        db_session.scalars(
            select(TrustedKnowledgeApprovalLink).where(
                TrustedKnowledgeApprovalLink.review_item_id == item.id
            )
        ).all()
    ) == 2


def test_provenance_replay_refuses_a_missing_evidence_child(
    db_session: Session,
) -> None:
    item = _seed_c5_item(db_session)
    ReviewTransitionService(settings=_settings()).transition(
        db=db_session,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    link = db_session.scalar(
        select(TrustedKnowledgeApprovalLink).where(
            TrustedKnowledgeApprovalLink.review_item_id == item.id,
            TrustedKnowledgeApprovalLink.promotion_effect_kind == 'primary',
        )
    )
    assert link is not None
    child = db_session.scalars(
        select(TrustedKnowledgeEvidenceLink).where(
            TrustedKnowledgeEvidenceLink.approval_link_id == link.id
        )
    ).first()
    assert child is not None
    db_session.delete(child)
    db_session.flush()
    with pytest.raises(TrustedProvenanceMismatch):
        find_explicit_promotion(db_session, item=item, settings=_settings())


def test_exact_reaffirmation_reuses_canonical_ids_and_adds_own_links(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db_session, settings = auto_review_db_session
    first_item = _seed_c5_item(db_session)
    created = ReviewTransitionService(settings=settings).transition(
        db=db_session,
        item_id=first_item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    second_item = _seed_c5_item(
        db_session,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(first_item.payload),
    )
    _seed_completed_validation(db_session, second_item, settings=settings)
    primary_link = db_session.scalar(
        select(TrustedKnowledgeApprovalLink).where(
            TrustedKnowledgeApprovalLink.review_item_id == first_item.id,
            TrustedKnowledgeApprovalLink.promotion_effect_kind == 'primary',
        )
    )
    companion_link = db_session.scalar(
        select(TrustedKnowledgeApprovalLink).where(
            TrustedKnowledgeApprovalLink.review_item_id == first_item.id,
            TrustedKnowledgeApprovalLink.promotion_effect_kind == 'companion',
        )
    )
    assert primary_link is not None and companion_link is not None
    directive = ReuseExistingPromotion(
        expected_type='history_event',
        expected_id=primary_link.knowledge_id,
        expected_claim_fingerprint=primary_link.claim_fingerprint,
        expected_companion_id=companion_link.knowledge_id,
        expected_companion_claim_fingerprint=companion_link.claim_fingerprint,
    )
    reaffirmed = _auto_resolution_service(settings).resolve(
        db=db_session,
        item_id=second_item.id,
        directive=directive,
    )
    assert reaffirmed.promotion is not None
    assert reaffirmed.promotion.effect == 'reaffirmed'
    assert reaffirmed.promotion.created_record_ids == created.promotion.created_record_ids
    assert reaffirmed.promotion.created_timeline_event_ids == (
        created.promotion.created_timeline_event_ids
    )
    item_ids = (first_item.id, second_item.id)
    assert len(
        db_session.scalars(
            select(HistoryEvent).where(HistoryEvent.source_review_item_id.in_(item_ids))
        ).all()
    ) == 1
    assert len(
        db_session.scalars(
            select(TimelineEvent).where(TimelineEvent.source_review_item_id.in_(item_ids))
        ).all()
    ) == 1
    approval_links = tuple(
        db_session.scalars(
            select(TrustedKnowledgeApprovalLink).where(
                TrustedKnowledgeApprovalLink.review_item_id.in_(item_ids)
            )
        ).all()
    )
    assert len(approval_links) == 4
    assert len(
        db_session.scalars(
            select(TrustedKnowledgeEvidenceLink).where(
                TrustedKnowledgeEvidenceLink.approval_link_id.in_(
                    [link.id for link in approval_links]
                )
            )
        ).all()
    ) == 8


def test_reaffirmation_replay_returns_same_canonical_result(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db_session, settings = auto_review_db_session
    first_item = _seed_c5_item(db_session)
    ReviewTransitionService(settings=settings).transition(
        db=db_session,
        item_id=first_item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    links = tuple(
        db_session.scalars(
            select(TrustedKnowledgeApprovalLink)
            .where(TrustedKnowledgeApprovalLink.review_item_id == first_item.id)
            .order_by(TrustedKnowledgeApprovalLink.promotion_effect_kind.desc())
        ).all()
    )
    primary, companion = links
    second_item = _seed_c5_item(
        db_session,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(first_item.payload),
    )
    _seed_completed_validation(db_session, second_item, settings=settings)
    directive = ReuseExistingPromotion(
        expected_type='history_event',
        expected_id=primary.knowledge_id,
        expected_claim_fingerprint=primary.claim_fingerprint,
        expected_companion_id=companion.knowledge_id,
        expected_companion_claim_fingerprint=companion.claim_fingerprint,
    )
    service = _auto_resolution_service(settings)
    first = service.resolve(db=db_session, item_id=second_item.id, directive=directive)
    replay = service.resolve(db=db_session, item_id=second_item.id, directive=directive)
    assert replay.replayed is True
    assert replay.promotion == first.promotion
    assert len(
        db_session.scalars(
            select(TrustedKnowledgeApprovalLink).where(
                TrustedKnowledgeApprovalLink.review_item_id.in_(
                    (first_item.id, second_item.id)
                )
            )
        ).all()
    ) == 4


@pytest.mark.parametrize(
    'readiness_failure',
    ['disabled', 'stale', 'key_mismatch', 'active_projection_anti_join'],
)
def test_auto_approval_is_human_only_when_admission_is_not_fresh(
    auto_review_db_session: tuple[Session, Settings],
    readiness_failure: str,
) -> None:
    db, settings = auto_review_db_session
    canonical = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=canonical.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    directive = _reuse_directive_for_item(db, canonical)
    candidate = _seed_c5_item(
        db,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(canonical.payload),
    )
    _seed_completed_validation(db, candidate, settings=settings)
    effective_settings = settings
    state = db.scalars(select(TrustedKnowledgeFingerprintProjectionState)).one()
    if readiness_failure == 'disabled':
        effective_settings = settings.model_copy(update={'auto_review_mode': 'disabled'})
    elif readiness_failure == 'stale':
        state.ready = False
        state.rebuild_required = True
    elif readiness_failure == 'key_mismatch':
        state.fingerprint_key_material_verifier = 'f' * 64
    else:
        projected = db.scalars(
            select(TrustedKnowledgeFingerprint).order_by(
                TrustedKnowledgeFingerprint.knowledge_type,
                TrustedKnowledgeFingerprint.knowledge_id,
            )
        ).first()
        assert projected is not None
        db.delete(projected)
    db.commit()

    with pytest.raises(AutoReviewHumanOnly):
        _auto_resolution_service(effective_settings).resolve(
            db=db,
            item_id=candidate.id,
            directive=directive,
        )
    assert db.get(ReviewItem, candidate.id).status == 'pending_review'


@pytest.mark.parametrize('control_failure', ['provider_breaker', 'rollout_stale'])
def test_auto_approval_rechecks_provider_and_rollout_control_snapshots(
    auto_review_db_session: tuple[Session, Settings],
    control_failure: str,
) -> None:
    db, settings = auto_review_db_session
    canonical = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=canonical.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    directive = _reuse_directive_for_item(db, canonical)
    candidate = _seed_c5_item(
        db,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(canonical.payload),
    )
    validation = _seed_completed_validation(db, candidate, settings=settings)
    workflow = db.get(AgentWorkflowThread, candidate.workflow_thread_id)
    assert workflow is not None
    if control_failure == 'provider_breaker':
        safety = db.scalar(
            select(AutoReviewProviderSafetyState).where(
                AutoReviewProviderSafetyState.purpose == 'validation',
                AutoReviewProviderSafetyState.provider
                == validation.validator_provider,
                AutoReviewProviderSafetyState.model == validation.validator_model,
                AutoReviewProviderSafetyState.reasoning_effort
                == validation.reasoning_effort,
            )
        )
        assert safety is not None
        _open_provider_breaker(db, safety, settings=settings)
    else:
        rollout = db.scalar(
            select(AutoReviewRolloutState).where(
                AutoReviewRolloutState.security_scope_id
                == workflow.security_scope_id,
                AutoReviewRolloutState.policy_version == AUTO_REVIEW_POLICY_VERSION,
            )
        )
        assert rollout is not None
        _invalidate_rollout(db, rollout, settings=settings)
    db.commit()

    with pytest.raises(AutoReviewHumanOnly):
        _auto_resolution_service(settings).resolve(
            db=db,
            item_id=candidate.id,
            directive=directive,
        )
    assert db.get(ReviewItem, candidate.id).status == 'pending_review'


@pytest.mark.parametrize('conflict_permission', ['internal', 'restricted'])
def test_auto_reaffirmation_rechecks_visible_and_hidden_title_collisions(
    auto_review_db_session: tuple[Session, Settings],
    conflict_permission: str,
) -> None:
    db, settings = auto_review_db_session
    canonical = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=canonical.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    directive = _reuse_directive_for_item(db, canonical)
    conflict = _seed_c5_item(
        db,
        payload={
            'title': canonical.payload['title'],
            'reason': 'A different approved claim shares this title.',
        },
        permission_level=conflict_permission,
    )
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=conflict.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    candidate = _seed_c5_item(
        db,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(canonical.payload),
    )
    _seed_completed_validation(db, candidate, settings=settings)

    with pytest.raises(AutoReviewHumanOnly):
        _auto_resolution_service(settings).resolve(
            db=db,
            item_id=candidate.id,
            directive=directive,
        )
    assert db.get(ReviewItem, candidate.id).status == 'pending_review'


@pytest.mark.parametrize(
    'readiness_failure',
    ['row_permission_mismatch', 'row_key_mismatch', 'active_projection_anti_join'],
)
def test_key_admin_status_rechecks_projection_identity_and_server_side_coverage(
    auto_review_db_session: tuple[Session, Settings],
    readiness_failure: str,
) -> None:
    db, settings = auto_review_db_session
    item = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    row = db.scalars(
        select(TrustedKnowledgeFingerprint).order_by(
            TrustedKnowledgeFingerprint.knowledge_type,
            TrustedKnowledgeFingerprint.knowledge_id,
        )
    ).first()
    assert row is not None
    if readiness_failure == 'row_permission_mismatch':
        row.permission_level = 'restricted'
    elif readiness_failure == 'row_key_mismatch':
        row.fingerprint_key_material_verifier = 'f' * 64
    else:
        db.delete(row)
    db.commit()
    factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    status = AutoReviewKeyAdminService(
        session_factory=factory,
        settings=settings,
    ).status()
    assert status.runtime_ready is True
    assert status.projection_ready is False
    assert status.projection_rebuild_required is True


def test_empty_source_and_projection_are_ready_at_rebuild_status_and_auto_preflight(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    base_factory, settings = auto_review_postgres
    connection = base_factory.kw['bind'].connect()
    outer_transaction = connection.begin()
    factory = sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode='create_savepoint',
    )
    try:
        connection.execute(
            text(
                'TRUNCATE TABLE history_events, timeline_events, '
                'trusted_knowledge_fingerprints CASCADE'
            )
        )
        rebuilt = rebuild_trusted_fingerprint_projection(
            session_factory=factory,
            settings=settings,
        )
        assert rebuilt.ready is True
        assert rebuilt.source_count == rebuilt.projected_count == 0
        status = AutoReviewKeyAdminService(
            session_factory=factory,
            settings=settings,
        ).status()
        assert status.projection_ready is True
        assert status.projection_rebuild_required is False
        with factory() as db:
            candidate = _seed_c5_item(
                db,
                graph_version='company-memory-review-v2.1-auto-review',
            )
            _seed_completed_validation(db, candidate, settings=settings)
            directive = ReuseExistingPromotion(
                expected_type='history_event',
                expected_id=999_999,
                expected_claim_fingerprint='a' * 64,
                expected_companion_id=999_998,
                expected_companion_claim_fingerprint='b' * 64,
            )
            with pytest.raises(AutoReviewHumanOnly) as exc_info:
                _auto_resolution_service(settings).resolve(
                    db=db,
                    item_id=candidate.id,
                    directive=directive,
                )
            assert str(exc_info.value.__cause__) == (
                'Automatic review candidate claim is not exact'
            )
            assert db.scalar(select(TrustedKnowledgeFingerprint.id)) is None
    finally:
        outer_transaction.rollback()
        connection.close()


def test_large_projection_rebuild_status_and_auto_preflight_use_bounded_parameters(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    base_factory, settings = auto_review_postgres
    connection = base_factory.kw['bind'].connect()
    outer_transaction = connection.begin()
    factory = sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode='create_savepoint',
    )
    expected_count = 7_282
    projection_parameter_counts: list[int] = []

    def observe_projection_parameters(
        _connection,
        _cursor,
        statement: str,
        parameters,
        _context,
        _executemany,
    ) -> None:
        if 'expected_trusted_fingerprints' in statement:
            projection_parameter_counts.append(len(parameters))

    event.listen(
        base_factory.kw['bind'],
        'before_cursor_execute',
        observe_projection_parameters,
    )
    try:
        connection.execute(
            text(
                'TRUNCATE TABLE history_events, timeline_events, '
                'trusted_knowledge_fingerprints CASCADE'
            )
        )
        with factory() as db:
            for start in range(0, expected_count, 500):
                db.execute(
                    insert(TimelineEvent),
                    [
                        {
                            'project_key': None,
                            'title': f'bounded projection {ordinal}',
                            'result_summary': f'exact result {ordinal}',
                            'source_links': ['https://example.test/large'],
                            'source_snippets': ['bounded evidence'],
                            'confidence_score': 1.0,
                            'permission_level': 'internal',
                            'review_status': 'approved',
                            'source_review_item_id': None,
                        }
                        for ordinal in range(start, min(start + 500, expected_count))
                    ],
                )
            db.commit()
        rebuilt = rebuild_trusted_fingerprint_projection(
            session_factory=factory,
            settings=settings,
        )
        assert rebuilt.ready is True
        assert rebuilt.source_count == rebuilt.projected_count == expected_count
        status = AutoReviewKeyAdminService(
            session_factory=factory,
            settings=settings,
        ).status()
        assert status.projection_ready is True
        assert status.projection_rebuild_required is False
        with factory() as db:
            candidate = _seed_c5_item(
                db,
                graph_version='company-memory-review-v2.1-auto-review',
            )
            _seed_completed_validation(db, candidate, settings=settings)
            directive = ReuseExistingPromotion(
                expected_type='history_event',
                expected_id=999_999,
                expected_claim_fingerprint='a' * 64,
                expected_companion_id=999_998,
                expected_companion_claim_fingerprint='b' * 64,
            )
            with pytest.raises(AutoReviewHumanOnly) as exc_info:
                _auto_resolution_service(settings).resolve(
                    db=db,
                    item_id=candidate.id,
                    directive=directive,
                )
            assert str(exc_info.value.__cause__) == (
                'Automatic review candidate claim is not exact'
            )
        assert projection_parameter_counts
        assert max(projection_parameter_counts) <= 3
    finally:
        event.remove(
            base_factory.kw['bind'],
            'before_cursor_execute',
            observe_projection_parameters,
        )
        outer_transaction.rollback()
        connection.close()


@pytest.mark.parametrize(
    'drift_kind',
    ['title', 'claim', 'project', 'exact_scope', 'legacy_scope', 'extra'],
)
def test_complete_projection_drift_disables_status_and_auto_admission(
    auto_review_db_session: tuple[Session, Settings],
    drift_kind: str,
) -> None:
    db, settings = auto_review_db_session
    canonical, candidate, directive = _prepare_exact_auto_reaffirmation(
        db,
        settings=settings,
    )
    knowledge_type = directive.expected_type
    knowledge_id = directive.expected_id
    if drift_kind == 'legacy_scope':
        legacy = HistoryEvent(
            title=f'legacy projection {uuid4().hex}',
            reason='scope is intentionally unavailable',
            project_key=None,
            source_links=['https://example.test/legacy'],
            source_snippets=['legacy evidence'],
            confidence_score=1.0,
            permission_level='internal',
            review_status='approved',
            source_review_item_id=None,
        )
        db.add(legacy)
        db.commit()
        rebuilt = rebuild_trusted_fingerprint_projection(
            session_factory=sessionmaker(bind=db.get_bind(), expire_on_commit=False),
            settings=settings,
        )
        assert rebuilt.ready is True
        knowledge_type = 'history_event'
        knowledge_id = legacy.id
    _corrupt_projection_identity(
        db,
        drift_kind=drift_kind,
        knowledge_type=knowledge_type,
        knowledge_id=knowledge_id,
    )
    db.commit()

    factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    status = AutoReviewKeyAdminService(
        session_factory=factory,
        settings=settings,
    ).status()
    assert status.projection_ready is False
    assert status.projection_rebuild_required is True

    with pytest.raises(AutoReviewHumanOnly) as exc_info:
        _auto_resolution_service(settings).resolve(
            db=db,
            item_id=candidate.id,
            directive=directive,
        )
    assert str(exc_info.value.__cause__) == 'Automatic review projection is not ready'
    assert db.get(ReviewItem, candidate.id).status == 'pending_review'


@pytest.mark.parametrize(
    'drift_kind',
    ['title', 'claim', 'project', 'exact_scope', 'legacy_scope', 'extra'],
)
def test_rebuild_finalization_rejects_complete_projection_drift(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    drift_kind: str,
) -> None:
    factory, settings = auto_review_postgres
    with factory() as setup:
        canonical = _seed_c5_item(setup)
        ReviewTransitionService(settings=settings).transition(
            db=setup,
            item_id=canonical.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        directive = _reuse_directive_for_item(setup, canonical)
        knowledge_type = directive.expected_type
        knowledge_id = directive.expected_id
        if drift_kind == 'legacy_scope':
            legacy = HistoryEvent(
                title=f'legacy rebuild {uuid4().hex}',
                reason='scope is intentionally unavailable',
                project_key=None,
                source_links=['https://example.test/legacy'],
                source_snippets=['legacy evidence'],
                confidence_score=1.0,
                permission_level='internal',
                review_status='approved',
                source_review_item_id=None,
            )
            setup.add(legacy)
            setup.flush()
            knowledge_type = 'history_event'
            knowledge_id = legacy.id
        setup.commit()

    class CorruptingProjectionSession(Session):
        def execute(self, statement, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = super().execute(statement, *args, **kwargs)
            table = getattr(statement, 'table', None)
            if (
                statement.__class__.__name__ == 'Delete'
                and getattr(table, 'name', None) == 'trusted_knowledge_fingerprints'
            ):
                self.info['projection_replaced'] = True
            return result

        def scalar(self, statement, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.info.get('projection_replaced') and not self.info.get(
                'projection_corrupted'
            ):
                self.info['projection_corrupted'] = True
                _corrupt_projection_identity(
                    self,
                    drift_kind=drift_kind,
                    knowledge_type=knowledge_type,
                    knowledge_id=knowledge_id,
                )
            return super().scalar(statement, *args, **kwargs)

    corrupting_factory = sessionmaker(
        bind=factory.kw['bind'],
        class_=CorruptingProjectionSession,
        expire_on_commit=False,
    )
    result = rebuild_trusted_fingerprint_projection(
        session_factory=corrupting_factory,
        settings=settings,
    )
    assert result.ready is False
    status = AutoReviewKeyAdminService(
        session_factory=factory,
        settings=settings,
    ).status()
    assert status.projection_ready is False
    assert status.projection_rebuild_required is True


@pytest.mark.parametrize('owner_failure', ['unknown', 'identity_mismatch'])
def test_auto_resolution_is_human_only_when_current_owner_is_not_exact(
    auto_review_db_session: tuple[Session, Settings],
    owner_failure: str,
) -> None:
    db, settings = auto_review_db_session
    _, candidate, directive = _prepare_exact_auto_reaffirmation(
        db,
        settings=settings,
    )
    service = AutoReviewResolutionService(
        settings=settings,
        current_permission_resolver=(
            (lambda _subject_id: None)
            if owner_failure == 'unknown'
            else _current_permission_resolver(resolved_subject_id='different-owner')
        ),
    )
    with pytest.raises(AutoReviewHumanOnly):
        service.resolve(db=db, item_id=candidate.id, directive=directive)
    assert db.get(ReviewItem, candidate.id).status == 'pending_review'


def test_auto_resolution_rechecks_concurrent_owner_permission_removal(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = auto_review_postgres
    rebuilt = rebuild_trusted_fingerprint_projection(
        session_factory=factory,
        settings=settings,
    )
    assert rebuilt.ready is True
    with factory() as setup:
        _, candidate, directive = _prepare_exact_auto_reaffirmation(
            setup,
            settings=settings,
        )
        candidate_id = candidate.id
        source_id = setup.scalar(
            select(AgentWorkflowEvidenceRef.canonical_row_id)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == candidate.workflow_thread_id
            )
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        )
        assert source_id is not None

    current_levels = ['public', 'internal']
    completed = Event()
    source_lock_attempted = Event()
    outcomes: list[str] = []

    def observe_source_lock(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        if 'FROM sources' in statement and 'FOR SHARE' in statement:
            source_lock_attempted.set()

    def current_permission_resolver(subject_id: str):
        return subject_id, tuple(current_levels)

    def resolve_after_source_lock() -> None:
        with factory() as worker:
            try:
                AutoReviewResolutionService(
                    settings=settings,
                    current_permission_resolver=current_permission_resolver,
                ).resolve(
                    db=worker,
                    item_id=candidate_id,
                    directive=directive,
                )
                outcomes.append('approved')
            except AutoReviewHumanOnly:
                outcomes.append('human_only')
                worker.rollback()
            finally:
                completed.set()

    event.listen(factory.kw['bind'], 'before_cursor_execute', observe_source_lock)
    try:
        with factory() as source_writer:
            source_writer.scalar(
                select(Source).where(Source.id == source_id).with_for_update()
            )
            worker_thread = Thread(target=resolve_after_source_lock)
            worker_thread.start()
            assert source_lock_attempted.wait(5)
            assert completed.is_set() is False
            current_levels[:] = ['public']
            source_writer.rollback()
            worker_thread.join(5)
    finally:
        event.remove(factory.kw['bind'], 'before_cursor_execute', observe_source_lock)

    assert completed.is_set()
    assert outcomes == ['human_only']
    with factory() as check:
        assert check.get(ReviewItem, candidate_id).status == 'pending_review'


@pytest.mark.parametrize(
    'drift_kind',
    ['source', 'source_permission', 'workflow', 'current_document'],
)
def test_auto_approval_rechecks_concurrent_canonical_drift_after_total_order_locks(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    drift_kind: str,
) -> None:
    factory, settings = auto_review_postgres
    rebuilt = rebuild_trusted_fingerprint_projection(
        session_factory=factory,
        settings=settings,
    )
    assert rebuilt.ready is True
    with factory() as setup:
        canonical = _seed_c5_item(setup)
        ReviewTransitionService(settings=settings).transition(
            db=setup,
            item_id=canonical.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        directive = _reuse_directive_for_item(setup, canonical)
        candidate = _seed_c5_item(
            setup,
            graph_version='company-memory-review-v2.1-auto-review',
            payload=dict(canonical.payload),
        )
        _seed_completed_validation(setup, candidate, settings=settings)
        candidate_id = candidate.id
        workflow_id = candidate.workflow_thread_id
        ref = setup.scalars(
            select(AgentWorkflowEvidenceRef)
            .where(AgentWorkflowEvidenceRef.workflow_thread_id == workflow_id)
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).first()
        assert ref is not None
        source_id = ref.canonical_row_id
        document = setup.scalar(select(Document).where(Document.source_id == source_id))
        assert document is not None
        document_id = document.id

    completed = Event()
    outcomes: list[str] = []

    def approve_after_lock() -> None:
        with factory() as worker:
            try:
                _auto_resolution_service(settings).resolve(
                    db=worker,
                    item_id=candidate_id,
                    directive=directive,
                )
            except AutoReviewHumanOnly:
                outcomes.append('human_only')
                worker.rollback()
            finally:
                completed.set()

    with factory() as writer:
        if drift_kind == 'source':
            source = writer.get(Source, source_id)
            assert source is not None
            source.server_content_signature = '9' * 64
        elif drift_kind == 'source_permission':
            source = writer.get(Source, source_id)
            assert source is not None
            source.permission_level = 'restricted'
        elif drift_kind == 'workflow':
            workflow = writer.get(AgentWorkflowThread, workflow_id)
            assert workflow is not None
            workflow.evidence_version_hash = '8' * 64
        else:
            locked_document = writer.get(Document, document_id)
            assert locked_document is not None
            locked_document.current_document_version_id = None
        writer.flush()
        worker_thread = Thread(target=approve_after_lock)
        worker_thread.start()
        assert completed.wait(0.15) is False
        writer.commit()
        worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['human_only']
    with factory() as check:
        assert check.get(ReviewItem, candidate_id).status == 'pending_review'


@pytest.mark.parametrize('drift_kind', ['permission', 'workflow_hash'])
def test_human_approval_rechecks_concurrent_evidence_identity_drift(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    drift_kind: str,
) -> None:
    factory, settings = auto_review_postgres
    with factory() as setup:
        candidate = _seed_c5_item(setup)
        candidate_id = candidate.id
        ref = setup.scalars(
            select(AgentWorkflowEvidenceRef)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == candidate.workflow_thread_id
            )
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).first()
        assert ref is not None
        source_id = ref.canonical_row_id
        workflow_id = candidate.workflow_thread_id

    completed = Event()
    outcomes: list[str] = []

    def approve_after_lock() -> None:
        with factory() as worker:
            try:
                ReviewTransitionService(settings=settings).transition(
                    db=worker,
                    item_id=candidate_id,
                    action='approve',
                    actor=human_review_actor(USERS['admin']),
                )
            except Exception as exc:  # noqa: BLE001 - records exact race outcome
                outcomes.append(type(exc).__name__)
                worker.rollback()
            finally:
                completed.set()

    with factory() as writer:
        if drift_kind == 'permission':
            source = writer.get(Source, source_id)
            assert source is not None
            source.permission_level = 'restricted'
        else:
            workflow = writer.get(AgentWorkflowThread, workflow_id)
            assert workflow is not None
            workflow.evidence_version_hash = '7' * 64
        writer.flush()
        worker_thread = Thread(target=approve_after_lock)
        worker_thread.start()
        assert completed.wait(0.15) is False
        writer.commit()
        worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['CanonicalEvidenceDrift']
    with factory() as check:
        assert check.get(ReviewItem, candidate_id).status == 'pending_review'


def test_legacy_human_approval_demotes_ready_projection_without_blocking(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db, settings = auto_review_db_session
    item = _item('history_event')
    db.add(item)
    db.commit()
    state = db.scalars(select(TrustedKnowledgeFingerprintProjectionState)).one()
    assert state.ready is True
    result = ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    db.commit()
    db.refresh(state)
    assert result.status == 'approved'
    assert state.ready is False
    assert state.rebuild_required is True


def test_human_projection_delta_demotes_when_a_preexisting_active_row_is_missing(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db, settings = auto_review_db_session
    first = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=first.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    db.commit()
    missing = db.scalars(
        select(TrustedKnowledgeFingerprint).where(
            TrustedKnowledgeFingerprint.knowledge_type == 'history_event'
        )
    ).first()
    assert missing is not None
    db.delete(missing)
    db.commit()
    state = db.scalars(select(TrustedKnowledgeFingerprintProjectionState)).one()
    assert state.ready is True

    second = _seed_c5_item(
        db,
        payload={
            'title': 'Independent human projection delta',
            'reason': 'This row must not conceal the missing preexisting row.',
        },
    )
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=second.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    db.commit()
    db.refresh(state)
    assert state.ready is False
    assert state.rebuild_required is True


def test_auto_call_locks_precede_review_item_lock(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = auto_review_postgres
    rebuild_trusted_fingerprint_projection(session_factory=factory, settings=settings)
    with factory() as setup:
        canonical = _seed_c5_item(setup)
        ReviewTransitionService(settings=settings).transition(
            db=setup,
            item_id=canonical.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        setup.commit()
        directive = _reuse_directive_for_item(setup, canonical)
        candidate = _seed_c5_item(
            setup,
            graph_version='company-memory-review-v2.1-auto-review',
            payload=dict(canonical.payload),
        )
        validation = _seed_completed_validation(setup, candidate, settings=settings)
        item_id = candidate.id
        validation_call_id = validation.validation_call_id

    completed = Event()
    outcomes: list[str] = []

    def approve() -> None:
        with factory() as worker:
            try:
                result = _auto_resolution_service(settings).resolve(
                    db=worker,
                    item_id=item_id,
                    directive=directive,
                )
                worker.commit()
                outcomes.append(result.status)
            finally:
                completed.set()

    with factory() as call_writer:
        call_writer.scalar(
            select(AutoReviewValidationCall)
            .where(AutoReviewValidationCall.id == validation_call_id)
            .with_for_update()
        )
        worker_thread = Thread(target=approve)
        worker_thread.start()
        assert completed.wait(0.15) is False
        with factory() as probe:
            locked_item = probe.scalar(
                select(ReviewItem)
                .where(ReviewItem.id == item_id)
                .with_for_update(nowait=True)
            )
            assert locked_item is not None
            probe.rollback()
        call_writer.rollback()
        worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['approved']


def test_auto_approval_rechecks_completed_validation_multiplicity_after_locator(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, settings = auto_review_postgres
    rebuild_trusted_fingerprint_projection(session_factory=factory, settings=settings)
    with factory() as setup:
        canonical = _seed_c5_item(setup)
        ReviewTransitionService(settings=settings).transition(
            db=setup,
            item_id=canonical.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        setup.commit()
        directive = _reuse_directive_for_item(setup, canonical)
        candidate = _seed_c5_item(
            setup,
            graph_version='company-memory-review-v2.1-auto-review',
            payload=dict(canonical.payload),
        )
        validation = _seed_completed_validation(setup, candidate, settings=settings)
        item_id = candidate.id
        validation_id = validation.id

    located = Event()
    proceed = Event()
    completed = Event()
    outcomes: list[str] = []
    original = ReviewTransitionService._discover_auto_approval_locator

    def pause_after_locator(self, db, *, preview, actor):
        locator = original(self, db, preview=preview, actor=actor)
        located.set()
        assert proceed.wait(5)
        return locator

    monkeypatch.setattr(
        ReviewTransitionService,
        '_discover_auto_approval_locator',
        pause_after_locator,
    )

    def approve() -> None:
        with factory() as worker:
            try:
                _auto_resolution_service(settings).resolve(
                    db=worker,
                    item_id=item_id,
                    directive=directive,
                )
            except AutoReviewHumanOnly:
                worker.rollback()
                outcomes.append('human_only')
            else:
                worker.commit()
                outcomes.append('approved')
            finally:
                completed.set()

    worker_thread = Thread(target=approve)
    worker_thread.start()
    assert located.wait(5)
    with factory() as inserter:
        original_validation = inserter.get(AutoReviewValidation, validation_id)
        assert original_validation is not None
        _seed_additional_completed_validation(inserter, original_validation)
    proceed.set()
    worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['human_only']
    with factory() as check:
        assert check.get(ReviewItem, item_id).status == 'pending_review'


def test_auto_approval_rechecks_validation_phantom_after_initial_call_locks(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, settings = auto_review_postgres
    rebuild_trusted_fingerprint_projection(session_factory=factory, settings=settings)
    with factory() as setup:
        claim_payload = {
            'title': f'validation-phantom-{uuid4().hex}',
            'reason': 'validation phantom boundary',
        }
        canonical = _seed_c5_item(setup, payload=claim_payload)
        ReviewTransitionService(settings=settings).transition(
            db=setup,
            item_id=canonical.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        setup.commit()
        directive = _reuse_directive_for_item(setup, canonical)
        candidate = _seed_c5_item(
            setup,
            graph_version='company-memory-review-v2.1-auto-review',
            payload=claim_payload,
        )
        validation = _seed_completed_validation(setup, candidate, settings=settings)
        item_id = candidate.id
        validation_id = validation.id

    calls_locked = Event()
    proceed = Event()
    completed = Event()
    outcomes: list[str] = []
    original = ReviewTransitionService._lock_and_require_auto_calls

    def pause_after_initial_call_locks(self, db, *, locator, safety):
        locked_validation_id = original(
            self,
            db,
            locator=locator,
            safety=safety,
        )
        calls_locked.set()
        assert proceed.wait(5)
        return locked_validation_id

    monkeypatch.setattr(
        ReviewTransitionService,
        '_lock_and_require_auto_calls',
        pause_after_initial_call_locks,
    )

    def approve() -> None:
        with factory() as worker:
            try:
                _auto_resolution_service(settings).resolve(
                    db=worker,
                    item_id=item_id,
                    directive=directive,
                )
            except AutoReviewHumanOnly:
                worker.rollback()
                outcomes.append('human_only')
            else:
                worker.commit()
                outcomes.append('approved')
            finally:
                completed.set()

    worker_thread = Thread(target=approve)
    worker_thread.start()
    assert calls_locked.wait(5)
    with factory() as inserter:
        original_validation = inserter.get(AutoReviewValidation, validation_id)
        assert original_validation is not None
        _seed_additional_completed_validation(inserter, original_validation)
    proceed.set()
    worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['human_only']
    with factory() as check:
        assert check.get(ReviewItem, item_id).status == 'pending_review'


def test_reuse_target_lock_precedes_current_document_lock(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.knowledge import trusted_provenance

    factory, settings = auto_review_postgres
    rebuild_trusted_fingerprint_projection(session_factory=factory, settings=settings)
    with factory() as setup:
        canonical = _seed_c5_item(setup)
        ReviewTransitionService(settings=settings).transition(
            db=setup,
            item_id=canonical.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        setup.commit()
        directive = _reuse_directive_for_item(setup, canonical)
        candidate = _seed_c5_item(
            setup,
            graph_version='company-memory-review-v2.1-auto-review',
            payload=dict(canonical.payload),
        )
        _seed_completed_validation(setup, candidate, settings=settings)
        item_id = candidate.id
        ref = setup.scalars(
            select(AgentWorkflowEvidenceRef)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == candidate.workflow_thread_id
            )
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).first()
        assert ref is not None
        document = setup.scalar(
            select(Document).where(Document.source_id == ref.canonical_row_id)
        )
        assert document is not None
        document_id = document.id

    completed = Event()
    target_locked = Event()
    outcomes: list[str] = []
    original_effect_from_target = trusted_provenance._effect_from_target

    def observe_target_lock(*args, **kwargs):  # type: ignore[no-untyped-def]
        effect = original_effect_from_target(*args, **kwargs)
        if (
            kwargs.get('knowledge_type') == directive.expected_type
            and kwargs.get('knowledge_id') == directive.expected_id
            and kwargs.get('lock') is True
        ):
            target_locked.set()
        return effect

    monkeypatch.setattr(
        trusted_provenance,
        '_effect_from_target',
        observe_target_lock,
    )

    def approve() -> None:
        with factory() as worker:
            try:
                result = _auto_resolution_service(settings).resolve(
                    db=worker,
                    item_id=item_id,
                    directive=directive,
                )
                worker.commit()
                outcomes.append(result.status)
            finally:
                completed.set()

    with factory() as document_writer:
        document_writer.scalar(
            select(Document)
            .where(Document.id == document_id)
            .with_for_update()
        )
        worker_thread = Thread(target=approve)
        worker_thread.start()
        assert target_locked.wait(5)
        assert completed.is_set() is False
        with factory() as probe:
            with pytest.raises(DBAPIError, match='could not obtain lock'):
                probe.scalar(
                    select(HistoryEvent)
                    .where(HistoryEvent.id == directive.expected_id)
                    .with_for_update(nowait=True)
                )
            probe.rollback()
        document_writer.rollback()
        worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['approved']


def test_nonapproval_source_lock_precedes_review_item_lock(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = auto_review_postgres
    with factory() as setup:
        candidate = _seed_c5_item(setup)
        item_id = candidate.id
        ref = setup.scalars(
            select(AgentWorkflowEvidenceRef)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == candidate.workflow_thread_id
            )
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).first()
        assert ref is not None
        source_id = ref.canonical_row_id

    completed = Event()
    outcomes: list[str] = []

    def reject() -> None:
        with factory() as worker:
            try:
                result = ReviewTransitionService(settings=settings).transition(
                    db=worker,
                    item_id=item_id,
                    action='reject',
                    actor=human_review_actor(USERS['admin']),
                )
                worker.commit()
                outcomes.append(result.status)
            finally:
                completed.set()

    with factory() as source_writer:
        source_writer.scalar(
            select(Source).where(Source.id == source_id).with_for_update()
        )
        worker_thread = Thread(target=reject)
        worker_thread.start()
        assert completed.wait(0.15) is False
        with factory() as probe:
            locked_item = probe.scalar(
                select(ReviewItem)
                .where(ReviewItem.id == item_id)
                .with_for_update(nowait=True)
            )
            assert locked_item is not None
            probe.rollback()
        source_writer.rollback()
        worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['rejected']


def test_batch_nonapproval_source_lock_precedes_review_item_lock(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = auto_review_postgres
    with factory() as setup:
        candidate = _seed_c5_item(setup)
        item_id = candidate.id
        ref = setup.scalars(
            select(AgentWorkflowEvidenceRef)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == candidate.workflow_thread_id
            )
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).first()
        assert ref is not None
        source_id = ref.canonical_row_id

    completed = Event()
    outcomes: list[str] = []

    def reject_many() -> None:
        with factory() as worker:
            try:
                result = ReviewTransitionService(settings=settings).transition_many(
                    db=worker,
                    item_ids=[item_id],
                    action='reject',
                    actor=human_review_actor(USERS['admin']),
                )
                worker.commit()
                outcomes.extend(item.status for item in result.results)
            finally:
                completed.set()

    with factory() as source_writer:
        source_writer.scalar(
            select(Source).where(Source.id == source_id).with_for_update()
        )
        worker_thread = Thread(target=reject_many)
        worker_thread.start()
        assert completed.wait(0.15) is False
        with factory() as probe:
            locked_item = probe.scalar(
                select(ReviewItem)
                .where(ReviewItem.id == item_id)
                .with_for_update(nowait=True)
            )
            assert locked_item is not None
            probe.rollback()
        source_writer.rollback()
        worker_thread.join(5)

    assert completed.is_set()
    assert outcomes == ['rejected']


def test_visible_nonmatching_title_bucket_forces_conflict() -> None:
    claim_a = _effect_fingerprint(
        'history_event', {'title': 'Same title', 'reason': 'A'}
    )
    claim_b = _effect_fingerprint(
        'history_event', {'title': 'Same title', 'reason': 'B'}
    )
    assert claim_a != claim_b
    assert trusted_title_collision_bucket(
        item_type='history_event',
        normalized_title='Same title',
        settings=_settings(),
    ) == trusted_title_collision_bucket(
        item_type='history_event',
        normalized_title='  Same\ttitle ',
        settings=_settings(),
    )


def test_hidden_collision_guard_returns_only_exists_and_never_loads_identity() -> None:
    statement = build_hidden_collision_exists_statement(
        knowledge_type='history_event',
        security_scope_id='scope-a',
        project_scope_hmac='a' * 64,
        normalized_title_bucket_hmac='b' * 64,
        visible_permission_levels=('public',),
    )
    sql = str(statement.compile(dialect=postgresql.dialect())).lower()
    assert 'select exists' in sql
    assert 'count(' not in sql
    select_clause = sql.split('from trusted_knowledge_fingerprints', 1)[0]
    assert 'knowledge_id' not in select_clause
    assert 'normalized_claim_fingerprint' not in select_clause
    assert 'permission_level' not in select_clause


def test_legacy_unknown_scope_is_not_reused() -> None:
    row = _snapshot()
    unknown = replace(
        row,
        scope_resolution='legacy_unknown',
        security_scope_id=None,
        normalized_claim_fingerprint=None,
    )
    assert unknown.normalized_claim_fingerprint is None
    assert unknown.scope_resolution != 'exact'


def test_fingerprint_projection_backfills_known_scope_and_marks_unknown_scope_collision_only() -> None:
    exact = _snapshot()
    unknown = replace(
        exact,
        knowledge_id=2,
        scope_resolution='legacy_unknown',
        security_scope_id=None,
        normalized_claim_fingerprint=None,
    )
    assert exact.security_scope_id == 'scope-a'
    assert exact.normalized_claim_fingerprint is not None
    assert unknown.security_scope_id is None
    assert unknown.normalized_claim_fingerprint is None


def test_missing_or_stale_projection_disables_auto_review_until_rebuilt() -> None:
    runtime = AutoReviewRuntimeKeyState(
        component='auto_review_trust_promotion',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='a' * 64,
        generation=1,
        ready=True,
    )
    assert projection_identity_ready(runtime, None, missing_active_row=False) is False
    projection = TrustedKnowledgeFingerprintProjectionState(
        component='trusted_knowledge_fingerprints',
        projection_schema_version='trusted-fingerprint-projection:v1',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='a' * 64,
        generation=1,
        ready=False,
        rebuild_required=True,
        source_active_count=1,
        projected_active_count=0,
        source_checksum='1'.zfill(64),
        projected_checksum='0' * 64,
    )
    assert projection_identity_ready(runtime, projection, missing_active_row=False) is False


def test_unprojected_hidden_row_barrier_forces_zero_call_human_only(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db_session, settings = auto_review_db_session
    runtime = db_session.scalars(select(AutoReviewRuntimeKeyState)).one()
    projection = db_session.scalars(
        select(TrustedKnowledgeFingerprintProjectionState)
    ).one()
    db_session.add(
        HistoryEvent(
            title='unprojected',
            reason='must disable automatic review',
            project_key=None,
            source_links=['https://example.test/unprojected'],
            source_snippets=['evidence'],
            confidence_score=1.0,
            permission_level='internal',
            review_status='approved',
            source_review_item_id=None,
        )
    )
    db_session.commit()
    missing = bool(
        db_session.scalar(
            build_missing_active_projection_exists_statement(
                expected_rows=source_projection_snapshots(
                    db_session,
                    settings=settings,
                    fingerprint_key_material_verifier=(
                        runtime.fingerprint_key_material_verifier
                    ),
                ),
            )
        )
    )
    assert missing is True
    assert projection_identity_ready(runtime, projection, missing_active_row=missing) is False


def test_projection_ready_flips_only_after_locked_anti_join_and_count_checksum_match() -> None:
    runtime = AutoReviewRuntimeKeyState(
        component='auto_review_trust_promotion',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='a' * 64,
        generation=2,
        ready=True,
    )
    projection = TrustedKnowledgeFingerprintProjectionState(
        component='trusted_knowledge_fingerprints',
        projection_schema_version='trusted-fingerprint-projection:v1',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='a' * 64,
        generation=2,
        ready=True,
        rebuild_required=False,
        source_active_count=2,
        projected_active_count=2,
        source_checksum='b' * 64,
        projected_checksum='b' * 64,
    )
    assert projection_identity_ready(runtime, projection, missing_active_row=False)
    projection.projected_active_count = 1
    assert not projection_identity_ready(runtime, projection, missing_active_row=False)


def test_projection_summary_delta_is_exact_for_insert_update_and_remove() -> None:
    zero = ProjectionSummary(active_count=0, checksum_hex='0' * 64)
    first = projection_row_digest(_snapshot(knowledge_id=1), settings=_settings())
    second = projection_row_digest(_snapshot(knowledge_id=1, claim='e' * 64), settings=_settings())
    inserted = ProjectionSummaryDelta(old_digest=None, new_digest=first).apply(zero)
    assert inserted == ProjectionSummary(active_count=1, checksum_hex=first)
    updated = ProjectionSummaryDelta(old_digest=first, new_digest=second).apply(inserted)
    assert updated == ProjectionSummary(active_count=1, checksum_hex=second)
    removed = ProjectionSummaryDelta(old_digest=second, new_digest=None).apply(updated)
    assert removed == zero


def test_provenance_and_projection_rows_snapshot_key_version_and_material_verifier() -> None:
    for table in (
        TrustedKnowledgeApprovalLink.__table__,
        TrustedKnowledgeEvidenceLink.__table__,
        TrustedKnowledgeFingerprint.__table__,
    ):
        assert table.c.fingerprint_key_version.nullable is False
        assert table.c.fingerprint_key_material_verifier.nullable is False


def test_rotation_waits_for_inflight_keyed_mutation_before_advancing_generation() -> None:
    engine = create_engine('sqlite:///:memory:', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    entered = Event()
    release = Event()
    rotated = Event()

    def mutation() -> None:
        with factory() as db, KeyedMutationGuard.generation_barrier(db):
            entered.set()
            release.wait(2)

    def rotation() -> None:
        entered.wait(2)
        with factory() as db, KeyedMutationGuard.generation_barrier(db, exclusive=True):
            rotated.set()

    first = Thread(target=mutation)
    second = Thread(target=rotation)
    first.start()
    second.start()
    assert entered.wait(1)
    assert not rotated.wait(0.1)
    release.set()
    first.join(2)
    second.join(2)
    assert rotated.is_set()


def test_every_postgres_guard_callsite_uses_the_same_fixed_lock_ids_and_order() -> None:
    assert AUTO_REVIEW_KEY_GENERATION_LOCK_ID == 1066041229503628369
    assert TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID == -2972884933094306491
    with pytest.raises(TypeError):
        KeyGenerationLockedContext(
            session_identity=1,
            generation=1,
            key_version='v1',
            material_verifier='a' * 64,
        )
    for model in (AutoReviewExtractionCall, AutoReviewValidationCall):
        compiled = str(
            build_nonterminal_provider_call_lock_statement(model).compile(
                dialect=postgresql.dialect()
            )
        )
        assert f'ORDER BY {model.__tablename__}.id FOR UPDATE' in compiled


def test_disabled_sqlite_human_approve_uses_process_guard_and_keeps_projection_not_ready() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        settings = _settings()
        db.add(
            TrustedKnowledgeFingerprintProjectionState(
                component='trusted_knowledge_fingerprints',
                projection_schema_version='trusted-fingerprint-projection:v1',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                    settings.agent_runtime_fingerprint_secret
                ),
                generation=1,
                ready=False,
                rebuild_required=True,
            )
        )
        db.commit()
        item = _seed_c5_item(db)
        result = ReviewTransitionService(settings=settings).transition(
            db=db,
            item_id=item.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        state = db.scalars(select(TrustedKnowledgeFingerprintProjectionState)).one()
        assert result.status == 'approved'
        assert state.ready is False
        assert state.rebuild_required is True


def test_healthy_human_approval_updates_summary_delta_and_preserves_ready(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db, settings = auto_review_db_session
    state = db.scalars(select(TrustedKnowledgeFingerprintProjectionState)).one()
    before_count = state.source_active_count
    before_checksum = state.source_checksum
    item = _seed_c5_item(db)
    result = ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    db.refresh(state)
    assert result.status == 'approved'
    assert state.ready is True
    assert state.rebuild_required is False
    assert state.source_active_count == before_count + 2
    assert state.projected_active_count == state.source_active_count
    assert state.source_checksum == state.projected_checksum
    assert state.source_checksum != before_checksum


def test_unprojectable_human_approval_demotes_to_rebuild_required_without_blocking_human(
    auto_review_db_session: tuple[Session, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.knowledge import trusted_fingerprint_projection as projection

    db, settings = auto_review_db_session
    item = _seed_c5_item(db)

    def refuse_projection(**_: object) -> str:
        raise ValueError('bounded projection failure')

    monkeypatch.setattr(projection, 'trusted_title_collision_bucket', refuse_projection)
    result = ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    state = db.scalars(select(TrustedKnowledgeFingerprintProjectionState)).one()
    assert result.status == 'approved'
    assert db.get(ReviewItem, item.id).status == 'approved'
    assert state.ready is False
    assert state.rebuild_required is True


def test_human_approve_succeeds_while_projection_not_ready_and_auto_review_stays_disabled(
    db_session: Session,
) -> None:
    settings = _settings()
    item = _seed_c5_item(db_session)
    result = ReviewTransitionService(settings=settings).transition(
        db=db_session,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    assert result.status == 'approved'
    assert db_session.get(ReviewItem, item.id).resolution_source == 'human'


def test_post_migration_v20_human_approval_writes_complete_explicit_provenance(
    db_session: Session,
) -> None:
    item = _seed_c5_item(db_session, graph_version='company-memory-review-v2.0')
    ReviewTransitionService(settings=_settings()).transition(
        db=db_session,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    links = tuple(
        db_session.scalars(
            select(TrustedKnowledgeApprovalLink).where(
                TrustedKnowledgeApprovalLink.review_item_id == item.id
            )
        ).all()
    )
    assert len(links) == 2
    for link in links:
        assert len(
            db_session.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id == link.id
                )
            ).all()
        ) == 2


def test_pre_migration_unbound_v20_human_approval_uses_legacy_base_without_childless_link(
    db_session: Session,
) -> None:
    item = _item('history_event')
    db_session.add(item)
    db_session.commit()
    ReviewTransitionService(settings=_settings()).transition(
        db=db_session,
        item_id=item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    history = db_session.scalars(select(HistoryEvent)).one()
    assert has_legacy_human_base(
        db_session,
        knowledge_type='history_event',
        knowledge_id=history.id,
    )
    assert db_session.scalars(select(TrustedKnowledgeApprovalLink)).all() == []


def test_reuse_history_requires_exactly_one_locked_matching_companion_bundle(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db, settings = auto_review_db_session
    first_item = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=first_item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    links = tuple(
        db.scalars(
            select(TrustedKnowledgeApprovalLink)
            .where(TrustedKnowledgeApprovalLink.review_item_id == first_item.id)
            .order_by(TrustedKnowledgeApprovalLink.promotion_effect_kind.desc())
        ).all()
    )
    primary, companion = links
    candidate = _seed_c5_item(db, payload=dict(first_item.payload))
    directive = ReuseExistingPromotion(
        expected_type='history_event',
        expected_id=primary.knowledge_id,
        expected_claim_fingerprint=primary.claim_fingerprint,
        expected_companion_id=companion.knowledge_id,
        expected_companion_claim_fingerprint=companion.claim_fingerprint,
    )
    bundle = validate_reuse_bundle(
        db,
        item=candidate,
        directive=directive,
        settings=settings,
    )
    assert bundle.record_ids == (primary.knowledge_id,)
    assert bundle.timeline_ids == (companion.knowledge_id,)
    with pytest.raises(TrustedProvenanceMismatch):
        validate_reuse_bundle(
            db,
            item=candidate,
            directive=replace(
                directive,
                expected_companion_id=None,
                expected_companion_claim_fingerprint=None,
            ),
            settings=settings,
        )
    other_origin = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=other_origin.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    historical_version = 'historical-v0'
    historical_verifier = 'e' * 64
    extra_primary = TrustedKnowledgeApprovalLink(
        knowledge_type='history_event',
        knowledge_id=primary.knowledge_id,
        review_item_id=other_origin.id,
        security_scope_id='scope-a',
        promotion_effect_kind='primary',
        resolution_source='human',
        claim_fingerprint='d' * 64,
        permission_level='internal',
        fingerprint_key_version=historical_version,
        fingerprint_key_material_verifier=historical_verifier,
        active=True,
    )
    db.add(extra_primary)
    db.flush()
    other_bindings = tuple(
        db.scalars(
            select(ReviewItemEvidenceRef).where(
                ReviewItemEvidenceRef.review_item_id == other_origin.id
            )
        ).all()
    )
    other_refs = {
        ref.id: ref
        for ref in db.scalars(
            select(AgentWorkflowEvidenceRef).where(
                AgentWorkflowEvidenceRef.id.in_(
                    [binding.workflow_evidence_ref_id for binding in other_bindings]
                )
            )
        ).all()
    }
    db.add_all(
        [
            TrustedKnowledgeEvidenceLink(
                approval_link_id=extra_primary.id,
                canonical_source_kind=other_refs[
                    binding.workflow_evidence_ref_id
                ].canonical_source_type,
                canonical_source_id=str(
                    other_refs[binding.workflow_evidence_ref_id].canonical_row_id
                ),
                canonical_version_or_signature=(
                    other_refs[binding.workflow_evidence_ref_id].external_revision
                    or other_refs[binding.workflow_evidence_ref_id].content_signature
                ),
                evidence_hash=f'{binding.id:x}'.zfill(64),
                fingerprint_key_version=historical_version,
                fingerprint_key_material_verifier=historical_verifier,
            )
            for binding in other_bindings
        ]
    )
    db.flush()
    with pytest.raises(TrustedProvenanceMismatch, match='origins are not exact'):
        validate_reuse_bundle(
            db,
            item=candidate,
            directive=directive,
            settings=settings,
        )


def test_missing_ambiguous_or_mismatched_companion_is_human_only(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db_session, settings = auto_review_db_session
    first_item = _seed_c5_item(db_session)
    ReviewTransitionService(settings=settings).transition(
        db=db_session,
        item_id=first_item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    primary = db_session.scalar(
        select(TrustedKnowledgeApprovalLink).where(
            TrustedKnowledgeApprovalLink.review_item_id == first_item.id,
            TrustedKnowledgeApprovalLink.promotion_effect_kind == 'primary',
        )
    )
    companion = db_session.scalar(
        select(TrustedKnowledgeApprovalLink).where(
            TrustedKnowledgeApprovalLink.review_item_id == first_item.id,
            TrustedKnowledgeApprovalLink.promotion_effect_kind == 'companion',
        )
    )
    assert primary is not None and companion is not None
    timeline = db_session.get(TimelineEvent, companion.knowledge_id)
    timeline.result_summary = 'mismatched persisted companion'
    db_session.commit()
    second_item = _seed_c5_item(
        db_session,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(first_item.payload),
    )
    _seed_completed_validation(db_session, second_item, settings=settings)
    with pytest.raises(AutoReviewHumanOnly):
        _auto_resolution_service(settings).resolve(
            db=db_session,
            item_id=second_item.id,
            directive=ReuseExistingPromotion(
                expected_type='history_event',
                expected_id=primary.knowledge_id,
                expected_claim_fingerprint=primary.claim_fingerprint,
                expected_companion_id=companion.knowledge_id,
                expected_companion_claim_fingerprint=companion.claim_fingerprint,
            ),
        )
    assert db_session.get(ReviewItem, second_item.id).status == 'pending_review'


def test_reaffirmed_bundle_revoke_is_atomic_for_primary_and_companion(
    auto_review_db_session: tuple[Session, Settings],
) -> None:
    db, settings = auto_review_db_session
    first_item = _seed_c5_item(db)
    ReviewTransitionService(settings=settings).transition(
        db=db,
        item_id=first_item.id,
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )
    directive = _reuse_directive_for_item(db, first_item)
    reaffirming_item = _seed_c5_item(
        db,
        graph_version='company-memory-review-v2.1-auto-review',
        payload=dict(first_item.payload),
    )
    _seed_completed_validation(db, reaffirming_item, settings=settings)
    result = _auto_resolution_service(settings).resolve(
        db=db,
        item_id=reaffirming_item.id,
        directive=directive,
    )
    assert result.promotion is not None
    assert result.promotion.effect == 'reaffirmed'
    links = tuple(
        db.scalars(
            select(TrustedKnowledgeApprovalLink).where(
                TrustedKnowledgeApprovalLink.review_item_id == reaffirming_item.id
            )
        ).all()
    )
    assert {link.promotion_effect_kind for link in links} == {'primary', 'companion'}
    assert all(link.active for link in links)
    assert all(
        len(
            db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id == link.id
                )
            ).all()
        )
        == 2
        for link in links
    )


class _OwnedRuntimeProbe:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        dispose_failure: Exception | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.dispose_failure = dispose_failure
        self.dispose_calls = 0

    def dispose(self) -> None:
        self.dispose_calls += 1
        if self.dispose_failure is not None:
            raise self.dispose_failure


def _install_owned_runtime(
    monkeypatch: pytest.MonkeyPatch,
    key_admin: object,
    runtime: _OwnedRuntimeProbe,
    initialize_calls: list[str],
) -> None:
    def initialize(database_url: str) -> _OwnedRuntimeProbe:
        initialize_calls.append(database_url)
        return runtime

    if hasattr(key_admin, '_load_database_contract'):
        contract = SimpleNamespace(
            initialize=initialize,
            configuration_error=DatabaseConfigurationError,
            initialization_error=DatabaseInitializationError,
        )
        monkeypatch.setattr(key_admin, '_load_database_contract', lambda: contract)
        return

    def initialize_legacy() -> sessionmaker:
        initialize('legacy-db-session-path')
        return runtime.session_factory

    monkeypatch.setattr(key_admin, '_initialize_cli_storage', initialize_legacy)


def _install_database_boundary_counter(
    monkeypatch: pytest.MonkeyPatch,
    key_admin: object,
    calls: list[str],
) -> None:
    def boundary_called() -> object:
        calls.append('called')
        raise AssertionError('database boundary must not run')

    if hasattr(key_admin, '_load_database_contract'):
        monkeypatch.setattr(key_admin, '_load_database_contract', boundary_called)
    else:
        monkeypatch.setattr(key_admin, '_initialize_cli_storage', boundary_called)


def _valid_cli_key_ring() -> FingerprintKeyRing:
    return FingerprintKeyRing(
        current_version='v1',
        current_secret=SecretStr('c' * 48),
        next_version='v2',
        next_secret=SecretStr('n' * 48),
    )


def _install_cli_service_probe(
    monkeypatch: pytest.MonkeyPatch,
    key_admin: object,
    *,
    command_calls: list[str] | None = None,
    construction_failure: Exception | None = None,
    command_failure: Exception | None = None,
    ready: bool = True,
) -> None:
    calls = command_calls if command_calls is not None else []

    class _ServiceProbe:
        def __init__(self, **_kwargs: object) -> None:
            if construction_failure is not None:
                raise construction_failure

        def _record(self, command: str) -> None:
            calls.append(command)
            if command_failure is not None:
                raise command_failure

        def status(self) -> object:
            self._record('status')
            return SimpleNamespace(
                runtime_generation=7,
                runtime_version='v1',
                runtime_ready=ready,
                projection_ready=ready,
                projection_rebuild_required=not ready,
                nonterminal_extraction_count=2,
                nonterminal_validation_count=3,
            )

        def bootstrap(self) -> object:
            self._record('bootstrap')
            return SimpleNamespace(
                schema_available=True,
                initialized=True,
                ready=ready,
            )

        def rebuild_trusted_fingerprint_projection(self) -> object:
            self._record('rebuild')
            return self._rebuild_result()

        def rotate_fingerprint_key(self, **_kwargs: object) -> object:
            self._record('rotate')
            return self._rebuild_result()

        @staticmethod
        def _rebuild_result() -> object:
            return SimpleNamespace(
                source_count=4,
                projected_count=4,
                source_checksum='source-checksum',
                projected_checksum='projected-checksum',
                replayed=False,
                ready=ready,
            )

    monkeypatch.setattr(key_admin, 'AutoReviewKeyAdminService', _ServiceProbe)


def _install_cli_settings(monkeypatch: pytest.MonkeyPatch, key_admin: object) -> None:
    settings = SimpleNamespace(
        resolved_database_url=lambda: 'sqlite:///:memory:',
    )
    monkeypatch.setattr(key_admin, 'Settings', lambda: settings)


def _assert_in_process_cli_error(
    captured: pytest.CaptureResult[str],
    *,
    result: int,
    code: str,
    exit_code: int,
    forbidden: tuple[str, ...] = (),
) -> None:
    expected = json.dumps({'ok': False, 'code': code}, separators=(',', ':'))
    assert result == exit_code
    assert captured.out == f'{expected}\n'
    assert captured.err == ''
    assert captured.out.count('\n') == 1
    assert set(json.loads(captured.out)) == {'ok', 'code'}
    combined_output = captured.out + captured.err
    for value in (
        'Traceback',
        'usage:',
        str(Path(__file__).resolve().parents[2]),
        *forbidden,
    ):
        assert value not in combined_output


@pytest.mark.parametrize(
    ('command', 'argv'),
    (
        ('status', ['status']),
        ('bootstrap', ['bootstrap']),
        ('rebuild', ['rebuild']),
        (
            'rotate',
            [
                'rotate',
                '--expected-version',
                'v1',
                '--next-version',
                'v2',
                '--reason',
                'runtime-ownership',
            ],
        ),
    ),
)
def test_key_admin_runtime_owned_and_disposed_once_for_each_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    argv: list[str],
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    runtime = _OwnedRuntimeProbe(sessionmaker())
    initialize_calls: list[str] = []
    command_calls: list[str] = []
    _install_cli_settings(monkeypatch, key_admin)
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    _install_cli_service_probe(
        monkeypatch,
        key_admin,
        command_calls=command_calls,
    )
    monkeypatch.setattr(
        key_admin._EnvironmentKeyRingSource,
        'load',
        lambda _self: _valid_cli_key_ring(),
    )

    result = key_admin.main(argv)
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ''
    assert captured.out.count('\n') == 1
    assert initialize_calls == ['sqlite:///:memory:']
    assert command_calls == [command]
    assert runtime.dispose_calls == 1


def test_key_admin_runtime_disposes_before_readiness_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    runtime = _OwnedRuntimeProbe(sessionmaker())
    initialize_calls: list[str] = []
    emissions: list[str] = []
    _install_cli_settings(monkeypatch, key_admin)
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    _install_cli_service_probe(monkeypatch, key_admin, ready=False)
    real_print = print

    def print_after_dispose(*values: object, **kwargs: object) -> None:
        assert runtime.dispose_calls == 1
        emissions.append(str(values[0]))
        real_print(*values, **kwargs)

    monkeypatch.setattr('builtins.print', print_after_dispose)

    result = key_admin.main(['status'])
    captured = capsys.readouterr()

    expected_payload = {
        'runtime_generation': 7,
        'runtime_version': 'v1',
        'runtime_ready': False,
        'projection_ready': False,
        'projection_rebuild_required': True,
        'nonterminal_extraction_count': 2,
        'nonterminal_validation_count': 3,
    }
    assert result == 3
    assert captured.out == (
        json.dumps(expected_payload, separators=(',', ':'), sort_keys=True) + '\n'
    )
    assert captured.err == ''
    assert emissions == [captured.out.rstrip('\n')]
    assert initialize_calls == ['sqlite:///:memory:']
    assert runtime.dispose_calls == 1


@pytest.mark.parametrize(
    ('failure_origin', 'failure', 'expected_code', 'expected_exit_code'),
    (
        (
            'command',
            AutoReviewKeyAdminError('projection_state_missing', 'sensitive-admin'),
            'projection_state_missing',
            2,
        ),
        ('construction', ValueError('sensitive-construction'), 'operation_failed', 3),
        ('command', TypeError('sensitive-dispatch'), 'operation_failed', 3),
    ),
)
def test_key_admin_runtime_disposes_on_admin_service_and_command_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_origin: str,
    failure: Exception,
    expected_code: str,
    expected_exit_code: int,
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    runtime = _OwnedRuntimeProbe(sessionmaker())
    initialize_calls: list[str] = []
    _install_cli_settings(monkeypatch, key_admin)
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    _install_cli_service_probe(
        monkeypatch,
        key_admin,
        construction_failure=failure if failure_origin == 'construction' else None,
        command_failure=failure if failure_origin == 'command' else None,
    )

    result = key_admin.main(['status'])
    captured = capsys.readouterr()

    _assert_in_process_cli_error(
        captured,
        result=result,
        code=expected_code,
        exit_code=expected_exit_code,
        forbidden=(str(failure),),
    )
    assert initialize_calls == ['sqlite:///:memory:']
    assert runtime.dispose_calls == 1


@pytest.mark.parametrize('command_failure', (None, ValueError('prior-command-error')))
def test_key_admin_cleanup_availability_overrides_command_outcome(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command_failure: Exception | None,
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    runtime = _OwnedRuntimeProbe(
        sessionmaker(),
        dispose_failure=DatabaseInitializationError(),
    )
    initialize_calls: list[str] = []
    _install_cli_settings(monkeypatch, key_admin)
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    _install_cli_service_probe(
        monkeypatch,
        key_admin,
        command_failure=command_failure,
    )

    result = key_admin.main(['status'])
    captured = capsys.readouterr()

    _assert_in_process_cli_error(
        captured,
        result=result,
        code='storage_unavailable',
        exit_code=3,
        forbidden=('prior-command-error',),
    )
    assert runtime.dispose_calls == 1


@pytest.mark.parametrize('command_failure', (None, ValueError('prior-command-error')))
def test_key_admin_cleanup_programmer_failure_overrides_command_outcome(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command_failure: Exception | None,
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    cleanup_failure = TypeError('sensitive-cleanup-programmer-failure')
    runtime = _OwnedRuntimeProbe(sessionmaker(), dispose_failure=cleanup_failure)
    initialize_calls: list[str] = []
    _install_cli_settings(monkeypatch, key_admin)
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    _install_cli_service_probe(
        monkeypatch,
        key_admin,
        command_failure=command_failure,
    )

    result = key_admin.main(['status'])
    captured = capsys.readouterr()

    _assert_in_process_cli_error(
        captured,
        result=result,
        code='operation_failed',
        exit_code=3,
        forbidden=(str(cleanup_failure), 'prior-command-error'),
    )
    assert runtime.dispose_calls == 1


@pytest.mark.parametrize(
    'resolver_failure',
    (
        ValueError('resolver-sensitive-value'),
        DatabaseInitializationError(),
    ),
)
def test_key_admin_resolved_url_failure_keeps_configuration_origin(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resolver_failure: Exception,
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    calls: list[str] = []

    def resolved_database_url() -> str:
        raise resolver_failure

    monkeypatch.setattr(
        key_admin,
        'Settings',
        lambda: SimpleNamespace(resolved_database_url=resolved_database_url),
    )
    _install_database_boundary_counter(monkeypatch, key_admin, calls)

    result = key_admin.main(['status'])
    captured = capsys.readouterr()

    _assert_in_process_cli_error(
        captured,
        result=result,
        code='configuration_refused',
        exit_code=2,
        forbidden=('resolver-sensitive-value',),
    )
    assert calls == []


@pytest.mark.parametrize(
    ('case', 'expected_code'),
    (
        ('missing', 'key_ring_unavailable'),
        ('blank-version', 'configuration_refused'),
        ('short-secret', 'configuration_refused'),
        ('equal-secret', 'configuration_refused'),
    ),
)
def test_key_admin_rotate_configuration_precedes_storage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected_code: str,
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    names = (
        'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_VERSION',
        'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_SECRET',
        'PARAWORKS_AUTO_REVIEW_NEXT_KEY_VERSION',
        'PARAWORKS_AUTO_REVIEW_NEXT_KEY_SECRET',
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    values: dict[str, str] = {}
    if case != 'missing':
        values = {
            names[0]: '' if case == 'blank-version' else 'v1',
            names[1]: 'short-sensitive' if case == 'short-secret' else 'c' * 48,
            names[2]: 'v2',
            names[3]: 'c' * 48 if case == 'equal-secret' else 'n' * 48,
        }
        for name, value in values.items():
            monkeypatch.setenv(name, value)
    calls: list[str] = []
    _install_cli_settings(monkeypatch, key_admin)
    _install_database_boundary_counter(monkeypatch, key_admin, calls)
    reason = 'test-key-validation'
    argv = [
        'rotate',
        '--expected-version',
        'v1',
        '--next-version',
        'v2',
        '--reason',
        reason,
    ]

    result = key_admin.main(argv)
    captured = capsys.readouterr()

    _assert_in_process_cli_error(
        captured,
        result=result,
        code=expected_code,
        exit_code=2,
        forbidden=(reason, *argv, *(value for value in values.values() if value)),
    )
    assert calls == []


@pytest.mark.parametrize('command', ('status', 'bootstrap', 'rebuild'))
def test_key_admin_nonrotate_never_loads_environment_key_ring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    runtime = _OwnedRuntimeProbe(sessionmaker())
    initialize_calls: list[str] = []
    _install_cli_settings(monkeypatch, key_admin)
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    _install_cli_service_probe(monkeypatch, key_admin)

    def unexpected_load(_self: object) -> FingerprintKeyRing:
        raise AssertionError('nonrotate-key-ring-sensitive-sentinel')

    monkeypatch.setattr(key_admin._EnvironmentKeyRingSource, 'load', unexpected_load)

    result = key_admin.main([command])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ''
    assert captured.out.count('\n') == 1
    assert 'nonrotate-key-ring-sensitive-sentinel' not in captured.out


def _key_admin_command_failure(failure_kind: str) -> Exception:
    dbapi_types: dict[str, type[DBAPIError]] = {
        'dbapi': DBAPIError,
        'integrity': IntegrityError,
        'programming': ProgrammingError,
        'data': DataError,
        'operational': OperationalError,
        'interface': InterfaceError,
    }
    for suffix, connection_invalidated in (('-true', True), ('-false', False)):
        if failure_kind.endswith(suffix):
            error_type = dbapi_types[failure_kind.removesuffix(suffix)]
            return error_type(
                'SELECT sensitive-command-statement',
                {'secret': 'sensitive-command-parameter'},
                RuntimeError('sensitive-command-dbapi-original'),
                connection_invalidated=connection_invalidated,
            )
    if failure_kind == 'timeout':
        return SQLAlchemyTimeoutError('sensitive-command-timeout')
    if failure_kind == 'disconnection':
        return DisconnectionError('sensitive-command-disconnection')
    if failure_kind == 'invalid-request':
        return InvalidRequestError('sensitive-command-invalid-request')
    if failure_kind == 'statement':
        return StatementError(
            'sensitive-command-statement-error',
            'SELECT sensitive',
            {'secret': 'sensitive'},
            RuntimeError('sensitive-command-statement-original'),
        )
    if failure_kind == 'sqlalchemy':
        return SQLAlchemyError('sensitive-command-sqlalchemy')
    if failure_kind == 'value':
        return ValueError('sensitive-command-value')
    if failure_kind == 'type':
        return TypeError('sensitive-command-type')
    raise AssertionError(f'unknown failure kind: {failure_kind}')


@pytest.mark.parametrize('failure_origin', ('construction', 'dispatch'))
@pytest.mark.parametrize(
    ('failure_kind', 'expected_code'),
    (
        ('dbapi-true', 'storage_unavailable'),
        ('integrity-true', 'storage_unavailable'),
        ('programming-true', 'storage_unavailable'),
        ('data-true', 'storage_unavailable'),
        ('operational-true', 'storage_unavailable'),
        ('operational-false', 'storage_unavailable'),
        ('interface-true', 'storage_unavailable'),
        ('interface-false', 'storage_unavailable'),
        ('timeout', 'storage_unavailable'),
        ('disconnection', 'storage_unavailable'),
        ('dbapi-false', 'operation_failed'),
        ('integrity-false', 'operation_failed'),
        ('programming-false', 'operation_failed'),
        ('data-false', 'operation_failed'),
        ('invalid-request', 'operation_failed'),
        ('statement', 'operation_failed'),
        ('sqlalchemy', 'operation_failed'),
        ('value', 'operation_failed'),
        ('type', 'operation_failed'),
    ),
)
def test_key_admin_command_failure_first_match_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_origin: str,
    failure_kind: str,
    expected_code: str,
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    runtime = _OwnedRuntimeProbe(sessionmaker())
    initialize_calls: list[str] = []
    failure = _key_admin_command_failure(failure_kind)
    _install_cli_settings(monkeypatch, key_admin)
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    _install_cli_service_probe(
        monkeypatch,
        key_admin,
        construction_failure=failure if failure_origin == 'construction' else None,
        command_failure=failure if failure_origin == 'dispatch' else None,
    )

    result = key_admin.main(['status'])
    captured = capsys.readouterr()

    _assert_in_process_cli_error(
        captured,
        result=result,
        code=expected_code,
        exit_code=3,
        forbidden=('sensitive-command',),
    )
    assert initialize_calls == ['sqlite:///:memory:']
    assert runtime.dispose_calls == 1


def _run_key_admin_module_cli(
    *args: str,
    env_updates: dict[str, str] | None = None,
    env_removals: tuple[str, ...] = (),
    pythonpath_entries: Sequence[Path] = (),
) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[2]
    isolated_cwd = Path(__file__).resolve().parent
    assert not (isolated_cwd / '.env').exists()
    allowed_parent_names = (
        'COMSPEC',
        'PATH',
        'PATHEXT',
        'SYSTEMROOT',
        'TEMP',
        'TMP',
        'WINDIR',
    )
    env = {
        name: os.environ[name]
        for name in allowed_parent_names
        if name in os.environ
    }
    env.update(
        {
            'AUTO_REVIEW_MODE': 'disabled',
            'AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS': '60',
            'AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS': '5',
            'AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS': '120',
            'AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS': '30',
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': 'sqlite:///:memory:',
            'DATABASE_URL': 'sqlite:///:memory:',
        }
    )
    env.update(env_updates or {})
    for name in env_removals:
        env.pop(name, None)
    pythonpath = [*(str(path) for path in pythonpath_entries), str(repository_root)]
    env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, '-m', 'backend.app.admin.auto_review_keys', *args],
        cwd=isolated_cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _assert_bounded_cli_error(
    completed: subprocess.CompletedProcess[str],
    *,
    code: str,
    exit_code: int,
    forbidden: tuple[str, ...] = (),
) -> None:
    expected = json.dumps(
        {'ok': False, 'code': code},
        separators=(',', ':'),
    )
    assert completed.returncode == exit_code
    assert completed.stdout == f'{expected}\n'
    assert completed.stderr == ''
    assert json.loads(completed.stdout) == {'ok': False, 'code': code}
    assert set(json.loads(completed.stdout)) == {'ok', 'code'}
    assert len(completed.stdout) < 120
    combined_output = completed.stdout + completed.stderr
    for value in (
        'Traceback',
        'usage:',
        'input_value',
        'ValidationError',
        'AutoReviewKeyAdminError',
        'trusted_fingerprint_projection',
        str(Path(__file__).resolve().parents[2]),
        *forbidden,
    ):
        assert value not in combined_output


def _write_database_cli_probe(tmp_path: Path) -> Path:
    (tmp_path / 'sitecustomize.py').write_text(
        """
import os
import sys

from sqlalchemy.dialects import registry

registry.register(
    'paraworks_probe',
    'paraworks_probe_dialect',
    'ProbeDialect',
)

if os.getenv('PARAWORKS_TEST_BLOCK_DB_INITIALIZER') == '1':
    class _InitializerBlocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'backend.app.db.initialization':
                raise ModuleNotFoundError('non-storage-import-sensitive-sentinel')
            return None

    sys.meta_path.insert(0, _InitializerBlocker())
""".strip()
        + '\n',
        encoding='utf-8',
    )
    (tmp_path / 'paraworks_probe_dialect.py').write_text(
        """
import os
from pathlib import Path

from sqlalchemy.engine.default import DefaultDialect


class ProbeDialect(DefaultDialect):
    name = 'paraworks_probe'
    driver = 'probe'

    @classmethod
    def import_dbapi(cls):
        marker = os.getenv('PARAWORKS_TEST_DBAPI_MARKER')
        if marker:
            Path(marker).write_text('called', encoding='utf-8')
        failure = os.environ['PARAWORKS_TEST_DBAPI_FAILURE']
        if failure == 'module':
            raise ModuleNotFoundError('dbapi-module-sensitive-sentinel')
        if failure == 'import':
            raise ImportError('dbapi-native-sensitive-sentinel')
        if failure == 'oserror':
            raise OSError('dbapi-loader-sensitive-sentinel')
        raise RuntimeError('unknown probe failure')
""".strip()
        + '\n',
        encoding='utf-8',
    )
    return tmp_path


@pytest.mark.parametrize('unknown_option', ('--secret', '--definitely-unknown'))
def test_key_admin_module_cli_bounds_forbidden_and_unknown_arguments(
    unknown_option: str,
) -> None:
    supplied_value = 'forbidden-controller-value'
    completed = _run_key_admin_module_cli(
        'status',
        unknown_option,
        supplied_value,
        env_updates={'AUTO_REVIEW_MODE': 'disabled'},
    )

    _assert_bounded_cli_error(
        completed,
        code='configuration_refused',
        exit_code=2,
        forbidden=(unknown_option, supplied_value, 'error:'),
    )


def test_key_admin_module_cli_bounds_malformed_settings_before_initialization() -> None:
    fake_secret = 'm' * 48
    completed = _run_key_admin_module_cli(
        'status',
        env_updates={
            'AUTO_REVIEW_MODE': 'enforce',
            'AGENT_RUNTIME_FINGERPRINT_SECRET': fake_secret,
        },
        env_removals=('AUTO_REVIEW_ENFORCE_PERCENTAGE',),
    )

    _assert_bounded_cli_error(
        completed,
        code='configuration_refused',
        exit_code=2,
        forbidden=(fake_secret, 'enforce mode requires percentage'),
    )


@pytest.mark.parametrize(
    'invalid_database_url',
    (
        'not-a-storage-url-sensitive',
        'paraworks_unknown_dialect://role:password@127.0.0.1/database',
    ),
)
def test_key_admin_module_cli_bounds_storage_initialization_failure(
    invalid_database_url: str,
) -> None:
    completed = _run_key_admin_module_cli(
        'status',
        env_updates={
            'AUTO_REVIEW_MODE': 'disabled',
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': invalid_database_url,
            'DATABASE_URL': invalid_database_url,
        },
        env_removals=('PARAWORKS_DEMO_DATABASE_URL',),
    )

    _assert_bounded_cli_error(
        completed,
        code='configuration_refused',
        exit_code=2,
        forbidden=(
            invalid_database_url,
            'Could not parse SQLAlchemy URL',
            'paraworks_unknown_dialect',
        ),
    )


@pytest.mark.parametrize('failure', ('module', 'import', 'oserror'))
def test_key_admin_module_cli_classifies_owned_dbapi_load_failure_as_storage(
    tmp_path: Path,
    failure: str,
) -> None:
    probe_path = _write_database_cli_probe(tmp_path)
    sensitive_url = (
        'paraworks_probe://probe_role_test:probe_password@'
        '127.0.0.1:55432/probe_database_test'
    )
    completed = _run_key_admin_module_cli(
        'status',
        env_updates={
            'AUTO_REVIEW_MODE': 'disabled',
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': sensitive_url,
            'DATABASE_URL': sensitive_url,
            'PARAWORKS_TEST_DBAPI_FAILURE': failure,
        },
        pythonpath_entries=(probe_path,),
    )

    _assert_bounded_cli_error(
        completed,
        code='storage_unavailable',
        exit_code=3,
        forbidden=(
            sensitive_url,
            'paraworks_probe',
            'probe_password',
            'sensitive-sentinel',
        ),
    )


def test_key_admin_module_cli_classifies_initializer_import_failure_as_configuration(
    tmp_path: Path,
) -> None:
    probe_path = _write_database_cli_probe(tmp_path)
    completed = _run_key_admin_module_cli(
        'status',
        env_updates={
            'PARAWORKS_TEST_BLOCK_DB_INITIALIZER': '1',
        },
        pythonpath_entries=(probe_path,),
    )

    _assert_bounded_cli_error(
        completed,
        code='configuration_refused',
        exit_code=2,
        forbidden=('non-storage-import-sensitive-sentinel',),
    )


def test_key_admin_module_cli_validates_rotate_before_storage(tmp_path: Path) -> None:
    probe_path = _write_database_cli_probe(tmp_path)
    marker = tmp_path / 'dbapi-marker.txt'
    sensitive_url = (
        'paraworks_probe://rotate_role_test:rotate_password@'
        '127.0.0.1:55432/rotate_database_test'
    )
    invalid_secret = 'short-rotate-sensitive'
    reason = 'subprocess-key-validation-sensitive'
    completed = _run_key_admin_module_cli(
        'rotate',
        '--expected-version',
        'v1',
        '--next-version',
        'v2',
        '--reason',
        reason,
        env_updates={
            'PARAWORKS_DATABASE_URL': sensitive_url,
            'DATABASE_URL': sensitive_url,
            'PARAWORKS_TEST_DBAPI_FAILURE': 'module',
            'PARAWORKS_TEST_DBAPI_MARKER': str(marker),
            'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_VERSION': 'v1',
            'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_SECRET': invalid_secret,
            'PARAWORKS_AUTO_REVIEW_NEXT_KEY_VERSION': 'v2',
            'PARAWORKS_AUTO_REVIEW_NEXT_KEY_SECRET': 'n' * 48,
        },
        pythonpath_entries=(probe_path,),
    )

    _assert_bounded_cli_error(
        completed,
        code='configuration_refused',
        exit_code=2,
        forbidden=(sensitive_url, invalid_secret, reason, str(marker)),
    )
    assert not marker.exists()


def test_key_admin_module_cli_bounds_lazy_rebuild_refusal(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = auto_review_postgres
    mismatched_secret = 'w' * 48
    completed = _run_key_admin_module_cli(
        'rebuild',
        env_updates={
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': settings.database_url,
            'DATABASE_URL': settings.database_url,
            'AGENT_RUNTIME_FINGERPRINT_KEY_VERSION': 'v1',
            'AGENT_RUNTIME_FINGERPRINT_SECRET': mismatched_secret,
            'AUTO_REVIEW_MODE': 'disabled',
        },
    )
    try:
        _assert_bounded_cli_error(
            completed,
            code='runtime_key_identity_mismatch',
            exit_code=2,
            forbidden=(mismatched_secret,),
        )
    finally:
        rebuild_trusted_fingerprint_projection(
            session_factory=factory,
            settings=settings,
        )


@pytest.mark.parametrize(
    ('expected_version', 'current_secret_matches', 'expected_code'),
    (
        ('wrong-version', True, 'key_version_mismatch'),
        ('v1', False, 'old_key_identity_mismatch'),
    ),
)
def test_key_admin_module_cli_bounds_projection_admin_errors(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    expected_version: str,
    current_secret_matches: bool,
    expected_code: str,
) -> None:
    _, settings = auto_review_postgres
    current_secret = ('t' if current_secret_matches else 'w') * 48
    next_secret = 'u' * 48
    reason = 'subprocess-negative-safety'
    completed = _run_key_admin_module_cli(
        'rotate',
        '--expected-version',
        expected_version,
        '--next-version',
        'v2',
        '--reason',
        reason,
        env_updates={
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': settings.database_url,
            'AGENT_RUNTIME_FINGERPRINT_KEY_VERSION': 'v1',
            'AGENT_RUNTIME_FINGERPRINT_SECRET': current_secret,
            'AUTO_REVIEW_MODE': 'disabled',
            'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_VERSION': 'v1',
            'PARAWORKS_AUTO_REVIEW_CURRENT_KEY_SECRET': current_secret,
            'PARAWORKS_AUTO_REVIEW_NEXT_KEY_VERSION': 'v2',
            'PARAWORKS_AUTO_REVIEW_NEXT_KEY_SECRET': next_secret,
        },
    )

    _assert_bounded_cli_error(
        completed,
        code=expected_code,
        exit_code=2,
        forbidden=(current_secret, next_secret, expected_version, reason),
    )


@pytest.mark.parametrize(
    'module_name',
    (
        'backend.app.knowledge.claim_fingerprints',
        'backend.app.knowledge.trusted_fingerprint_projection',
        'backend.app.knowledge.trusted_provenance',
        'backend.app.review.auto_review_resolution',
    ),
)
@pytest.mark.parametrize('application_first', (False, True))
def test_task5_modules_import_in_isolated_direct_and_application_orders(
    module_name: str,
    application_first: bool,
) -> None:
    statements = ['import importlib']
    if application_first:
        statements.append("importlib.import_module('backend.app.main')")
    statements.append(f'importlib.import_module({module_name!r})')
    completed = subprocess.run(
        [sys.executable, '-c', ';'.join(statements)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0
    assert completed.stdout == ''
    assert completed.stderr == ''


def test_key_admin_status_exit_code_tracks_fresh_readiness_without_key_output(
    auto_review_db_session: tuple[Session, Settings],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from backend.app.admin import auto_review_keys as key_admin

    db, settings = auto_review_db_session
    factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    runtime = _OwnedRuntimeProbe(factory)
    initialize_calls: list[str] = []
    _install_owned_runtime(monkeypatch, key_admin, runtime, initialize_calls)
    monkeypatch.setattr(key_admin, 'Settings', lambda: settings)
    assert key_admin.main(['status']) == 0
    healthy_output = capsys.readouterr().out
    assert '"projection_ready":true' in healthy_output
    assert settings.agent_runtime_fingerprint_secret not in healthy_output
    assert fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    ) not in healthy_output

    row = db.scalars(
        select(TrustedKnowledgeFingerprint).order_by(
            TrustedKnowledgeFingerprint.knowledge_type,
            TrustedKnowledgeFingerprint.knowledge_id,
        )
    ).first()
    assert row is not None
    db.delete(row)
    db.commit()
    assert key_admin.main(['status']) == 3
    unhealthy_output = capsys.readouterr().out
    assert '"projection_ready":false' in unhealthy_output
    assert '"projection_rebuild_required":true' in unhealthy_output
    assert len(unhealthy_output) < 500
    assert runtime.dispose_calls == 2


def test_projection_rebuild_cli_reports_exact_summary_and_replay(
    db_session: Session,
) -> None:
    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
    from backend.app.knowledge.trusted_fingerprint_projection import (
        rebuild_trusted_fingerprint_projection,
    )

    settings = _settings()
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add_all(
        [
            AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier=verifier,
                generation=1,
                ready=False,
            ),
            TrustedKnowledgeFingerprintProjectionState(
                component='trusted_knowledge_fingerprints',
                projection_schema_version='trusted-fingerprint-projection:v1',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier=verifier,
                generation=1,
                ready=False,
                rebuild_required=True,
            ),
            HistoryEvent(
                title='legacy exact collision title',
                reason='legacy scope cannot be proven',
                project_key=None,
                source_links=['https://example.test/legacy'],
                source_snippets=['legacy evidence'],
                confidence_score=1.0,
                permission_level='internal',
                review_status='approved',
                source_review_item_id=None,
            ),
        ]
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.get_bind(), expire_on_commit=False)
    first = rebuild_trusted_fingerprint_projection(
        session_factory=factory,
        settings=settings,
    )
    result = rebuild_trusted_fingerprint_projection(
        session_factory=factory,
        settings=settings,
    )
    assert first.replayed is False
    assert result.source_count == result.projected_count == 1
    assert result.source_checksum == result.projected_checksum
    assert result.replayed is True
    assert result.ready is False


def test_disabled_only_rotation_waits_for_old_generation_and_rebuilds_new_projection(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, v1_settings = auto_review_postgres
    rebuild_trusted_fingerprint_projection(
        session_factory=factory,
        settings=v1_settings,
    )
    with factory() as db:
        canonical = _seed_c5_item(db)
        ReviewTransitionService(settings=v1_settings).transition(
            db=db,
            item_id=canonical.id,
            action='approve',
            actor=human_review_actor(USERS['admin']),
        )
        db.commit()
        old_links = tuple(
            db.scalars(
                select(TrustedKnowledgeApprovalLink)
                .where(TrustedKnowledgeApprovalLink.review_item_id == canonical.id)
                .order_by(TrustedKnowledgeApprovalLink.id)
            ).all()
        )
        old_snapshots = {
            link.id: (
                link.claim_fingerprint,
                link.fingerprint_key_version,
                link.fingerprint_key_material_verifier,
            )
            for link in old_links
        }
        canonical_id = canonical.id
        canonical_payload = dict(canonical.payload)

    rotated = False
    v2_secret = 'u' * 48
    try:
        result = _rotation_service(
            factory,
            settings=v1_settings,
            current_version='v1',
            current_secret='t' * 48,
            next_version='v2',
            next_secret=v2_secret,
        ).rotate_fingerprint_key(
            expected_version='v1',
            next_version='v2',
            reason='disposable historical reaffirmation test',
            principal='system:local-auto-review-key-admin',
        )
        rotated = True
        assert result.ready is True
        v2_settings = v1_settings.model_copy(
            update={
                'agent_runtime_fingerprint_key_version': 'v2',
                'agent_runtime_fingerprint_secret': v2_secret,
            }
        )
        with factory() as db:
            historical = tuple(
                db.scalars(
                    select(TrustedKnowledgeApprovalLink)
                    .where(TrustedKnowledgeApprovalLink.review_item_id == canonical_id)
                    .order_by(TrustedKnowledgeApprovalLink.id)
                ).all()
            )
            assert {
                link.id: (
                    link.claim_fingerprint,
                    link.fingerprint_key_version,
                    link.fingerprint_key_material_verifier,
                )
                for link in historical
            } == old_snapshots
            fingerprints = {
                (row.knowledge_type, row.knowledge_id): row.normalized_claim_fingerprint
                for row in db.scalars(select(TrustedKnowledgeFingerprint)).all()
            }
            primary = next(link for link in historical if link.promotion_effect_kind == 'primary')
            companion = next(
                link for link in historical if link.promotion_effect_kind == 'companion'
            )
            candidate = _seed_c5_item(
                db,
                graph_version='company-memory-review-v2.1-auto-review',
                payload=canonical_payload,
                settings=v2_settings,
            )
            _seed_completed_validation(db, candidate, settings=v2_settings)
            reaffirmed = _auto_resolution_service(v2_settings).resolve(
                db=db,
                item_id=candidate.id,
                directive=ReuseExistingPromotion(
                    expected_type='history_event',
                    expected_id=primary.knowledge_id,
                    expected_claim_fingerprint=fingerprints[
                        ('history_event', primary.knowledge_id)
                    ],
                    expected_companion_id=companion.knowledge_id,
                    expected_companion_claim_fingerprint=fingerprints[
                        ('timeline_event', companion.knowledge_id)
                    ],
                ),
            )
            assert reaffirmed.promotion is not None
            assert reaffirmed.promotion.effect == 'reaffirmed'
            new_links = tuple(
                db.scalars(
                    select(TrustedKnowledgeApprovalLink).where(
                        TrustedKnowledgeApprovalLink.review_item_id == candidate.id
                    )
                ).all()
            )
            assert len(new_links) == 2
            assert {link.fingerprint_key_version for link in new_links} == {'v2'}
            db.commit()
    finally:
        if rotated:
            restored = _rotation_service(
                factory,
                settings=v1_settings,
                current_version='v2',
                current_secret=v2_secret,
                next_version='v1',
                next_secret='t' * 48,
            ).rotate_fingerprint_key(
                expected_version='v2',
                next_version='v1',
                reason='restore disposable test key generation',
                principal='system:local-auto-review-key-admin',
            )
            assert restored.ready is True


@pytest.mark.parametrize('provider_attempt_count', [0, 1])
def test_rotation_refuses_attempt_zero_or_inflight_provider_call_until_ledger_recovery(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
    provider_attempt_count: int,
) -> None:
    factory, settings = auto_review_postgres
    with factory() as db:
        call, validation = _seed_claimed_validation_call(
            db,
            provider_attempt_count=provider_attempt_count,
            settings=settings,
        )
        call_id = call.id
        validation_id = validation.id
    service = _rotation_service(
        factory,
        settings=settings,
        current_version='v1',
        current_secret='t' * 48,
        next_version='v2',
        next_secret='u' * 48,
    )
    with pytest.raises(AutoReviewKeyAdminError) as exc_info:
        service.rotate_fingerprint_key(
            expected_version='v1',
            next_version='v2',
            reason='must refuse nonterminal provider ledger',
            principal='system:local-auto-review-key-admin',
        )
    assert exc_info.value.code == 'nonterminal_provider_calls'
    with factory() as db:
        retained = db.get(AutoReviewValidationCall, call_id)
        retained_child = db.get(AutoReviewValidation, validation_id)
        assert retained is not None and retained.status == 'claimed'
        assert retained_child is not None and retained_child.status == 'claimed'
        _mark_claimed_validation_failed(db, retained, retained_child)


def test_rotation_succeeds_only_after_marked_attempt_is_conservatively_charged_terminal(
    auto_review_postgres: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = auto_review_postgres
    with factory() as db:
        call, validation = _seed_claimed_validation_call(
            db,
            provider_attempt_count=1,
            settings=settings,
        )
        _mark_claimed_validation_failed(db, call, validation)
        assert call.charged_input_tokens == call.reserved_input_tokens
        assert call.charged_output_tokens == call.reserved_output_tokens
        assert call.charged_cost_usd == call.reserved_cost_usd
    v2_secret = 'u' * 48
    rotated = False
    try:
        result = _rotation_service(
            factory,
            settings=settings,
            current_version='v1',
            current_secret='t' * 48,
            next_version='v2',
            next_secret=v2_secret,
        ).rotate_fingerprint_key(
            expected_version='v1',
            next_version='v2',
            reason='terminal provider ledger permits disposable rotation',
            principal='system:local-auto-review-key-admin',
        )
        rotated = True
        assert result.ready is True
        with factory() as db:
            runtime = db.scalars(select(AutoReviewRuntimeKeyState)).one()
            assert runtime.fingerprint_key_version == 'v2'
    finally:
        if rotated:
            restored = _rotation_service(
                factory,
                settings=settings,
                current_version='v2',
                current_secret=v2_secret,
                next_version='v1',
                next_secret='t' * 48,
            ).rotate_fingerprint_key(
                expected_version='v2',
                next_version='v1',
                reason='restore disposable test key generation',
                principal='system:local-auto-review-key-admin',
            )
            assert restored.ready is True
