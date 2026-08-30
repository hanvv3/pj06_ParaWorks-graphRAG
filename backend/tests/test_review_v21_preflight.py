from dataclasses import replace
from decimal import Decimal

import pytest
from sqlalchemy import select

from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_preflight import (
    PreparedReviewRequestV21,
    V21PreparedReviewConfig,
    build_prepared_review_identity_v21,
    create_or_reuse_review_thread,
    prepare_review_request,
    review_thread_matches_prepared,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models.agent_workflows import AgentWorkflowRequest
from backend.app.models.source import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
from backend.app.schemas.review_workflow import ReviewWorkflowRunRequest


def _plan_identity() -> dict[str, object]:
    return {
        'agent_name': 'mail_document_agent',
        'provider': 'openai',
        'model': 'gpt-5.4-mini-2026-03-17',
        'reasoning_effort': 'none',
        'route_version': 'auto-review-extraction-route:v1',
        'prompt_version': 'mail-document-extraction:v1',
        'output_contract_version': 'mail-document-extraction:v1',
        'extraction_registry_version': 'auto-review-extraction-registry:v1',
        'cost_policy_version': 'auto-review-extraction-cost:v1',
        'token_estimator_version': 'openai-o200k-extraction:v1',
        'tokenizer_encoding': 'o200k_base',
        'reply_priming_tokens': 16,
        'framing_safety_tokens': 512,
        'max_input_chars': 24000,
        'max_input_tokens': 10000,
        'max_output_tokens': 2048,
        'max_candidates': 1,
        'max_provider_attempts': 1,
        'input_usd_per_1m': '0.750000',
        'output_usd_per_1m': '4.500000',
        'provider_safety_state_version': 5,
        'timing': [60, 5, 120, 30],
        'prepared_content_hmac': 'e' * 64,
        'prepared_character_count': 800,
        'framed_input_tokens': 900,
        'reserved_cost_usd': '0.016716',
    }


def _config() -> V21PreparedReviewConfig:
    return V21PreparedReviewConfig(
        configured_auto_review_mode='shadow',
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        validator_reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v2',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version='auto-review-policy:v1',
        cost_policy_version='auto-review-cost:v1',
        fingerprint_key_version='test-v1',
        fingerprint_key_material_verifier='a' * 64,
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
        enforce_percentage=10,
        authorized_percentage_at_launch=10,
        rollout_authorization_generation=3,
        validation_provider_safety_state_version=7,
        rollout_control_epoch=9,
        extraction_plan_set_hmac='b' * 64,
        extraction_provider_safety_snapshot_set_hmac='c' * 64,
        confirmed_extraction_cost_ceiling_usd=Decimal('0.016716'),
        confirmed_validation_cost_ceiling_usd=Decimal('0.048864'),
        confirmed_total_cost_ceiling_usd=Decimal('0.065580'),
        total_budget_limit_usd=Decimal('0.200000'),
        extraction_plan_identities=(_plan_identity(),),
    )


def _prepared(
    config: V21PreparedReviewConfig | None = None,
) -> PreparedReviewRequestV21:
    config = config or _config()
    identity = {
        **config.extraction_plan_identities[0],
        'max_provider_attempts': config.max_provider_attempts,
        'timing': [
            config.provider_timeout_seconds,
            config.provider_send_start_window_seconds,
            config.provider_attempt_lease_seconds,
            config.provider_commit_grace_seconds,
        ],
    }
    config = replace(config, extraction_plan_identities=(identity,))
    return build_prepared_review_identity_v21(
        source_refs=(),
        agent_names=('mail_document_agent',),
        evidence_version_hash='d' * 64,
        security_scope_id='scope-1',
        config=config,
        fingerprint_secret=b'test-secret-at-least-thirty-two-bytes',
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('configured_auto_review_mode', 'enforce'),
        ('policy_version', 'auto-review-policy:v2'),
        ('validator_model', 'gpt-5.6-sol'),
        ('validator_output_contract_version', 'candidate-validation-batch:v2'),
        ('enforce_percentage', 100),
        ('confirmed_total_cost_ceiling_usd', Decimal('0.065581')),
    ),
)
def test_v21_identity_changes_with_mode_policy_model_output_contract_percentage_or_ceiling(
    field, value
):
    baseline = _prepared()
    changed = _prepared(replace(_config(), **{field: value}))
    assert changed.input_hash != baseline.input_hash


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('fingerprint_key_material_verifier', 'e' * 64),
        ('cost_policy_version', 'auto-review-cost:v2'),
        ('max_provider_attempts', 2),
    ),
)
def test_v21_identity_changes_with_key_material_cost_policy_or_attempt_cap(
    field, value
):
    assert (
        _prepared(replace(_config(), **{field: value})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    'field',
    (
        'provider_timeout_seconds',
        'provider_send_start_window_seconds',
        'provider_attempt_lease_seconds',
        'provider_commit_grace_seconds',
    ),
)
def test_v21_identity_changes_with_any_shared_provider_timing_value(field):
    config = _config()
    assert (
        _prepared(replace(config, **{field: getattr(config, field) + 1})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    'field', ('validator_input_usd_per_1m', 'validator_output_usd_per_1m')
)
def test_v21_identity_changes_with_either_validator_price_snapshot(field):
    assert (
        _prepared(replace(_config(), **{field: Decimal('99.000000')})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    (('authorized_percentage_at_launch', 100), ('rollout_authorization_generation', 4)),
)
def test_v21_identity_changes_with_authorized_percentage_or_rollout_generation(
    field, value
):
    assert (
        _prepared(replace(_config(), **{field: value})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    (('validation_provider_safety_state_version', 8), ('rollout_control_epoch', 10)),
)
def test_v21_identity_changes_with_provider_safety_state_or_rollout_control_epoch(
    field, value
):
    assert (
        _prepared(replace(_config(), **{field: value})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize('field', ('reply_priming_tokens', 'framing_safety_tokens'))
def test_v21_identity_changes_with_reply_priming_or_framing_safety_constant(field):
    config = _config()
    assert (
        _prepared(replace(config, **{field: getattr(config, field) + 1})).input_hash
        != _prepared().input_hash
    )


def test_v21_thread_replay_requires_every_stored_auto_review_field_to_match():
    prepared = _prepared()
    assert prepared.graph_version == COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
    assert prepared.matches_stored_snapshot(prepared.stored_snapshot())
    for field, value in prepared.stored_snapshot().items():
        replacement = 'changed' if isinstance(value, str) else (value + 1)
        assert not prepared.matches_stored_snapshot(
            {**prepared.stored_snapshot(), field: replacement}
        )


def test_v21_create_persists_v21_graph_checkpoint_and_every_request_snapshot(
    db_session,
) -> None:
    settings = Settings(
        agent_runtime_security_scope_id='scope-1',
        agent_runtime_fingerprint_secret='test-secret-at-least-thirty-two-bytes',
        agent_runtime_fingerprint_key_version='test-v1',
    )
    actor = DemoUser(
        id='owner',
        email='owner@example.test',
        role='employee',
        permission_levels={'internal'},
        name='Owner',
        title='Tester',
        department='Quality',
    )
    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            name='mail_document_agent',
            owner='Developer B',
            input_contract='EvidencePacket',
            output_contract='AgentRunResult',
            prompt_versions=('mail-document-extraction:v1',),
            supported_permissions=('internal',),
            capabilities=('review_draft',),
        )
    )
    server_signature = '1' * 64
    source = Source(
        source_type='gmail',
        source_id='gmail:v21-storage',
        source_url='https://private.example/v21-storage',
        title='private title',
        permission_level='internal',
        raw_metadata={
            'content_signature': 'v21-signature',
            'review_batch_mode': 'v2_explicit',
            'review_batch_signature': server_signature,
        },
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=server_signature,
    )
    db_session.add(source)
    db_session.flush()
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='v1',
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version='v1',
        body='private canonical body',
    )
    db_session.add(version)
    db_session.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_gmail_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='message/rfc822',
        document_version_label='v1',
        revision_id='revision-1',
        content_signature=server_signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=server_signature,
        parser_policy_version='server-source-parser-policy:v1',
        parser_version='source-event-paragraph-parser:v1',
        chunk_policy_version='paragraph-chunks:1200:v1',
        chunk_count=1,
    )
    db_session.add(parser_run)
    db_session.flush()
    db_session.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=parser_run.id,
            chunk_index=0,
            text='private canonical body',
            source_snippet='private canonical body',
            permission_level='internal',
            metadata_={},
        )
    )
    document.current_document_version_id = version.id
    db_session.commit()
    request = ReviewWorkflowRunRequest(
        source_refs=[
            {
                'source_type': 'gmail',
                'source_id': source.source_id,
                'version_or_signature': server_signature,
            }
        ],
        agent_names=['mail_document_agent'],
        client_request_id='v21-storage-key',
    )
    v20 = prepare_review_request(
        db_session,
        request=request,
        actor=actor,
        registry=registry,
        settings=settings,
    )
    prepared = build_prepared_review_identity_v21(
        source_refs=v20.source_refs,
        agent_names=v20.agent_names,
        security_scope_id='scope-1',
        config=_config(),
        fingerprint_secret=b'test-secret-at-least-thirty-two-bytes',
    )

    created = create_or_reuse_review_thread(
        db_session,
        prepared=prepared,
        request=request,
        actor=actor,
        settings=settings,
    )

    stored = db_session.scalar(select(AgentWorkflowRequest))
    assert created.thread.graph_version == COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
    assert created.thread.checkpoint_thread_id.startswith('review-v21:')
    assert stored is not None
    assert {
        field: getattr(stored, field) for field in prepared.stored_snapshot()
    } == prepared.stored_snapshot()
    assert stored.auto_review_extraction_provider == 'openai'
    assert stored.auto_review_extraction_model == 'gpt-5.4-mini-2026-03-17'
    assert stored.auto_review_extraction_reasoning_effort == 'none'
    assert (
        stored.auto_review_extraction_route_version == 'auto-review-extraction-route:v1'
    )
    assert stored.selected_extraction_agent_count == 1
    assert stored.extraction_max_input_chars_per_agent == 24000
    assert stored.extraction_max_input_tokens_per_agent == 10000
    assert stored.extraction_max_output_tokens_per_agent == 2048
    assert review_thread_matches_prepared(
        db_session,
        thread=created.thread,
        prepared=prepared,
        settings=settings,
    )
