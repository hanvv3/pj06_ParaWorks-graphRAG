from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import backend.app.models as models
from backend.app.db.base import Base

RUNTIME_TABLES = {
    'agent_run_cost_components',
    'rag_provider_safety_authorities',
    'rag_provider_readiness',
    'rag_provider_safety_transitions',
    'rag_advisory_lock_key_registry',
}

AGENT_RUN_V2_COLUMNS = {
    'run_contract_version',
    'run_record_phase',
    'total_charged_cost_usd',
    'projection_owner_fence_hmac',
}

ASSISTANT_INTEGRITY_COLUMNS = {
    'content_write_mode',
    'content_hmac_schema_version',
    'assistant_message_content_hmac',
    'content_hmac_key_version',
    'content_hmac_key_material_verifier',
    'content_origin',
    'content_origin_hmac',
    'rag_result_hmac',
    'linked_agent_run_id',
    'dependency_set_hmac_schema_version',
    'dependency_set_hmac',
    'parent_selected_evidence_projection_hmac',
    'model_influence_set_hmac',
}

DEPENDENCY_V2_COLUMNS = {
    'dependency_serving_scope',
    'dependency_role',
    'dependency_child_hmac',
    'approval_provenance_hmac',
    'evidence_link_set_hmac',
    'legacy_dependency_identity_hmac',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'selected_v1_citation_projection_hmac',
    'serving_identity_hmac',
    'serving_version_fingerprint',
    'support_mode',
}


def _legacy_agent_run() -> models.AgentRun:
    return models.AgentRun(
        agent_name='legacy',
        prompt_version='rag-answer:v1',
        status='complete',
        source_window='legacy',
        cache_key='legacy',
        model_name='deterministic',
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        permission_level='internal',
        completed_at=datetime.now(UTC),
    )


def _v2_parent(**overrides: object) -> models.AgentRun:
    values: dict[str, object] = {
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v2',
        'status': 'running',
        'source_window': 'rag-v2:admission:enforce:ask:keyword',
        'cache_key': 'rag-v2-admission:' + 'a' * 64,
        'model_name': 'rag-v2-admission',
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'estimated_cost_usd': 0.0,
        'permission_level': 'restricted',
        'run_contract_version': 'rag-run:v2',
        'run_record_phase': 'admission',
        'total_charged_cost_usd': Decimal('0.000000'),
        'projection_owner_fence_hmac': None,
        'completed_at': None,
    }
    values.update(overrides)
    return models.AgentRun(**values)


def _component(parent_id: int, component: str, ordinal: int, **overrides: object):
    identities = {
        'query_embedding': (
            'text-embedding-3-small',
            'rag-query-embedding-config:v1',
            'rag-query-embedding-cost:v1',
            'openai-cl100k-text-embedding-3-small:v1',
        ),
        'answer_generation': (
            'gpt-5.4-mini-2026-03-17',
            'rag-answer-model-config:v1',
            'rag-answer-cost:v1',
            'openai-o200k-rag-answer:v1',
        ),
    }
    model, config, cost, estimator = identities[component]
    values: dict[str, object] = {
        'agent_run_id': parent_id,
        'component': component,
        'component_ordinal': ordinal,
        'dispatch_state': 'terminal',
        'attempted': False,
        'dispatch_count': 0,
        'reserved_input_tokens': 0,
        'reserved_output_tokens': 0,
        'actual_input_tokens': None,
        'actual_output_tokens': None,
        'reserved_cost_usd': Decimal('0.000000'),
        'charged_cost_usd': Decimal('0.000000'),
        'charge_basis': 'zero',
        'overrun': False,
        'provider': 'openai',
        'model': model,
        'authorized_model_config_version': config,
        'authorized_model_config_snapshot_hmac': 'b' * 64,
        'authorized_cost_policy_version': cost,
        'authorized_token_estimator_version': estimator,
        'authorized_policy_snapshot_hmac': 'c' * 64,
        'dispatch_fence_hmac': None,
        'process_instance_hmac': None,
        'terminal_outcome': None,
    }
    values.update(overrides)
    return models.AgentRunCostComponent(**values)


def test_runtime_models_export_additive_tables_and_nullable_historical_columns() -> None:
    assert set(Base.metadata.tables) >= RUNTIME_TABLES
    assert set(Base.metadata.tables['agent_runs'].c.keys()) >= AGENT_RUN_V2_COLUMNS
    assert set(
        Base.metadata.tables['assistant_messages'].c.keys()
    ) >= ASSISTANT_INTEGRITY_COLUMNS
    assert set(
        Base.metadata.tables['assistant_message_evidence_dependencies'].c.keys()
    ) >= DEPENDENCY_V2_COLUMNS
    for column in AGENT_RUN_V2_COLUMNS:
        assert Base.metadata.tables['agent_runs'].c[column].nullable
    for column in ASSISTANT_INTEGRITY_COLUMNS:
        assert Base.metadata.tables['assistant_messages'].c[column].nullable
    for column in DEPENDENCY_V2_COLUMNS:
        assert Base.metadata.tables['assistant_message_evidence_dependencies'].c[
            column
        ].nullable


def test_sqlite_preserves_historical_parent_and_enforces_v2_parent_shape() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        legacy = _legacy_agent_run()
        db.add(legacy)
        db.commit()
        assert legacy.run_contract_version is None
        assert legacy.total_charged_cost_usd is None

        valid = _v2_parent()
        db.add(valid)
        db.commit()
        assert valid.total_charged_cost_usd == Decimal('0.000000')

        db.add(_v2_parent(run_record_phase='unknown'))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(
            _v2_parent(
                run_record_phase='cost_finalized_pending_projection',
                projection_owner_fence_hmac=None,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


def test_sqlite_enforces_component_identity_order_and_terminal_zero_shape() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        parent = _v2_parent()
        db.add(parent)
        db.flush()
        db.add_all(
            [
                _component(parent.id, 'query_embedding', 0),
                _component(parent.id, 'answer_generation', 1),
            ]
        )
        db.commit()

        db.add(_component(parent.id, 'query_embedding', 1))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(
            _component(
                parent.id,
                'query_embedding',
                0,
                reserved_input_tokens=1,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


def test_runtime_numeric_and_int4_storage_contracts_are_exact() -> None:
    component = Base.metadata.tables['agent_run_cost_components']
    assert component.c.reserved_input_tokens.type.__class__.__name__ == 'BigInteger'
    assert component.c.actual_output_tokens.type.__class__.__name__ == 'BigInteger'
    assert component.c.reserved_cost_usd.type.precision == 24
    assert component.c.reserved_cost_usd.type.scale == 6
    assert component.c.charged_cost_usd.type.precision == 24
    assert component.c.charged_cost_usd.type.scale == 6

    advisory = Base.metadata.tables['rag_advisory_lock_key_registry']
    assert advisory.c.key1.type.__class__.__name__ == 'Integer'
    assert advisory.c.key2.type.__class__.__name__ == 'Integer'
    assert {column.name for column in advisory.primary_key.columns} == {'key1', 'key2'}


def test_provider_tables_declare_singleton_family_and_append_only_shapes() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    schema = inspect(engine)
    authority_checks = {
        item['name']
        for item in schema.get_check_constraints('rag_provider_safety_authorities')
    }
    readiness_uniques = {
        item['name']
        for item in schema.get_unique_constraints('rag_provider_readiness')
    }
    transition_uniques = {
        item['name']
        for item in schema.get_unique_constraints('rag_provider_safety_transitions')
    }
    assert 'ck_rag_provider_safety_authority_singleton' in authority_checks
    assert 'uq_rag_provider_readiness_family' in readiness_uniques
    assert 'uq_rag_provider_safety_transition_generation' in transition_uniques


def test_v2_parent_rejects_partial_historical_and_invalid_terminal_lifecycle() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_legacy_agent_run())
        db.commit()

        partial = _legacy_agent_run()
        partial.run_contract_version = 'rag-run:v2'
        with pytest.raises(IntegrityError):
            db.add(partial)
            db.commit()
        db.rollback()

        db.add(
            _v2_parent(
                status='complete',
                run_record_phase='final',
                completed_at=None,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


@pytest.mark.parametrize(
    'overrides',
    (
        {
            'dispatch_state': 'dispatching',
            'attempted': True,
            'dispatch_count': 1,
            'reserved_input_tokens': 10,
            'reserved_cost_usd': Decimal('0.001000'),
            'charged_cost_usd': Decimal('0.001000'),
            'charge_basis': 'reserved',
            'dispatch_fence_hmac': 'd' * 64,
            'process_instance_hmac': 'e' * 64,
        },
        {
            'attempted': True,
            'dispatch_count': 1,
            'reserved_input_tokens': 10,
            'actual_input_tokens': 7,
            'actual_output_tokens': 0,
            'reserved_cost_usd': Decimal('0.001000'),
            'charged_cost_usd': Decimal('0.000700'),
            'charge_basis': 'actual',
            'dispatch_fence_hmac': 'd' * 64,
            'process_instance_hmac': 'e' * 64,
            'terminal_outcome': 'component_succeeded',
        },
        {
            'dispatch_state': 'abandoned_unknown',
            'attempted': True,
            'dispatch_count': 1,
            'reserved_input_tokens': 10,
            'reserved_cost_usd': Decimal('0.001000'),
            'charged_cost_usd': Decimal('0.001000'),
            'charge_basis': 'reserved',
            'dispatch_fence_hmac': 'd' * 64,
            'process_instance_hmac': 'e' * 64,
            'terminal_outcome': 'abandoned_unknown',
        },
    ),
)
def test_component_accepts_each_authoritative_runtime_state(
    overrides: dict[str, object],
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        parent = _v2_parent()
        db.add(parent)
        db.flush()
        db.add(_component(parent.id, 'query_embedding', 0, **overrides))
        db.commit()


def test_assistant_historical_null_shape_and_future_integrity_shape() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        conversation = models.AssistantConversation(user_id='owner', title='title')
        db.add(conversation)
        db.flush()
        historical = models.AssistantMessage(
            conversation_id=conversation.id,
            role='assistant',
            content='legacy',
            evidence_contract_version=None,
            serving_dependency_count=None,
        )
        db.add(historical)
        db.commit()
        assert historical.content_write_mode is None
        assert historical.assistant_message_content_hmac is None

        parent = _v2_parent(
            status='complete',
            run_record_phase='final',
            completed_at=datetime.now(UTC),
        )
        db.add(parent)
        db.flush()
        canned = models.AssistantMessage(
            conversation_id=conversation.id,
            role='assistant',
            content='  exact bytes  ',
            evidence_contract_version='none-v1',
            serving_dependency_count=0,
            content_write_mode='rag_v2_exact',
            content_hmac_schema_version='assistant-message-content-hmac:v1',
            assistant_message_content_hmac='a' * 64,
            content_hmac_key_version='runtime-key-v1',
            content_hmac_key_material_verifier='b' * 64,
            content_origin='rag_canned',
            content_origin_hmac='c' * 64,
            rag_result_hmac='d' * 64,
            linked_agent_run_id=parent.id,
        )
        db.add(canned)
        db.commit()
        assert canned.content == '  exact bytes  '

        invalid = models.AssistantMessage(
            conversation_id=conversation.id,
            role='assistant',
            content='partial',
            evidence_contract_version='none-v1',
            serving_dependency_count=0,
            content_write_mode='rag_v2_exact',
        )
        db.add(invalid)
        with pytest.raises(IntegrityError):
            db.commit()


def test_dependency_metadata_declares_future_scope_role_and_hmac_checks() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    checks = {
        item['name']
        for item in inspect(engine).get_check_constraints(
            'assistant_message_evidence_dependencies'
        )
    }
    assert {
        'ck_assistant_message_dependency_v2_scope_role',
        'ck_assistant_message_dependency_v2_hmacs',
        'ck_assistant_message_dependency_v2_support',
    } <= checks


def test_provider_active_family_and_authority_singleton_are_enforced() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    authority = models.RagProviderSafetyAuthority(
        id=1,
        authority_uuid='00000000-0000-0000-0000-000000000001',
        designated_environment_id='test',
        global_safety_generation=0,
        envelope_digest='a' * 64,
        fingerprint_key_version='runtime-key-v1',
        fingerprint_key_material_verifier='b' * 64,
    )
    with Session(engine) as db:
        db.add(authority)
        db.commit()
        db.add(
            models.RagProviderSafetyAuthority(
                id=2,
                authority_uuid='00000000-0000-0000-0000-000000000002',
                designated_environment_id='test',
                global_safety_generation=0,
                envelope_digest='c' * 64,
                fingerprint_key_version='runtime-key-v1',
                fingerprint_key_material_verifier='d' * 64,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


@pytest.mark.parametrize(
    'missing_field',
    (
        'content_hmac_schema_version',
        'assistant_message_content_hmac',
        'content_hmac_key_version',
        'content_hmac_key_material_verifier',
        'content_origin',
        'content_origin_hmac',
        'rag_result_hmac',
        'linked_agent_run_id',
    ),
)
def test_rag_v2_exact_message_rejects_every_partial_integrity_shape(
    missing_field: str,
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        conversation = models.AssistantConversation(user_id='owner', title='title')
        parent = _v2_parent(
            status='complete',
            run_record_phase='final',
            completed_at=datetime.now(UTC),
        )
        db.add_all([conversation, parent])
        db.flush()
        values: dict[str, object] = {
            'conversation_id': conversation.id,
            'role': 'assistant',
            'content': 'exact',
            'evidence_contract_version': 'none-v1',
            'serving_dependency_count': 0,
            'content_write_mode': 'rag_v2_exact',
            'content_hmac_schema_version': 'assistant-message-content-hmac:v1',
            'assistant_message_content_hmac': 'a' * 64,
            'content_hmac_key_version': 'runtime-key-v1',
            'content_hmac_key_material_verifier': 'b' * 64,
            'content_origin': 'rag_canned',
            'content_origin_hmac': 'c' * 64,
            'rag_result_hmac': 'd' * 64,
            'linked_agent_run_id': parent.id,
        }
        values[missing_field] = None
        db.add(models.AssistantMessage(**values))
        with pytest.raises(IntegrityError):
            db.commit()


def _readiness(**overrides: object) -> models.RagProviderReadiness:
    values: dict[str, object] = {
        'authority_id': 1,
        'component': 'query_embedding',
        'provider': 'openai',
        'model': 'text-embedding-3-small',
        'reasoning_or_config_identity': 'rag-query-embedding-config:v1',
        'active': True,
        'authorized_model_config_version': 'rag-query-embedding-config:v1',
        'authorized_model_config_snapshot_hmac': 'a' * 64,
        'authorized_cost_policy_version': 'rag-query-embedding-cost:v1',
        'authorized_token_estimator_version': (
            'openai-cl100k-text-embedding-3-small:v1'
        ),
        'authorized_fingerprint_key_version': 'runtime-key-v1',
        'authorized_fingerprint_key_material_verifier': 'b' * 64,
        'authorized_policy_snapshot_hmac': 'c' * 64,
        'state': 'ready',
        'state_version': 1,
        'family_safety_generation': 0,
    }
    values.update(overrides)
    return models.RagProviderReadiness(**values)


def test_readiness_rejects_partial_overrun_and_reset_attribution() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(
            models.RagProviderSafetyAuthority(
                id=1,
                authority_uuid='00000000-0000-0000-0000-000000000001',
                designated_environment_id='test',
                global_safety_generation=0,
                envelope_digest='a' * 64,
                fingerprint_key_version='runtime-key-v1',
                fingerprint_key_material_verifier='b' * 64,
            )
        )
        db.commit()

        db.add(_readiness(overrun_agent_run_id=123))
        with pytest.raises(IntegrityError):
            db.commit()


def test_provider_transition_rejects_partial_family_snapshot() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(
            models.RagProviderSafetyTransition(
                authority_id=1,
                readiness_id=1,
                global_safety_generation=0,
                transition_kind='bootstrap',
                prior_state=None,
                new_state=None,
                prior_state_version=None,
                new_state_version=1,
                prior_family_safety_generation=None,
                new_family_safety_generation=0,
                envelope_digest='a' * 64,
                reviewed_transition_reference_hmac='b' * 64,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(_readiness(reset_by='reviewer'))
        with pytest.raises(IntegrityError):
            db.commit()


@pytest.mark.parametrize(
    (
        'generation',
        'kind',
        'readiness_id',
        'prior_state',
        'new_state',
        'prior_version',
        'new_version',
        'prior_family_generation',
        'new_family_generation',
    ),
    (
        (0, 'rebind_required', 1, 'ready', 'rebind_required', 1, 2, 0, 0),
        (1, 'bootstrap', None, None, None, None, None, None, None),
    ),
)
def test_provider_transition_bootstrap_exists_iff_generation_zero(
    generation: int,
    kind: str,
    readiness_id: int | None,
    prior_state: str | None,
    new_state: str | None,
    prior_version: int | None,
    new_version: int | None,
    prior_family_generation: int | None,
    new_family_generation: int | None,
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    assert {
        constraint.name
        for constraint in models.RagProviderSafetyTransition.__table__.constraints
    } >= {'ck_rag_provider_safety_transition_bootstrap_generation'}
    with Session(engine) as db:
        db.add(
            models.RagProviderSafetyTransition(
                authority_id=1,
                readiness_id=readiness_id,
                global_safety_generation=generation,
                transition_kind=kind,
                prior_state=prior_state,
                new_state=new_state,
                prior_state_version=prior_version,
                new_state_version=new_version,
                prior_family_safety_generation=prior_family_generation,
                new_family_safety_generation=new_family_generation,
                envelope_digest='a' * 64,
                reviewed_transition_reference_hmac='b' * 64,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


@pytest.mark.parametrize(
    ('key1', 'key2', 'namespace', 'digest', 'payload'),
    (
        (2**31, 0, 'static', 'a' * 64, b'{}'),
        (0, -(2**31) - 1, 'dynamic', 'a' * 64, b'{}'),
        (0, 0, 'unknown', 'a' * 64, b'{}'),
        (0, 0, 'static', 'a' * 63, b'{}'),
        (0, 0, 'static', 'a' * 64, b''),
    ),
)
def test_advisory_registry_rejects_out_of_range_or_noncanonical_shape(
    key1: int,
    key2: int,
    namespace: str,
    digest: str,
    payload: bytes,
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(
            models.RagAdvisoryLockKey(
                key1=key1,
                key2=key2,
                identity_namespace=namespace,
                lock_identity_digest=digest,
                lock_identity_canonical_bytes=payload,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


@pytest.mark.parametrize(
    'missing_field',
    (
        'overrun_agent_run_id',
        'overrun_input_tokens',
        'overrun_output_tokens',
        'overrun_cost_usd',
        'overrun_observed_at',
    ),
)
def test_provider_overrun_attribution_rejects_each_missing_field(
    missing_field: str,
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        parent = _v2_parent()
        db.add(parent)
        db.flush()
        values: dict[str, object] = {
            'overrun_agent_run_id': parent.id,
            'overrun_input_tokens': 7,
            'overrun_output_tokens': 3,
            'overrun_cost_usd': Decimal('0.010000'),
            'overrun_observed_at': datetime.now(UTC),
            'state': 'blocked_overrun',
        }
        values[missing_field] = None
        db.add(_readiness(**values))
        with pytest.raises(IntegrityError):
            db.commit()


def _assembled_message_values(conversation_id: int, run_id: int) -> dict[str, object]:
    return {
        'conversation_id': conversation_id,
        'role': 'assistant',
        'content': 'assembled',
        'evidence_contract_version': 'assistant-evidence:v1',
        'serving_dependency_count': 1,
        'content_write_mode': 'rag_v2_exact',
        'content_hmac_schema_version': 'assistant-message-content-hmac:v1',
        'assistant_message_content_hmac': 'a' * 64,
        'content_hmac_key_version': 'runtime-key-v1',
        'content_hmac_key_material_verifier': 'b' * 64,
        'content_origin': 'rag_assembled',
        'content_origin_hmac': 'c' * 64,
        'rag_result_hmac': 'd' * 64,
        'linked_agent_run_id': run_id,
        'dependency_set_hmac_schema_version': 'assistant-dependency-set-hmac:v2',
        'dependency_set_hmac': 'e' * 64,
        'parent_selected_evidence_projection_hmac': 'f' * 64,
        'model_influence_set_hmac': '1' * 64,
    }


@pytest.mark.parametrize(
    'missing_field',
    (
        'evidence_contract_version',
        'serving_dependency_count',
        'dependency_set_hmac_schema_version',
        'dependency_set_hmac',
        'parent_selected_evidence_projection_hmac',
        'model_influence_set_hmac',
    ),
)
def test_assembled_message_rejects_each_nullable_required_field(
    missing_field: str,
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        conversation = models.AssistantConversation(user_id='owner', title='title')
        parent = _v2_parent(
            status='complete',
            run_record_phase='final',
            completed_at=datetime.now(UTC),
        )
        db.add_all([conversation, parent])
        db.flush()
        values = _assembled_message_values(conversation.id, parent.id)
        values[missing_field] = None
        db.add(models.AssistantMessage(**values))
        with pytest.raises(IntegrityError):
            db.commit()


@pytest.mark.parametrize('contaminating_field', ('rag_result_hmac', 'linked_agent_run_id'))
def test_legacy_trimmed_integrity_rejects_rag_v2_only_fields(
    contaminating_field: str,
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        conversation = models.AssistantConversation(user_id='owner', title='title')
        parent = _v2_parent()
        db.add_all([conversation, parent])
        db.flush()
        values = _assembled_message_values(conversation.id, parent.id)
        values.update(
            content_write_mode='legacy_trimmed',
            content_origin='legacy_evidence',
            model_influence_set_hmac=None,
            rag_result_hmac=None,
            linked_agent_run_id=None,
        )
        values[contaminating_field] = '2' * 64 if contaminating_field.endswith('hmac') else parent.id
        db.add(models.AssistantMessage(**values))
        with pytest.raises(IntegrityError):
            db.commit()


@pytest.mark.parametrize(
    'missing_field',
    (
        'dependency_role',
        'dependency_child_hmac',
        'model_content_hmac',
        'canonical_citation_projection_hmac',
        'selected_v1_citation_projection_hmac',
    ),
)
def test_v2_dependency_rejects_each_nullable_required_field(
    missing_field: str,
) -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        conversation = models.AssistantConversation(user_id='owner', title='title')
        message = models.AssistantMessage(
            **_assembled_message_values(1, 1),
        )
        db.add(conversation)
        db.flush()
        message.conversation_id = conversation.id
        db.add(message)
        db.flush()
        values: dict[str, object] = {
            'assistant_message_id': message.id,
            'candidate_ordinal': 0,
            'serving_document_id': 'chunk:1',
            'dependency_kind': 'raw_chunk',
            'dependency_set_hmac': 'e' * 64,
            'serving_content_hash': '3' * 64,
            'permission_level': 'internal',
            'fingerprint_key_version': 'runtime-key-v1',
            'fingerprint_key_material_verifier': 'b' * 64,
            'document_chunk_id': 1,
            'document_version_id': 1,
            'source_id': 1,
            'parser_run_id': 1,
            'server_content_signature_schema': 'server-source-content:v1',
            'server_content_signature': '4' * 64,
            'legacy_human_base': False,
            'dependency_serving_scope': 'rag_v2',
            'dependency_role': 'selected_citation',
            'dependency_child_hmac': '5' * 64,
            'model_content_hmac': '6' * 64,
            'canonical_citation_projection_hmac': '7' * 64,
            'selected_v1_citation_projection_hmac': '8' * 64,
            'serving_identity_hmac': '9' * 64,
            'serving_version_fingerprint': 'a' * 64,
            'support_mode': 'source_observation',
        }
        values[missing_field] = None
        db.add(models.AssistantMessageEvidenceDependency(**values))
        with pytest.raises(IntegrityError):
            db.commit()
