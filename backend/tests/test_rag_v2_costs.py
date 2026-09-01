from __future__ import annotations

import copy
import pickle
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_cost_ledger import (
    RagCostLedger,
    RagCostLedgerError,
    _assemble_rag_cost_ledger,
)
from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService
from backend.app.agent_runtime.rag_runtime_contracts import (
    ADMISSION_SOURCE_WINDOWS,
    TERMINAL_SOURCE_WINDOWS,
    AuthorizedProviderPolicySnapshot,
    RagRunAdmission,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    _issue_classified_provider_observation as StrictProviderOutcome,
)
from backend.app.db.base import Base
from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.app.rag.retrieval import PreparedPaidCallBudget, StrictProviderUsage
from backend.tests.test_rag_v2_cost_policy import _policy

_TEST_COST_POLICY = _policy()


def _snapshot(
    component: str,
    policy: RagCostPolicy | None = None,
) -> AuthorizedProviderPolicySnapshot:
    query = component == 'query_embedding'
    config_hmac = (
        policy.query_embedding_model_config_snapshot_hmac
        if policy is not None and query
        else policy.answer_model_config_snapshot_hmac
        if policy is not None
        else 'a' * 64
    )
    policy_hmac = (
        policy.authorized_policy_snapshot_hmac(component)
        if policy is not None
        else 'c' * 64
    )
    return AuthorizedProviderPolicySnapshot(
        component=component,
        provider='openai',
        model='text-embedding-3-small' if query else 'gpt-5.4-mini-2026-03-17',
        reasoning_or_config_identity='dimensions:1536' if query else 'reasoning:none',
        authorized_model_config_version=(
            'rag-query-embedding-config:v1' if query else 'rag-answer-model-config:v1'
        ),
        authorized_model_config_snapshot_hmac=config_hmac,
        authorized_cost_policy_version=(
            'rag-query-embedding-cost:v1' if query else 'rag-answer-cost:v1'
        ),
        authorized_token_estimator_version=(
            'openai-cl100k-text-embedding-3-small:v1'
            if query
            else 'openai-o200k-rag-answer:v1'
        ),
        fingerprint_key_version='test-v1',
        fingerprint_key_material_verifier='b' * 64,
        authorized_policy_snapshot_hmac=policy_hmac,
    )


def _budget(component: str, reserve: str) -> PreparedPaidCallBudget:
    token_shapes = {
        ('query_embedding', '0.000001'): (10, 0),
        ('query_embedding', '0.000010'): (500, 0),
        ('answer_generation', '0.001234'): (1, 274),
        ('answer_generation', '0.002000'): (2, 444),
    }
    estimated_input_tokens, maximum_output_tokens = token_shapes[(component, reserve)]
    return PreparedPaidCallBudget(
        component=component,
        estimated_input_tokens=estimated_input_tokens,
        maximum_output_tokens=maximum_output_tokens,
        reserved_cost_usd=Decimal(reserve),
        cost_policy_snapshot_hmac=(
            _TEST_COST_POLICY.authorized_policy_snapshot_hmac(component)
        ),
        estimator_input_hmac='d' * 64,
    )


def _ledger(
    tmp_path: Path,
    commits: list[str] | None = None,
    safety_events: list[object] | None = None,
    provider_client: object | None = None,
    cost_policy: RagCostPolicy | None = None,
) -> RagCostLedger:
    selected_policy = cost_policy or _TEST_COST_POLICY
    engine = create_engine(
        f'sqlite+pysqlite:///{(tmp_path / (uuid4().hex + ".db")).as_posix()}'
    )
    Base.metadata.create_all(engine)
    service = RagProviderSafetyService(
        latch_path=tmp_path / f'{uuid4().hex}.json',
        identity_secret=b'task-12-test-identity-secret',
        designated_environment_id='test',
    )
    with engine.connect() as connection:
        service.bootstrap(
            connection,
            (
                _snapshot('query_embedding', selected_policy),
                _snapshot('answer_generation', selected_policy),
            ),
            reviewed_transition_reference_hmac='9' * 64,
        )
    return _assemble_rag_cost_ledger(
        Session(engine),
        identity_secret=b'task-12-test-identity-secret',
        after_commit=(None if commits is None else lambda: commits.append('commit')),
        cost_policy=selected_policy,
        provider_safety=service,
        provider_connection_factory=engine.connect,
        designated_environment_id='test',
        designated_host_id='pytest-host',
    )


def _admit(
    ledger: RagCostLedger,
    run_id: int,
    *,
    backend: str = 'pgvector',
) -> None:
    ledger.create_admission(
        agent_run_id=run_id,
        surface='ask',
        mode='enforce',
        cutover_stage='ask',
        configured_backend=backend,
        query_context_version='direct-query:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac=None,
        source_window=f'rag-v2:admission:enforce:ask:{backend}',
        components=(
            (
                _snapshot('query_embedding', _TEST_COST_POLICY),
                _budget('query_embedding', '0.000010'),
            ),
            (
                _snapshot('answer_generation', _TEST_COST_POLICY),
                _budget('answer_generation', '0.002000'),
            ),
        ),
    )


def test_literal_sets_are_exact_and_admission_commits_exact_two_children(
    tmp_path: Path,
):
    assert len(ADMISSION_SOURCE_WINDOWS) == 9
    assert len(TERMINAL_SOURCE_WINDOWS) == 18
    commits: list[str] = []
    ledger = _ledger(tmp_path, commits)
    admission = ledger.create_admission(
        agent_run_id=7,
        surface='ask',
        mode='enforce',
        cutover_stage='ask',
        configured_backend='keyword',
        query_context_version='direct-query:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac=None,
        source_window='rag-v2:admission:enforce:ask:keyword',
        components=(
            (
                _snapshot('query_embedding', _TEST_COST_POLICY),
                _budget('query_embedding', '0.000001'),
            ),
            (
                _snapshot('answer_generation', _TEST_COST_POLICY),
                _budget('answer_generation', '0.001234'),
            ),
        ),
    )
    assert type(admission) is RagRunAdmission
    assert admission.component_order == ('query_embedding', 'answer_generation')
    assert admission.total_reserved_cost_usd == Decimal('0.001235')
    assert admission.query_embedding_provider_policy_snapshot_hmac is None
    assert admission.answer_provider_policy_snapshot_hmac is not None
    assert commits == ['commit']


def test_claim_is_committed_before_opaque_one_use_grant_and_actual_replaces_reserve(
    tmp_path: Path,
):
    commits: list[str] = []
    ledger = _ledger(tmp_path, commits)
    _admit(ledger, 9)
    grant = ledger.claim_component(
        run_id=9,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    assert commits == ['commit', 'commit']
    assert grant.reserved_cost_usd == Decimal('0.000010')
    parent = ledger._session.get(AgentRun, 9)
    assert parent is not None
    assert Decimal(parent.total_charged_cost_usd) == Decimal('0.000010')
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises((TypeError, ValueError)):
            operation(grant)
    grant.consume_at_dispatch()
    with pytest.raises(RagCostLedgerError):
        grant.consume_at_dispatch()

    final = ledger.finalize_component(
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
    assert final.charge_basis == 'actual'
    assert final.charged_cost_usd == Decimal('0.000001')


def test_response_less_uses_full_reserve_and_known_overrun_is_unclamped(
    tmp_path: Path,
):
    safety_events: list[object] = []
    ledger = _ledger(tmp_path, safety_events=safety_events)
    _admit(ledger, 10)
    query = ledger.claim_component(
        run_id=10,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    query.consume_at_dispatch()
    response_less = ledger.finalize_component(
        grant=query,
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
    assert response_less.charge_basis == 'reserved'
    assert response_less.charged_cost_usd == Decimal('0.000010')

    overrun_ledger = _ledger(tmp_path)
    _admit(overrun_ledger, 1010)
    query_ok = overrun_ledger.claim_component(
        run_id=1010,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    overrun_ledger.consume_committed_grant(query_ok)
    overrun_ledger.finalize_component(
        grant=query_ok,
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
    answer = overrun_ledger.claim_component(
        run_id=1010,
        component='answer_generation',
        prepared=_budget('answer_generation', '0.002000'),
    )
    answer.consume_at_dispatch()
    overrun = overrun_ledger.finalize_component(
        grant=answer,
        observation=StrictProviderOutcome(
            component='answer_generation',
            classification='known_overrun',
            terminal_outcome='provider_usage_overrun',
            provider_dispatch_started=True,
            provider_response_received=True,
            strict_usage=StrictProviderUsage(5000, 2000, 7000),
            actual_cost_usd=Decimal('0.012750'),
            safety_action='block_overrun',
        ),
    )
    assert overrun.overrun is True
    assert overrun.charged_cost_usd == Decimal('0.012750')
    assert overrun_ledger.total_charged_cost(1010) == Decimal('0.012751')


def test_failed_selected_component_closes_impossible_sibling_and_parent_final(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    _admit(ledger, 122)
    grant = ledger.claim_component(
        run_id=122,
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
    assert final.parent_run_record_phase == 'final'
    with pytest.raises(RagCostLedgerError):
        ledger.claim_component(
            run_id=122,
            component='answer_generation',
            prepared=_budget('answer_generation', '0.002000'),
        )


def test_claim_rejects_route_unused_query_component(tmp_path: Path):
    ledger = _ledger(tmp_path)
    _admit(ledger, 123, backend='keyword')

    with pytest.raises(RagCostLedgerError, match='route'):
        ledger.claim_component(
            run_id=123,
            component='query_embedding',
            prepared=_budget('query_embedding', '0.000010'),
        )


def test_pgvector_answer_cannot_claim_before_query_terminal(tmp_path: Path):
    ledger = _ledger(tmp_path)
    ledger.create_admission(
        agent_run_id=124,
        surface='ask',
        mode='enforce',
        cutover_stage='ask',
        configured_backend='pgvector',
        query_context_version='direct-query:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac=None,
        source_window='rag-v2:admission:enforce:ask:pgvector',
        components=(
            (_snapshot('query_embedding', _TEST_COST_POLICY),
             _budget('query_embedding', '0.000010')),
            (_snapshot('answer_generation', _TEST_COST_POLICY),
             _budget('answer_generation', '0.002000')),
        ),
    )

    with pytest.raises(RagCostLedgerError, match='order'):
        ledger.claim_component(
            run_id=124,
            component='answer_generation',
            prepared=_budget('answer_generation', '0.002000'),
        )


def test_projectionless_failure_closes_exact_two_terminal_zero_children(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    _admit(ledger, 11, backend='keyword')
    terminal = ledger.finalize_projectionless_failure(
        run_id=11, outcome='abandoned_unknown'
    )
    assert terminal.run_record_phase == 'admission_only'
    assert terminal.source_window == 'rag-v2:admission:enforce:ask:keyword'
    assert terminal.cache_key == (
        'rag-v2-admission:' + terminal.admission_cache_identity_hmac
    )
    assert terminal.terminal_identity_hmac is None
    assert all(
        final.attempted is False
        and final.dispatch_count == 0
        and final.terminal_outcome is None
        and final.reserved_cost_usd == Decimal('0.000000')
        and final.charged_cost_usd == Decimal('0.000000')
        for final in terminal.component_finals
    )


def test_unconsumed_forged_outcome_cannot_finalize_or_trigger_ignored_safety(
    tmp_path: Path,
):
    safety_events = []
    ledger = _ledger(tmp_path, safety_events=safety_events)
    _admit(ledger, 12)
    grant = ledger.claim_component(
        run_id=12,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    forged = StrictProviderOutcome(
        component='query_embedding',
        classification='response_less_failure',
        terminal_outcome='component_succeeded',
        provider_dispatch_started=True,
        provider_response_received=False,
        strict_usage=None,
        actual_cost_usd=None,
        safety_action='block_overrun',
    )
    with pytest.raises((RagCostLedgerError, ValueError)):
        ledger.finalize_component(grant=grant, observation=forged)
    assert safety_events == []


@pytest.mark.parametrize(
    (
        'classification', 'terminal', 'usage', 'actual', 'action',
        'expected_basis',
    ),
    (
        (
            'validated_success', 'component_succeeded',
            StrictProviderUsage(20, 0, 20), Decimal('0.000001'), 'unchanged',
            'actual',
        ),
        (
            'response_less_failure', 'retriever_unavailable', None, None,
            'unchanged', 'reserved',
        ),
        (
            'response_identity_invalid', 'provider_response_identity_invalid',
            StrictProviderUsage(20, 0, 20), Decimal('0.000001'),
            'block_remediation', 'actual',
        ),
        (
            'response_identity_invalid', 'provider_response_identity_invalid',
            None, None, 'block_remediation', 'reserved',
        ),
        (
            'embedding_payload_invalid', 'provider_embedding_payload_invalid',
            StrictProviderUsage(20, 0, 20), Decimal('0.000001'),
            'block_remediation', 'actual',
        ),
        (
            'usage_contract_invalid', 'provider_safety_unavailable', None, None,
            'block_remediation', 'reserved',
        ),
        (
            'usage_storage_invalid', 'provider_safety_unavailable', None, None,
            'block_remediation', 'reserved',
        ),
        (
            'citation_validation_failed', 'citation_validation_failed',
            StrictProviderUsage(20, 0, 20), Decimal('0.000001'), 'unchanged',
            'actual',
        ),
        (
            'evidence_validation_failed', 'evidence_unavailable',
            StrictProviderUsage(20, 0, 20), Decimal('0.000001'), 'unchanged',
            'actual',
        ),
    ),
)
def test_accounting_classification_matrix_is_policy_and_safety_owned(
    tmp_path: Path,
    classification: str,
    terminal: str,
    usage: StrictProviderUsage | None,
    actual: Decimal | None,
    action: str,
    expected_basis: str,
):
    ledger = _ledger(tmp_path)
    _admit(ledger, 120)
    grant = ledger.claim_component(
        run_id=120,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    ledger.consume_committed_grant(grant)

    final = ledger.finalize_component(
        grant=grant,
        observation=StrictProviderOutcome(
            component='query_embedding',
            classification=classification,
            terminal_outcome=terminal,
            provider_dispatch_started=True,
            provider_response_received=classification != 'response_less_failure',
            strict_usage=usage,
            actual_cost_usd=actual,
            safety_action=action,
        ),
    )

    assert final.charge_basis == expected_basis
    assert final.charged_cost_usd == (
        Decimal('0.000010') if actual is None else actual
    )


def test_known_overrun_must_be_real_before_safety_is_mutated(tmp_path: Path):
    ledger = _ledger(tmp_path)
    _admit(ledger, 121)
    grant = ledger.claim_component(
        run_id=121,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    ledger.consume_committed_grant(grant)
    with pytest.raises(ValueError, match='overrun classification'):
        ledger.finalize_component(
            grant=grant,
            observation=StrictProviderOutcome(
                component='query_embedding',
                classification='known_overrun',
                terminal_outcome='provider_usage_overrun',
                provider_dispatch_started=True,
                provider_response_received=True,
                strict_usage=StrictProviderUsage(20, 0, 20),
                actual_cost_usd=Decimal('0.000001'),
                safety_action='block_overrun',
            ),
        )


def test_second_terminal_child_moves_parent_to_pending_projection_with_owner_fence(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    _admit(ledger, 13)
    for component, reserve, usage, actual in (
        ('query_embedding', '0.000010', StrictProviderUsage(20, 0, 20), Decimal('0.000001')),
        ('answer_generation', '0.002000', StrictProviderUsage(1, 10, 11), Decimal('0.000046')),
    ):
        grant = ledger.claim_component(
            run_id=13, component=component, prepared=_budget(component, reserve)
        )
        grant.consume_at_dispatch()
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
    assert final.parent_run_record_phase == 'cost_finalized_pending_projection'


def test_shadow_pgvector_success_closes_unused_answer_component_terminal_zero(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    ledger.create_admission(
        agent_run_id=127,
        surface='search',
        mode='shadow',
        cutover_stage='search',
        configured_backend='pgvector',
        query_context_version='direct-query:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac=None,
        source_window='rag-v2:admission:shadow:search:pgvector',
        components=(
            (_snapshot('query_embedding', _TEST_COST_POLICY), _budget('query_embedding', '0.000010')),
            (_snapshot('answer_generation', _TEST_COST_POLICY), _budget('answer_generation', '0.002000')),
        ),
    )
    grant = ledger.claim_component(
        run_id=127,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    ledger.consume_committed_grant(grant)

    final = ledger.finalize_component(
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

    children = tuple(ledger._session.scalars(
        select(AgentRunCostComponent)
        .where(AgentRunCostComponent.agent_run_id == 127)
        .order_by(AgentRunCostComponent.component_ordinal)
    ))
    assert final.parent_run_record_phase == 'cost_finalized_pending_projection'
    assert children[1].dispatch_state == 'terminal'
    assert children[1].attempted is False
    assert children[1].charged_cost_usd == Decimal('0.000000')
    assert final.projection_owner_fence_hmac is not None


def test_pre_send_refusal_closes_selected_and_sibling_terminal_zero(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    _admit(ledger, 130)

    terminal = ledger.finalize_pre_send_refusal(
        run_id=130,
        component='query_embedding',
        outcome='provider_safety_unavailable',
    )

    assert terminal.run_record_phase == 'final'
    assert terminal.outcome == 'provider_safety_unavailable'
    assert all(
        child.attempted is False
        and child.dispatch_count == 0
        and child.charge_basis == 'zero'
        and child.terminal_outcome is None
        for child in terminal.component_finals
    )


def test_dead_dispatch_recovery_never_redispatches_and_preserves_prior_actual(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    _admit(ledger, 131)
    query = ledger.claim_component(
        run_id=131,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    ledger.consume_committed_grant(query)
    ledger.finalize_component(
        grant=query,
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
    answer = ledger.claim_component(
        run_id=131,
        component='answer_generation',
        prepared=_budget('answer_generation', '0.002000'),
    )

    attestation = ledger._expected_dead_process_attestation_hmac(
        run_id=131,
        dead_process_instance_hmac=ledger.process_instance_hmac,
    )
    with pytest.raises(RagCostLedgerError, match='dead-process'):
        ledger.recover_incomplete_run(run_id=131)
    terminal = ledger.recover_incomplete_run(
        run_id=131,
        dead_process_attestation_hmac=attestation,
    )

    assert terminal.status == 'failed'
    assert terminal.run_record_phase == 'admission_only'
    assert terminal.outcome == 'abandoned_unknown'
    assert terminal.total_charged_cost_usd == Decimal('0.002001')
    assert terminal.component_finals[0].charged_cost_usd == Decimal('0.000001')
    assert terminal.component_finals[1].dispatch_state == 'abandoned_unknown'
    assert terminal.component_finals[1].charged_cost_usd == Decimal('0.002000')
    with pytest.raises(RagCostLedgerError):
        ledger.consume_committed_grant(answer)


def test_reviewed_intercomponent_recovery_preserves_actual_and_zeroes_sibling(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    _admit(ledger, 133)
    query = ledger.claim_component(
        run_id=133,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    ledger.consume_committed_grant(query)
    ledger.finalize_component(
        grant=query,
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
        run_id=133,
        dead_process_instance_hmac=ledger.process_instance_hmac,
    )

    terminal = ledger.recover_incomplete_run(
        run_id=133,
        dead_process_attestation_hmac=attestation,
    )

    assert terminal.run_record_phase == 'admission_only'
    assert terminal.total_charged_cost_usd == Decimal('0.000001')
    assert terminal.component_finals[0].charge_basis == 'actual'
    assert terminal.component_finals[1].charge_basis == 'zero'


def test_pending_projection_recovery_requires_exact_owner_fence(tmp_path: Path):
    ledger = _ledger(tmp_path)
    _admit(ledger, 132)
    final = None
    for component, reserve, usage, actual in (
        ('query_embedding', '0.000010', StrictProviderUsage(20, 0, 20), Decimal('0.000001')),
        ('answer_generation', '0.002000', StrictProviderUsage(1, 10, 11), Decimal('0.000046')),
    ):
        grant = ledger.claim_component(
            run_id=132, component=component, prepared=_budget(component, reserve)
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
    assert final is not None and final.projection_owner_fence_hmac is not None
    with pytest.raises(RagCostLedgerError, match='owner fence'):
        ledger.recover_incomplete_run(
            run_id=132,
            projection_owner_fence_hmac='0' * 64,
            expected_runtime_cost_snapshot_hmac=final.runtime_cost_snapshot_hmac,
        )
    with pytest.raises(RagCostLedgerError, match='cost snapshot'):
        ledger.recover_incomplete_run(
            run_id=132,
            projection_owner_fence_hmac=final.projection_owner_fence_hmac,
            expected_runtime_cost_snapshot_hmac='0' * 64,
        )

    terminal = ledger.recover_incomplete_run(
        run_id=132,
        projection_owner_fence_hmac=final.projection_owner_fence_hmac,
        expected_runtime_cost_snapshot_hmac=final.runtime_cost_snapshot_hmac,
    )
    assert terminal.outcome == 'persistence_failed'
    assert terminal.run_record_phase == 'final'
    assert terminal.total_charged_cost_usd == Decimal('0.000047')


@pytest.mark.parametrize(
    ('mode', 'surface', 'backend', 'window'),
    (
        ('enforce', 'ask', 'keyword', 'rag-v2:admission:shadow:assistant:pgvector'),
        ('shadow', 'assistant', 'pgvector', 'rag-v2:admission:enforce:ask:keyword'),
    ),
)
def test_admission_rejects_source_window_not_derived_from_route(
    mode: str, surface: str, backend: str, window: str, tmp_path: Path
):
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger.create_admission(
            agent_run_id=14,
            surface=surface,
            mode=mode,
            cutover_stage=surface,
            configured_backend=backend,
            query_context_version='direct-query:v1',
            current_text_hmac='1' * 64,
            retrieval_query_hmac='2' * 64,
            security_scope_fingerprint='3' * 64,
            admission_cache_identity_hmac=None,
            source_window=window,
            components=(
                (
                    _snapshot('query_embedding', _TEST_COST_POLICY),
                    _budget('query_embedding', '0.000010'),
                ),
                (
                    _snapshot('answer_generation', _TEST_COST_POLICY),
                    _budget('answer_generation', '0.002000'),
                ),
            ),
        )


@pytest.mark.parametrize(
    ('mode', 'surface', 'stage', 'backend', 'context'),
    (
        ('enforce', 'search', 'ask', 'keyword', 'direct-query:v1'),
        ('enforce', 'assistant', 'search', 'keyword', 'assistant-context:v1'),
        ('shadow', 'ask', 'none', 'pgvector', 'direct-query:v1'),
        ('shadow', 'ask', 'ask', 'keyword', 'direct-query:v1'),
        ('enforce', 'assistant', 'assistant', 'keyword', 'direct-query:v1'),
        ('enforce', 'ask', 'ask', 'keyword', 'assistant-context:v1'),
    ),
)
def test_admission_rejects_inactive_stage_backend_and_context_combinations(
    tmp_path: Path,
    mode: str,
    surface: str,
    stage: str,
    backend: str,
    context: str,
):
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match='admission'):
        ledger.create_admission(
            agent_run_id=140,
            surface=surface,
            mode=mode,
            cutover_stage=stage,
            configured_backend=backend,
            query_context_version=context,
            current_text_hmac='1' * 64,
            retrieval_query_hmac='2' * 64,
            security_scope_fingerprint='3' * 64,
            admission_cache_identity_hmac=None,
            source_window=(
                f'rag-v2:admission:{mode}:{surface}:{backend}'
            ),
            components=(
                (
                    _snapshot('query_embedding', _TEST_COST_POLICY),
                    _budget('query_embedding', '0.000010'),
                ),
                (
                    _snapshot('answer_generation', _TEST_COST_POLICY),
                    _budget('answer_generation', '0.002000'),
                ),
            ),
        )


def test_shadow_pgvector_admission_exposes_only_query_provider_authority(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    admission = ledger.create_admission(
        agent_run_id=141,
        surface='assistant',
        mode='shadow',
        cutover_stage='assistant',
        configured_backend='pgvector',
        query_context_version='assistant-context:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac=None,
        source_window='rag-v2:admission:shadow:assistant:pgvector',
        components=(
            (
                _snapshot('query_embedding', _TEST_COST_POLICY),
                _budget('query_embedding', '0.000010'),
            ),
            (
                _snapshot('answer_generation', _TEST_COST_POLICY),
                _budget('answer_generation', '0.002000'),
            ),
        ),
    )
    assert admission.query_embedding_provider_policy_snapshot_hmac is not None
    assert admission.answer_provider_policy_snapshot_hmac is None
