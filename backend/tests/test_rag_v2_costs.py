from __future__ import annotations

import copy
import pickle
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_cost_ledger import RagCostLedger, RagCostLedgerError
from backend.app.agent_runtime.rag_runtime_contracts import (
    ADMISSION_SOURCE_WINDOWS,
    TERMINAL_SOURCE_WINDOWS,
    AuthorizedProviderPolicySnapshot,
    RagRunAdmission,
    StrictProviderOutcome,
)
from backend.app.db.base import Base
from backend.app.rag.retrieval import PreparedPaidCallBudget, StrictProviderUsage


def _snapshot(component: str) -> AuthorizedProviderPolicySnapshot:
    query = component == 'query_embedding'
    return AuthorizedProviderPolicySnapshot(
        component=component,
        provider='openai',
        model='text-embedding-3-small' if query else 'gpt-5.4-mini-2026-03-17',
        reasoning_or_config_identity='dimensions:1536' if query else 'reasoning:none',
        authorized_model_config_version=(
            'rag-query-embedding-config:v1' if query else 'rag-answer-model-config:v1'
        ),
        authorized_model_config_snapshot_hmac='a' * 64,
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
        authorized_policy_snapshot_hmac='c' * 64,
    )


def _budget(component: str, reserve: str) -> PreparedPaidCallBudget:
    return PreparedPaidCallBudget(
        component=component,
        estimated_input_tokens=10,
        maximum_output_tokens=0 if component == 'query_embedding' else 100,
        reserved_cost_usd=Decimal(reserve),
        cost_policy_snapshot_hmac='c' * 64,
        estimator_input_hmac='d' * 64,
    )


def _ledger(commits: list[str] | None = None) -> RagCostLedger:
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    return RagCostLedger(
        Session(engine),
        identity_secret=b'task-12-test-identity-secret',
        after_commit=(None if commits is None else lambda: commits.append('commit')),
    )


def _admit(ledger: RagCostLedger, run_id: int) -> None:
    ledger.create_admission(
        agent_run_id=run_id,
        surface='ask',
        mode='enforce',
        cutover_stage='ask',
        configured_backend='keyword',
        query_context_version='direct-query:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac='4' * 64,
        source_window='rag-v2:admission:enforce:ask:keyword',
        components=(
            (_snapshot('query_embedding'), _budget('query_embedding', '0.000010')),
            (_snapshot('answer_generation'), _budget('answer_generation', '0.002000')),
        ),
    )


def test_literal_sets_are_exact_and_admission_commits_exact_two_children():
    assert len(ADMISSION_SOURCE_WINDOWS) == 9
    assert len(TERMINAL_SOURCE_WINDOWS) == 18
    commits: list[str] = []
    ledger = _ledger(commits)
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
        admission_cache_identity_hmac='4' * 64,
        source_window='rag-v2:admission:enforce:ask:keyword',
        components=(
            (_snapshot('query_embedding'), _budget('query_embedding', '0.000001')),
            (_snapshot('answer_generation'), _budget('answer_generation', '0.001234')),
        ),
    )
    assert type(admission) is RagRunAdmission
    assert admission.component_order == ('query_embedding', 'answer_generation')
    assert admission.total_reserved_cost_usd == Decimal('0.001235')
    assert commits == ['commit']


def test_claim_is_committed_before_opaque_one_use_grant_and_actual_replaces_reserve():
    commits: list[str] = []
    ledger = _ledger(commits)
    _admit(ledger, 9)
    grant = ledger.claim_component(
        run_id=9,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    assert commits == ['commit', 'commit']
    assert grant.reserved_cost_usd == Decimal('0.000010')
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises((TypeError, ValueError)):
            operation(grant)
    grant.consume_at_dispatch()
    with pytest.raises(RagCostLedgerError):
        grant.consume_at_dispatch()

    final = ledger.finalize_component(
        grant=grant,
        outcome=StrictProviderOutcome(
            component='query_embedding',
            classification='validated_success',
            terminal_outcome='component_succeeded',
            provider_dispatch_started=True,
            provider_response_received=True,
            strict_usage=StrictProviderUsage(20, 0, 20),
            actual_cost_usd=Decimal('0.000003'),
            safety_action='unchanged',
        ),
    )
    assert final.charge_basis == 'actual'
    assert final.charged_cost_usd == Decimal('0.000003')


def test_response_less_uses_full_reserve_and_known_overrun_is_unclamped():
    ledger = _ledger()
    _admit(ledger, 10)
    query = ledger.claim_component(
        run_id=10,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    response_less = ledger.finalize_component(
        grant=query,
        outcome=StrictProviderOutcome(
            component='query_embedding',
            classification='response_less_failure',
            terminal_outcome='model_provider_failed',
            provider_dispatch_started=True,
            provider_response_received=False,
            strict_usage=None,
            actual_cost_usd=None,
            safety_action='unchanged',
        ),
    )
    assert response_less.charge_basis == 'reserved'
    assert response_less.charged_cost_usd == Decimal('0.000010')

    answer = ledger.claim_component(
        run_id=10,
        component='answer_generation',
        prepared=_budget('answer_generation', '0.002000'),
    )
    overrun = ledger.finalize_component(
        grant=answer,
        outcome=StrictProviderOutcome(
            component='answer_generation',
            classification='known_overrun',
            terminal_outcome='provider_usage_overrun',
            provider_dispatch_started=True,
            provider_response_received=True,
            strict_usage=StrictProviderUsage(5000, 2000, 7000),
            actual_cost_usd=Decimal('0.020001'),
            safety_action='block_overrun',
        ),
    )
    assert overrun.overrun is True
    assert overrun.charged_cost_usd == Decimal('0.020001')
    assert ledger.total_charged_cost(10) == Decimal('0.020011')


def test_projectionless_failure_closes_exact_two_terminal_zero_children():
    ledger = _ledger()
    _admit(ledger, 11)
    terminal = ledger.finalize_projectionless_failure(
        run_id=11, outcome='retriever_not_configured'
    )
    assert terminal.run_record_phase == 'admission_only'
    assert terminal.source_window == 'rag-v2:admission:enforce:ask:keyword'
    assert terminal.cache_key == 'rag-v2-admission:' + '4' * 64
    assert terminal.terminal_identity_hmac is None
    assert all(
        final.attempted is False
        and final.dispatch_count == 0
        and final.terminal_outcome is None
        and final.reserved_cost_usd == Decimal('0.000000')
        and final.charged_cost_usd == Decimal('0.000000')
        for final in terminal.component_finals
    )
