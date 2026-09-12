from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_cost_ledger import (
    RagCostLedgerError,
    RagCostPersistenceError,
    _assemble_rag_cost_ledger,
)
from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.tests.test_rag_task14_transport_delivery import _embedding_case
from backend.tests.test_rag_v2_costs import _budget, _ledger, _snapshot


def _deferred(ledger, *, run_id=143, backend='keyword', surface='ask'):
    policy = ledger.cost_policy_authority
    assert hasattr(policy, 'reserve_answer_generation'), 'staged ceiling is missing'
    reserve = policy.reserve_answer_generation()
    ledger.create_admission(
        agent_run_id=run_id, surface=surface, mode='enforce', cutover_stage='assistant',
        configured_backend=backend, query_context_version='direct-query:v1',
        current_text_hmac='1' * 64, retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64, admission_cache_identity_hmac=None,
        source_window=f'rag-v2:admission:enforce:{surface}:{backend}',
        components=((_snapshot('query_embedding', policy), _budget('query_embedding', '0.000010')),
                    (_snapshot('answer_generation', policy), reserve)),
    )
    return reserve


def _rows(ledger, run_id=143):
    ledger._session.expire_all()
    return tuple(ledger._session.scalars(select(AgentRunCostComponent).where(
        AgentRunCostComponent.agent_run_id == run_id,
    ).order_by(AgentRunCostComponent.component_ordinal)))


def test_ceiling_is_not_dispatch_authority_until_concrete_budget_binding(tmp_path):
    ledger = _ledger(tmp_path)
    reserve = _deferred(ledger)
    assert reserve.reserved_cost_usd == Decimal('0.009804')
    concrete = _budget('answer_generation', '0.002000')
    with pytest.raises(RagCostLedgerError):
        ledger.claim_component(run_id=143, component='answer_generation', prepared=concrete)
    ledger._session.rollback()
    ledger.bind_answer_budget(run_id=143, prepared=concrete)
    query, answer = _rows(ledger)
    assert answer.reserved_cost_usd == Decimal('0.002000')
    assert answer.reserved_input_tokens == 2
    assert answer.reserved_output_tokens == 444
    assert query.dispatch_count == answer.dispatch_count == 0
    assert ledger.total_charged_cost(143) == Decimal('0.000000')
    grant = ledger.claim_component(run_id=143, component='answer_generation', prepared=concrete)
    assert grant.reserved_cost_usd == Decimal('0.002000')


@pytest.mark.parametrize('after_claim', (False, True))
def test_concrete_answer_budget_cannot_be_rebound(tmp_path, after_claim):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    concrete = _budget('answer_generation', '0.002000')
    ledger.bind_answer_budget(run_id=143, prepared=concrete)
    if after_claim:
        ledger.claim_component(run_id=143, component='answer_generation', prepared=concrete)
    with pytest.raises(RagCostLedgerError):
        ledger.bind_answer_budget(run_id=143, prepared=concrete)
    assert _rows(ledger)[1].reserved_cost_usd == Decimal('0.002000')


@pytest.mark.parametrize('change', (
    {'reserved_cost_usd': Decimal('0.012001')},
    {'cost_policy_snapshot_hmac': 'f' * 64},
    {'estimated_input_tokens': 10001},
))
def test_fabricated_or_expanded_concrete_budget_has_zero_dispatch(tmp_path, change):
    ledger = _ledger(tmp_path)
    reserve = _deferred(ledger)
    with pytest.raises((ValueError, RagCostLedgerError)):
        ledger.bind_answer_budget(
            run_id=143, prepared=replace(_budget('answer_generation', '0.002000'), **change),
        )
    ledger._session.rollback()
    assert _rows(ledger)[1].reserved_cost_usd == reserve.reserved_cost_usd
    assert _rows(ledger)[1].dispatch_count == 0


def test_search_refuses_generation_reservation_before_admission_write(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match='unused'):
        _deferred(ledger, surface='search')
    assert ledger._session.get(AgentRun, 143) is None


def test_failed_binding_commit_cannot_produce_a_claim_or_retry(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    concrete = _budget('answer_generation', '0.002000')
    original = ledger._session.commit

    def fail():
        raise RuntimeError('commit failed')

    monkeypatch.setattr(ledger._session, 'commit', fail)
    with pytest.raises(RagCostPersistenceError, match='commit acknowledgement unavailable'):
        ledger.bind_answer_budget(run_id=143, prepared=concrete)
    monkeypatch.setattr(ledger._session, 'commit', original)
    with pytest.raises(RagCostLedgerError):
        ledger.bind_answer_budget(run_id=143, prepared=concrete)
    with pytest.raises(RagCostLedgerError):
        ledger.claim_component(run_id=143, component='answer_generation', prepared=concrete)
    assert _rows(ledger)[1].dispatch_count == 0


def test_stale_durable_reservation_cannot_be_bound(tmp_path):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    row = _rows(ledger)[1]
    row.reserved_input_tokens = 9999
    ledger._session.commit()
    with pytest.raises(RagCostLedgerError):
        ledger.bind_answer_budget(run_id=143, prepared=_budget('answer_generation', '0.002000'))
    assert _rows(ledger)[1].dispatch_count == 0


def test_concurrent_duplicate_binding_has_one_winner(tmp_path):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    ledger._session.rollback()

    def bind():
        try:
            ledger.bind_answer_budget(run_id=143, prepared=_budget('answer_generation', '0.002000'))
            return 'bound'
        except RagCostLedgerError:
            return 'refused'

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: bind(), range(2)))
    assert sorted(outcomes) == ['bound', 'refused']
    assert _rows(ledger)[1].dispatch_count == 0


def test_provider_free_pending_is_exact_two_zero_without_provider_authority(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    assert hasattr(ledger, 'commit_provider_free_pending'), 'provider-free pending entry is missing'

    def forbidden(*_args, **_kwargs):
        pytest.fail('provider-free pending consulted provider safety')

    monkeypatch.setattr(ledger.provider_safety_authority, 'require_ready', forbidden)
    monkeypatch.setattr(ledger.provider_safety_authority, 'finalization_barrier', forbidden)
    pending = ledger.commit_provider_free_pending(
        run_id=143, corpus_generation=5, vector_index_generation=None,
    )
    assert pending.parent_agent_run_id == 143
    assert pending.prepared_corpus_generation == 5
    parent = ledger._session.get(AgentRun, 143)
    assert parent.run_record_phase == 'cost_finalized_pending_projection'
    assert parent.status == 'running'
    assert parent.projection_owner_fence_hmac == pending.projection_owner_fence_hmac
    assert parent.metadata_['runtime_cost_snapshot_hmac'] == pending.terminal_cost_snapshot_hmac
    for row in _rows(ledger):
        assert row.dispatch_state == 'terminal'
        assert row.dispatch_count == 0
        assert row.reserved_cost_usd == row.charged_cost_usd == Decimal('0.000000')
    with pytest.raises(RagCostLedgerError):
        ledger.commit_provider_free_pending(run_id=143, corpus_generation=5, vector_index_generation=None)


def test_provider_free_pending_refuses_claimed_components(tmp_path):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    budget = _budget('answer_generation', '0.002000')
    ledger.bind_answer_budget(run_id=143, prepared=budget)
    ledger.claim_component(run_id=143, component='answer_generation', prepared=budget)
    assert hasattr(ledger, 'commit_provider_free_pending'), 'provider-free pending entry is missing'
    with pytest.raises(RagCostLedgerError):
        ledger.commit_provider_free_pending(run_id=143, corpus_generation=5, vector_index_generation=None)
    assert _rows(ledger)[1].dispatch_state == 'dispatching'
    assert _rows(ledger)[1].charged_cost_usd == Decimal('0.002000')


def test_embedding_only_pending_preserves_real_transport_charge(tmp_path):
    ledger, authority, grant, dispatch, client, _ = _embedding_case(tmp_path)
    delivered = authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    assert hasattr(ledger, 'commit_embedding_only_pending'), 'embedding-only pending entry is missing'
    pending = ledger.commit_embedding_only_pending(run_id=141, corpus_generation=1, vector_index_generation=1)
    query, answer = _rows(ledger, 141)
    assert query.charged_cost_usd == delivered.component_final.charged_cost_usd
    assert query.dispatch_count == 1
    assert answer.dispatch_count == 0
    assert answer.dispatch_state == 'terminal'
    assert answer.reserved_cost_usd == answer.charged_cost_usd == Decimal('0.000000')
    assert ledger._session.get(AgentRun, 141).metadata_['runtime_cost_snapshot_hmac'] == pending.terminal_cost_snapshot_hmac
    assert len(client.seen) == 1
    with pytest.raises(RagCostLedgerError):
        ledger.commit_embedding_only_pending(run_id=141, corpus_generation=1, vector_index_generation=1)


def test_failed_embedding_cannot_enter_safe_pending(tmp_path):
    ledger, authority, grant, dispatch, _, _ = _embedding_case(tmp_path, malformed=True)
    authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    assert hasattr(ledger, 'commit_embedding_only_pending'), 'embedding-only pending entry is missing'
    with pytest.raises(RagCostLedgerError):
        ledger.commit_embedding_only_pending(run_id=141, corpus_generation=1, vector_index_generation=1)
    assert ledger._session.get(AgentRun, 141).status == 'failed'


def test_pending_rejects_changed_durable_embedding_charge(tmp_path):
    ledger, authority, grant, dispatch, _, _ = _embedding_case(tmp_path)
    authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    query = _rows(ledger, 141)[0]
    query.charged_cost_usd = Decimal('0.000002')
    ledger._session.get(AgentRun, 141).total_charged_cost_usd = Decimal('0.000002')
    ledger._session.commit()
    with pytest.raises(RagCostLedgerError):
        ledger.commit_embedding_only_pending(run_id=141, corpus_generation=1, vector_index_generation=1)
    assert ledger._session.get(AgentRun, 141).run_record_phase == 'admission'


def test_other_request_ledger_cannot_take_provider_free_projection_ownership(tmp_path):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    peer = _assemble_rag_cost_ledger(
        Session(ledger._session.get_bind()), identity_secret=ledger._secret,
        cost_policy=ledger.cost_policy_authority,
        provider_safety=ledger.provider_safety_authority,
        provider_connection_factory=ledger.provider_connection_factory,
        designated_environment_id='test', designated_host_id='pytest-host',
        runtime_health=ledger.runtime_health_authority,
    )
    with pytest.raises(RagCostLedgerError):
        peer.commit_provider_free_pending(run_id=143, corpus_generation=5, vector_index_generation=None)
    assert _rows(ledger)[1].dispatch_state == 'not_attempted'


def test_pending_commit_ack_failure_returns_no_dto_and_cannot_repeat(tmp_path):
    ledger = _ledger(tmp_path)
    _deferred(ledger)

    def fail_ack():
        raise RuntimeError('pending ACK unavailable')

    ledger._after_commit = fail_ack
    with pytest.raises(RagCostPersistenceError, match='commit acknowledgement unavailable'):
        ledger.commit_provider_free_pending(run_id=143, corpus_generation=5, vector_index_generation=None)
    ledger._after_commit = None
    assert ledger._session.get(AgentRun, 143).run_record_phase == 'cost_finalized_pending_projection'
    with pytest.raises(RagCostLedgerError):
        ledger.commit_provider_free_pending(run_id=143, corpus_generation=5, vector_index_generation=None)


def test_keyword_search_admits_zero_reserves_without_preparing_paid_budgets(tmp_path):
    ledger = _ledger(tmp_path)
    policy = ledger.cost_policy_authority
    assert hasattr(policy, 'reserve_unused_component'), 'unused component reservation is missing'
    admission = ledger.create_admission(
        agent_run_id=144, surface='search', mode='enforce', cutover_stage='search',
        configured_backend='keyword', query_context_version='direct-query:v1',
        current_text_hmac='1' * 64, retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64, admission_cache_identity_hmac=None,
        source_window='rag-v2:admission:enforce:search:keyword',
        components=tuple((_snapshot(component, policy), policy.reserve_unused_component(component))
                         for component in ('query_embedding', 'answer_generation')),
    )
    assert admission.total_reserved_cost_usd == Decimal('0.000000')
    pending = ledger.commit_provider_free_pending(run_id=144, corpus_generation=1, vector_index_generation=None)
    assert pending.parent_agent_run_id == 144
    assert all(row.reserved_cost_usd == 0 and row.dispatch_count == 0 for row in _rows(ledger, 144))


def test_blocked_provider_family_does_not_block_zero_provider_pending(tmp_path):
    ledger = _ledger(tmp_path)
    _deferred(ledger)
    with ledger.provider_connection_factory() as connection:
        ledger.provider_safety_authority.block_remediation(
            connection, 'answer_generation', agent_run_id=143,
            category='provider_safety_unavailable', input_tokens=0, output_tokens=0,
            cost_usd=Decimal('0.000000'),
        )
    pending = ledger.commit_provider_free_pending(run_id=143, corpus_generation=5, vector_index_generation=None)
    assert pending.parent_agent_run_id == 143
    assert ledger.total_charged_cost(143) == Decimal('0.000000')


def test_paid_pending_rechecks_blocked_family_without_losing_paid_cost(tmp_path):
    ledger, authority, grant, dispatch, client, _ = _embedding_case(tmp_path)
    delivered = authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    with ledger.provider_connection_factory() as connection:
        ledger.provider_safety_authority.block_remediation(
            connection, 'query_embedding', agent_run_id=141,
            category='provider_safety_unavailable', input_tokens=0, output_tokens=0,
            cost_usd=Decimal('0.000000'),
        )
    with pytest.raises(RagProviderSafetyError):
        ledger.commit_embedding_only_pending(run_id=141, corpus_generation=1, vector_index_generation=1)
    assert ledger.total_charged_cost(141) == delivered.component_final.charged_cost_usd
    assert ledger._session.get(AgentRun, 141).run_record_phase == 'admission'
    assert len(client.seen) == 1
