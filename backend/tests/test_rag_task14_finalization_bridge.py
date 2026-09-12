from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.provider_send_fence import _assemble_rag_evidence_barrier
from backend.app.agent_runtime.rag_cost_ledger import (
    RagCostLedgerError,
    _assemble_rag_cost_ledger,
)
from backend.app.agent_runtime.rag_finalization import (
    RagFinalizationError,
    RagFinalizationService,
    SqlAlchemyRagFinalizationBoundary,
    _assemble_paid_rag_phase2_authority,
    _exact_paid_embedding_cost,
)
from backend.app.agent_runtime.rag_provider_transport import RagProviderTransportError
from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    prepare_direct_request_text,
)
from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.app.rag.embeddings import validate_query_embedding_vector
from backend.tests.test_rag_v2_finalization import (
    _ClosableConnection,
    _owner_capability,
    _prepared,
    _readiness,
)
from backend.tests.test_rag_v2_provider_transport import (
    _TEST_SETTINGS,
    _admit_transport,
    _authority,
    _Client,
    _prepared_query,
    _transport_ledger,
)


def _search_case(tmp_path):
    client = _Client([])
    client.response = {
        'object': 'list', 'model': 'text-embedding-3-small',
        'data': [{'object': 'embedding', 'index': 0,
                  'embedding': [0.25, *([0.0] * 1535)]}],
        'usage': {'prompt_tokens': 1, 'total_tokens': 1},
    }
    ledger = _transport_ledger(tmp_path, client)
    budget = _admit_transport(ledger, 151, surface='search', scope_hmac=security_scope_fingerprint(
        _prepared().security_scope, settings=_TEST_SETTINGS,
    ))
    prepared = _prepared_query(budget)
    grant = ledger.claim_component(run_id=151, component='query_embedding', prepared=budget)
    authority = _authority(ledger)
    dispatch = authority.prepare(grant=grant, prepared=prepared)
    delivery = authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    return ledger, delivery, client


def test_real_search_embedding_preserves_distinct_identity_domains(tmp_path):
    ledger, delivery, client = _search_case(tmp_path)
    parent = ledger._session.get(AgentRun, 151)
    row = ledger._session.scalar(select(AgentRunCostComponent).where(
        AgentRunCostComponent.agent_run_id == 151,
        AgentRunCostComponent.component == 'query_embedding',
    ))
    prepared = delivery.output.prepared
    assert parent.metadata_['retrieval_query_hmac'] != prepared.retrieval_query_hmac
    assert row.dispatch_fence_hmac != prepared.attempt_fence_hmac
    assert row.authorized_policy_snapshot_hmac != prepared.provider_policy_snapshot_hmac
    assert _exact_paid_embedding_cost(row, delivery.output, secret=ledger._secret)
    assert parent.run_record_phase == 'cost_finalized_pending_projection'
    assert ledger.total_charged_cost(151) == Decimal('0.000001')
    assert len(client.seen) == 1


def test_already_pending_search_returns_carrier_without_new_transition(tmp_path, monkeypatch):
    ledger, delivery, _ = _search_case(tmp_path)
    parent = ledger._session.get(AgentRun, 151)
    owner = parent.projection_owner_fence_hmac

    def no_commit():
        raise AssertionError('reading committed pending authority must not commit')

    monkeypatch.setattr(ledger._session, 'commit', no_commit)
    pending = ledger.load_pending_projection(
        run_id=151, corpus_generation=1, vector_index_generation=1,
    )
    assert pending.parent_agent_run_id == 151
    assert pending.projection_owner_fence_hmac == owner
    assert pending.terminal_cost_snapshot_hmac == parent.metadata_['runtime_cost_snapshot_hmac']
    assert delivery.component_final.parent_run_record_phase == 'cost_finalized_pending_projection'


def test_other_dispatch_same_query_result_cannot_finalize_billed_dispatch(tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    ledger, delivery, _ = _search_case(first)
    _, other, _ = _search_case(second)
    row = ledger._session.scalar(select(AgentRunCostComponent).where(
        AgentRunCostComponent.agent_run_id == 151,
        AgentRunCostComponent.component == 'query_embedding',
    ))
    assert delivery.output.prepared == other.output.prepared
    assert delivery.output.vector == other.output.vector
    assert not _exact_paid_embedding_cost(row, other.output, secret=ledger._secret)


@pytest.mark.parametrize('field,value', (
    ('dispatch_fence_hmac', 'f' * 64),
    ('charged_cost_usd', Decimal('0.000002')),
))
def test_pending_carrier_refuses_changed_committed_cost(tmp_path, field, value):
    from backend.app.agent_runtime.rag_cost_ledger import RagCostLedgerError

    ledger, _, _ = _search_case(tmp_path)
    row = ledger._session.scalar(select(AgentRunCostComponent).where(
        AgentRunCostComponent.agent_run_id == 151,
        AgentRunCostComponent.component == 'query_embedding',
    ))
    setattr(row, field, value)
    ledger._session.commit()
    with pytest.raises(RagCostLedgerError, match='changed'):
        ledger.load_pending_projection(run_id=151, corpus_generation=1, vector_index_generation=1)


@pytest.mark.parametrize('tamper', ('receipt', 'vector', 'cost_policy'))
def test_delivery_receipt_rejects_changed_result_or_budget(tmp_path, tamper):
    ledger, delivery, _ = _search_case(tmp_path)
    row = ledger._session.scalar(select(AgentRunCostComponent).where(
        AgentRunCostComponent.agent_run_id == 151,
        AgentRunCostComponent.component == 'query_embedding',
    ))
    result = delivery.output
    if tamper == 'receipt':
        result = replace(result, committed_dispatch_hmac=None)
    elif tamper == 'vector':
        result = replace(result, vector=validate_query_embedding_vector([0.5, *([0.0] * 1535)]))
    else:
        result = replace(result, prepared=replace(result.prepared, budget=replace(
            result.prepared.budget, cost_policy_snapshot_hmac='f' * 64,
        )))
    assert not _exact_paid_embedding_cost(row, result, secret=ledger._secret)


class _SQLiteFinalizationPort(SqlAlchemyRagFinalizationBoundary):
    """Fake PG synchronization/retrieval only; real row validation/projection/write."""

    def __init__(self, ledger, prepared):
        self._db = ledger._session
        self._secret = ledger._secret
        self._settings = _TEST_SETTINGS
        self._prefix_generations = (1, 1)
        self._pending = None
        self._pending_parent = None
        self._pending_children = ()
        self._prepared = prepared
        self._projection_coordinator = SimpleNamespace(
            lock_canonical_tail=lambda _ids: object(), validate_tail_context=lambda _tail: None,
        )
        binding = ledger._terminal_bindings[(151, 'query_embedding')]
        self._phase2_authority = _assemble_paid_rag_phase2_authority(
            provider_safety=ledger._provider_safety,
            safety_connection_factory=ledger._provider_connection_factory,
            safety_requirements=((binding.policy_snapshot, binding),),
            owner_connection_factory=_ClosableConnection,
            owner_capability_factory=_owner_capability,
            load_current_owner_fence=lambda _run_id: None,
            evidence_barrier=_assemble_rag_evidence_barrier(load_current_identity=lambda: 'a' * 64),
            load_current_readiness=lambda: _readiness(snapshot='4' * 64),
        )
        self._phase2_authority._current_readiness = _readiness(snapshot='4' * 64)

    def acquire_request_database_authority(self):
        return nullcontext()

    def close_request_database_authority(self):
        pass

    def acquire_phase2(self, pending, prepared, *, branch):
        assert branch == 'paid_embedding_only'
        self._pending = pending
        return nullcontext()

    @contextmanager
    def begin(self):
        self._db.rollback()  # End the ledger's read-only carrier lookup.
        with self._db.begin():
            yield

    def acquire_projection_prefix(self):
        pass

    def retrieve_fresh(self, request):
        assert request.query_embedding_result is self._prepared.query_embedding_result
        return self._prepared.retrieval_result

    def commit(self):
        self._db.commit()


def _search_finalization(delivery):
    base = _prepared()
    return replace(
        base, product_kind='search', tentative_outcome='search_projected',
        prepared_text=prepare_direct_request_text(
            '민감한 근거', key=_TEST_SETTINGS.agent_runtime_fingerprint_secret.encode(),
        ),
        query_embedding_result=delivery.output,
        retrieval_result=replace(base.retrieval_result,
            configured_backend='pgvector', effective_backend='pgvector', hidden_match_count=0,
            query_embedding_receipt=delivery.output.receipt,
        ),
        canned_message_identity=None,
    )


def test_real_ledger_transport_and_finalizer_commit_search_product(tmp_path):
    ledger, delivery, client = _search_case(tmp_path)
    pending = ledger.load_pending_projection(run_id=151, corpus_generation=1, vector_index_generation=1)
    prepared = _search_finalization(delivery)
    service = RagFinalizationService(
        transaction_boundary=_SQLiteFinalizationPort(ledger, prepared), settings=_TEST_SETTINGS,
    )
    product = service.finalize_paid_embedding_only_safe(pending, prepared)
    parent = ledger._session.get(AgentRun, 151)
    assert product.outcome == 'search_projected'
    assert product.answer_text is None
    assert parent.status == 'complete'
    assert parent.run_record_phase == 'final'
    assert parent.projection_owner_fence_hmac is None
    assert parent.total_charged_cost_usd == Decimal('0.000001')
    assert 'committed_dispatch_hmac' not in str(parent.metadata_)
    assert '민감한 근거' not in str(parent.metadata_)
    assert len(client.seen) == 1
    with pytest.raises(RagFinalizationError):
        service.finalize_paid_embedding_only_safe(pending, prepared)


def test_swapped_same_query_delivery_cannot_commit_other_search_product(tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    ledger, _, _ = _search_case(first)
    _, other, _ = _search_case(second)
    pending = ledger.load_pending_projection(run_id=151, corpus_generation=1, vector_index_generation=1)
    prepared = _search_finalization(other)
    service = RagFinalizationService(
        transaction_boundary=_SQLiteFinalizationPort(ledger, prepared), settings=_TEST_SETTINGS,
    )
    with pytest.raises(RagFinalizationError, match='paid embedding cost'):
        service.finalize_paid_embedding_only_safe(pending, prepared)
    parent = ledger._session.get(AgentRun, 151)
    assert parent.run_record_phase == 'cost_finalized_pending_projection'
    assert parent.total_charged_cost_usd == Decimal('0.000001')


def test_pending_carrier_is_not_available_to_another_request_ledger(tmp_path):
    ledger, _, _ = _search_case(tmp_path)
    peer = _assemble_rag_cost_ledger(
        Session(ledger._session.get_bind()), identity_secret=ledger._secret,
        cost_policy=ledger.cost_policy_authority,
        provider_safety=ledger.provider_safety_authority,
        provider_connection_factory=ledger.provider_connection_factory,
        designated_environment_id='test', designated_host_id='pytest-host',
        runtime_health=ledger.runtime_health_authority,
    )
    with pytest.raises(RagCostLedgerError, match='request-owned'):
        peer.load_pending_projection(run_id=151, corpus_generation=1, vector_index_generation=1)


@pytest.mark.parametrize('tamper', ('embedding_domain', 'context', 'query'))
def test_transport_refuses_wrong_admitted_query_identity_without_dispatch(tmp_path, tamper):
    client = _Client([])
    ledger = _transport_ledger(tmp_path, client)
    budget = _admit_transport(ledger, 151, surface='search')
    prepared = _prepared_query(budget)
    parent = ledger._session.get(AgentRun, 151)
    metadata = dict(parent.metadata_)
    if tamper == 'embedding_domain':
        metadata['retrieval_query_hmac'] = prepared.retrieval_query_hmac
    elif tamper == 'context':
        metadata['query_context_version'] = 'assistant-context:v1'
    else:
        metadata['retrieval_query_hmac'] = prepare_direct_request_text(
            '다른 질의', key=_TEST_SETTINGS.agent_runtime_fingerprint_secret.encode(),
        ).retrieval_query_hmac
    parent.metadata_ = metadata
    ledger._session.commit()
    grant = ledger.claim_component(run_id=151, component='query_embedding', prepared=budget)
    authority = _authority(ledger)
    with pytest.raises(RagProviderTransportError, match='misaligned'):
        authority.prepare(grant=grant, prepared=prepared)
    assert client.seen == []
