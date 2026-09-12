from __future__ import annotations

import pickle

import pytest
from langchain_core.messages import AIMessage

from backend.app.agent_runtime.rag_provider_transport import RagProviderTransportError
from backend.app.models.agent_runs import AgentRun
from backend.app.rag.retrieval import QueryEmbeddingCallResult
from backend.tests.test_rag_v2_provider_transport import (
    _admit_transport,
    _answer_transport_case,
    _authority,
    _Client,
    _prepared_query,
    _transport_ledger,
)


def _embedding_case(tmp_path, *, malformed=False):
    client = _Client([])
    ledger = _transport_ledger(tmp_path, client)
    budget = _admit_transport(ledger, 141)
    client.response = {
        'object': 'list', 'model': 'text-embedding-3-small',
        'data': [{'object': 'embedding', 'index': 0,
                  'embedding': [0.25, *([0.0] * 1535)]}],
        'usage': {'prompt_tokens': 1, 'total_tokens': 1},
    }
    if malformed:
        client.response['data'][0]['embedding'] = [0.0] * 1536
    grant = ledger.claim_component(
        run_id=141, component='query_embedding', prepared=budget,
    )
    authority = _authority(ledger)
    prepared = _prepared_query(budget)
    dispatch = authority.prepare(grant=grant, prepared=prepared)
    return ledger, authority, grant, dispatch, client, prepared


def test_successful_embedding_delivery_uses_one_dispatch_and_committed_actual_cost(tmp_path):
    ledger, authority, grant, dispatch, client, prepared = _embedding_case(tmp_path)
    assert hasattr(authority, 'dispatch_and_finalize'), 'validated output delivery is missing'
    result = authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    assert type(result.output) is QueryEmbeddingCallResult
    assert result.output.prepared is prepared
    assert result.output.vector.coordinates == (0.25, *([0.0] * 1535))
    assert result.output.validated_input_tokens == 1
    assert result.output.actual_cost_usd == result.component_final.charged_cost_usd
    assert result.component_final.charge_basis == 'actual'
    assert ledger.total_charged_cost(141) == result.component_final.charged_cost_usd
    assert len(client.seen) == 1
    assert '0.25' not in repr(result)
    with pytest.raises(TypeError):
        pickle.dumps(result)
    with pytest.raises(RagProviderTransportError):
        authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    assert len(client.seen) == 1


def test_failed_embedding_delivery_charges_but_never_releases_vector(tmp_path):
    ledger, authority, grant, dispatch, client, _ = _embedding_case(tmp_path, malformed=True)
    assert hasattr(authority, 'dispatch_and_finalize'), 'validated output delivery is missing'
    result = authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    assert result.output is None
    assert result.component_final.terminal_outcome == 'provider_embedding_payload_invalid'
    assert result.component_final.charge_basis == 'actual'
    assert ledger._session.get(AgentRun, 141).status == 'failed'
    assert len(client.seen) == 1


@pytest.mark.parametrize('selected', ('E1', 'E9'))
def test_answer_delivery_returns_only_validated_blocks_after_cost_commit(tmp_path, selected):
    ledger, authority, grant, dispatch, client, _ = _answer_transport_case(tmp_path, 142)
    client.response = {
        'raw': AIMessage(
            content='do not retain provider raw payload',
            usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
            response_metadata={'model': 'gpt-5.4-mini-2026-03-17',
                               'object': 'response', 'service_tier': 'default'},
        ),
        'parsed': {'answer_blocks': [{'text': '답변', 'evidence_slot_ids': [selected],
                                      'support_mode': 'source_observation'}],
                   'insufficient_evidence_reason': None},
        'parsing_error': None,
    }
    assert hasattr(authority, 'dispatch_and_finalize'), 'validated output delivery is missing'
    result = authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    if selected == 'E1':
        assert result.output.assembled_answer == '답변'
        assert result.output.selected_slot_ids == ('E1',)
        assert result.component_final.parent_run_record_phase == 'cost_finalized_pending_projection'
    else:
        assert result.output is None
        assert result.component_final.terminal_outcome == 'citation_validation_failed'
    parent = ledger._session.get(AgentRun, 142)
    assert '답변' not in str(parent.metadata_)
    assert 'do not retain provider raw payload' not in repr(result)
    assert len(client.seen) == 1


def test_cost_finalization_failure_never_returns_provider_output(tmp_path, monkeypatch):
    ledger, authority, grant, dispatch, client, _ = _embedding_case(tmp_path)
    assert hasattr(authority, 'dispatch_and_finalize'), 'validated output delivery is missing'

    def fail(**_kwargs):
        raise RuntimeError('cost commit unavailable')

    monkeypatch.setattr(ledger, 'finalize_component', fail)
    with pytest.raises(RuntimeError, match='cost commit unavailable'):
        authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    assert len(client.seen) == 1
    with pytest.raises(RagProviderTransportError):
        authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    assert len(client.seen) == 1
