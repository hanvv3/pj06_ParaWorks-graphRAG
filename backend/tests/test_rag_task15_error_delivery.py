from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from backend.app.models import AgentRun
from backend.tests.test_rag_api_delivery import _graph_http
from backend.tests.test_rag_v2_graph import (
    _answer_context,
    _paid_context,
    _with_paid_embedding_answer,
)


def _post(http, surface):
    return http.post(
        f'/{surface}', json={('question' if surface == 'ask' else 'query'): 'observation'}
    )


@pytest.mark.parametrize('surface', ['ask', 'search'])
@pytest.mark.parametrize('failure', ['not_ready', 'invalid', 'value_error', 'io_error'])
def test_actual_readiness_error_delivery(tmp_path, surface, failure):
    context, provider = _paid_context(tmp_path)
    readiness = replace(
        context.services.provider_transport._load_current_readiness(), ready=False
    )

    def inspect(**kwargs):
        if failure == 'value_error':
            raise ValueError('unrelated value failure')
        if failure == 'io_error':
            raise OSError('unrelated storage failure')
        return object() if failure == 'invalid' else readiness

    context = replace(
        context, surface=surface,
        services=replace(context.services, index_readiness=SimpleNamespace(inspect=inspect)),
    )
    with _graph_http(context) as http:
        response = _post(http, surface)
    assert response.status_code == (503 if failure == 'not_ready' else 500)
    assert response.json() == {'detail': {'code': (
        'retriever_unavailable' if failure == 'not_ready' else 'unexpected_internal_error'
    )}}
    assert provider.seen == []
    assert context.services.db.query(AgentRun).count() == 0


@pytest.mark.parametrize('surface', ['ask', 'search'])
@pytest.mark.parametrize('failure', ['safety', 'commit', 'unknown_ack', 'generic_transport'])
def test_real_safety_barrier_refusal_delivers_acknowledged_terminal(
    tmp_path, monkeypatch, surface, failure
):
    from backend.app.agent_runtime.rag_provider_transport import (
        RagProviderDispatchAuthority,
    )

    context, provider = (_answer_context if surface == 'ask' else _paid_context)(tmp_path)
    original = RagProviderDispatchAuthority.prepare
    attempts = []

    def fail():
        attempts.append(failure)
        raise OSError('private storage failure')

    def prepare(authority, **kwargs):
        prepared = original(authority, **kwargs)
        if failure == 'generic_transport':
            monkeypatch.setattr(type(authority._barrier), '_run', lambda self, **kwargs: fail())
            return prepared
        grant = kwargs['grant']
        with authority._connection_factory() as connection:
            authority._safety.block_remediation(
                connection, grant.component, category='provider_safety_unavailable',
                agent_run_id=grant.agent_run_id, input_tokens=0, output_tokens=0,
                cost_usd=Decimal('0'),
            )
        if failure == 'commit':
            monkeypatch.setattr(authority._store._session, 'commit', fail)
        elif failure == 'unknown_ack':
            monkeypatch.setattr(authority._store, '_after_commit', fail)
        return prepared

    monkeypatch.setattr(RagProviderDispatchAuthority, 'prepare', prepare)
    with _graph_http(context) as http:
        response = _post(http, surface)
    assert provider.seen == []
    parent = context.services.db.query(AgentRun).one()
    if failure != 'commit':
        assert parent.status == 'failed'
        assert parent.metadata_['outcome'] == 'provider_safety_unavailable'
    else:
        assert parent.status == 'running'
    assert len(attempts) == int(failure != 'safety')
    expected = {
        'safety': (503, 'provider_safety_unavailable'),
        'commit': (500, 'persistence_failed'),
        'unknown_ack': (500, 'persistence_failed'),
        'generic_transport': (500, 'unexpected_internal_error'),
    }[failure]
    assert response.status_code == expected[0]
    assert response.json() == {'detail': {'code': expected[1]}}


@pytest.mark.parametrize('unknown_ack', [False, True])
@pytest.mark.parametrize('paid_embedding', [False, True])
def test_actual_answer_binding_commit_failure_is_persistence_failed(
    tmp_path, monkeypatch, unknown_ack, paid_embedding
):
    context, provider = _answer_context(tmp_path)
    if paid_embedding:
        context = _with_paid_embedding_answer(context, provider)
    ledger = context.services.cost_ledger
    original = context.services.answer_model.prepare_input
    attempts = []

    def fail():
        attempts.append('commit_ack' if unknown_ack else 'commit')
        raise RuntimeError('private database failure detail')

    def prepare(**kwargs):
        result = original(**kwargs)
        if unknown_ack:
            monkeypatch.setattr(ledger, '_after_commit', fail)
        else:
            monkeypatch.setattr(ledger._session, 'commit', fail)
        return result

    monkeypatch.setattr(context.services.answer_model, 'prepare_input', prepare)
    with _graph_http(context) as http:
        response = _post(http, 'ask')
    assert response.status_code == 500
    assert response.json() == {'detail': {'code': 'persistence_failed'}}
    assert len(provider.seen) == int(paid_embedding)
    assert len(attempts) == 1
    assert 161 not in ledger._answer_reservations
    assert (161, 'answer_generation') not in ledger._admission_budgets
    parent = context.services.db.query(AgentRun).one()
    assert parent.status == 'running'
    assert parent.run_record_phase == 'admission'
    if paid_embedding:
        from backend.app.models.rag_runtime import AgentRunCostComponent

        component = context.services.db.query(AgentRunCostComponent).filter_by(
            component='query_embedding'
        ).one()
        assert component.terminal_outcome == 'component_succeeded'
        assert component.charged_cost_usd == Decimal('0.000001')


@pytest.mark.parametrize('surface', ['ask', 'search'])
@pytest.mark.parametrize('unknown_ack', [False, True])
def test_actual_paid_component_commit_failure_never_releases_or_resends(
    tmp_path, monkeypatch, surface, unknown_ack
):
    from backend.app.agent_runtime.rag_provider_transport import (
        RagProviderDispatchAuthority,
        RagProviderTransportError,
    )
    from backend.app.models.rag_runtime import AgentRunCostComponent

    context, provider = (_answer_context if surface == 'ask' else _paid_context)(tmp_path)
    ledger = context.services.cost_ledger
    original_prepare = RagProviderDispatchAuthority.prepare
    captured = []
    attempts = []

    def prepare(authority, **kwargs):
        prepared = original_prepare(authority, **kwargs)
        captured.append((authority, kwargs['grant'], prepared))
        return prepared

    def fail():
        attempts.append('commit')
        raise OSError('private database detail')

    original_send = provider.send

    def send(*args, **kwargs):
        response = original_send(*args, **kwargs)
        if unknown_ack:
            monkeypatch.setattr(ledger, '_after_commit', fail)
        else:
            monkeypatch.setattr(ledger._session, 'commit', fail)
        return response

    monkeypatch.setattr(RagProviderDispatchAuthority, 'prepare', prepare)
    monkeypatch.setattr(provider, 'send', send)
    with _graph_http(context) as http:
        response = _post(http, surface)
    assert response.status_code == 500
    assert response.json() == {'detail': {'code': 'persistence_failed'}}
    assert attempts == ['commit']
    authority, grant, prepared = captured[0]
    assert grant.consumed is True
    with pytest.raises(RagProviderTransportError):
        authority.dispatch_and_finalize(grant=grant, prepared=prepared)
    assert len(provider.seen) == 1
    assert attempts == ['commit']
    component = context.services.db.query(AgentRunCostComponent).filter_by(
        component=grant.component,
    ).one()
    assert component.charged_cost_usd > 0
    assert component.charge_basis == ('actual' if unknown_ack else 'reserved')
