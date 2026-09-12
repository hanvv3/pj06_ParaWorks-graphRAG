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


@pytest.mark.parametrize('case', ['query_ask', 'query_search', 'answer', 'paid_answer'])
@pytest.mark.parametrize('failure', ['blocked', 'commit', 'unknown_ack', 'io', 'sql', 'after_barrier'])
def test_actual_preclaim_safety_refusal_closes_only_authenticated_admission(
    tmp_path, monkeypatch, case, failure
):
    from sqlalchemy.exc import SQLAlchemyError

    from backend.app.models.rag_runtime import AgentRunCostComponent

    surface = 'search' if case == 'query_search' else 'ask'
    if case.startswith('query'):
        context, provider = _paid_context(tmp_path)
        context = replace(context, surface=surface)
        target = 'query_embedding'
    else:
        context, provider = _answer_context(tmp_path)
        if case == 'paid_answer':
            context = _with_paid_embedding_answer(context, provider)
        target = 'answer_generation'
    ledger = context.services.cost_ledger
    transport = context.services.provider_transport
    claim = ledger.claim_component
    failures = []
    finalization_transactions = []
    finalize = ledger.finalize_inter_component_failure

    def finalize_after_read_release(**kwargs):
        finalization_transactions.append(ledger._session.in_transaction())
        return finalize(**kwargs)

    monkeypatch.setattr(ledger, 'finalize_inter_component_failure', finalize_after_read_release)

    def fail(*args, **kwargs):
        failures.append(failure)
        if failure == 'sql':
            raise SQLAlchemyError('private database error')
        raise OSError('private storage error')

    def mutate_after_commit():
        from contextlib import contextmanager

        @contextmanager
        def already_locked():
            # Fault injection on the owning thread: the outer barrier still
            # owns the real sidecar. Avoid recursively acquiring its OS lock.
            yield

        failures.append(failure)
        with monkeypatch.context() as patch:
            patch.setattr(transport._safety._authority, 'locked', already_locked)
            with transport._connection_factory() as connection:
                transport._safety.block_remediation(
                    connection,
                    'answer_generation' if target == 'query_embedding' else 'query_embedding',
                    category='provider_safety_unavailable', agent_run_id=161,
                    input_tokens=0, output_tokens=0, cost_usd=Decimal('0'),
                )

    def block_before_claim(**kwargs):
        if kwargs['component'] == target:
            if failure in {'io', 'sql'}:
                monkeypatch.setattr(transport._safety, '_binding', fail)
            else:
                with transport._connection_factory() as connection:
                    transport._safety.block_remediation(
                        connection, target, category='provider_safety_unavailable',
                        agent_run_id=kwargs['run_id'], input_tokens=0, output_tokens=0,
                        cost_usd=Decimal('0'),
                    )
                if failure == 'commit':
                    monkeypatch.setattr(ledger._session, 'commit', fail)
                elif failure == 'unknown_ack':
                    monkeypatch.setattr(ledger, '_after_commit', fail)
                elif failure == 'after_barrier':
                    monkeypatch.setattr(ledger, '_after_commit', mutate_after_commit)
        return claim(**kwargs)

    monkeypatch.setattr(ledger, 'claim_component', block_before_claim)
    with _graph_http(context) as http:
        response = _post(http, surface)
    expected = {
        'blocked': (503, 'provider_safety_unavailable'),
        'commit': (500, 'persistence_failed'),
        'unknown_ack': (500, 'persistence_failed'),
        'io': (500, 'unexpected_internal_error'),
        'sql': (500, 'unexpected_internal_error'),
        'after_barrier': (500, 'persistence_failed'),
    }[failure]
    assert response.status_code == expected[0]
    assert response.json() == {'detail': {'code': expected[1]}}
    assert len(provider.seen) == int(case == 'paid_answer')
    assert len(failures) == int(failure != 'blocked')
    assert finalization_transactions == ([] if failure in {'io', 'sql'} else [False])
    parent = context.services.db.query(AgentRun).one()
    if failure in {'blocked', 'unknown_ack', 'after_barrier'}:
        assert parent.status == 'failed'
        assert parent.run_record_phase == 'final'
        assert parent.metadata_['outcome'] == 'provider_safety_unavailable'
        assert parent.completed_at is not None
    else:
        assert parent.status == 'running'
        assert parent.run_record_phase == 'admission'
    rows = context.services.db.query(AgentRunCostComponent).all()
    assert len(rows) == 2
    for row in rows:
        assert (row.agent_run_id, row.component) not in ledger._active_grants
        if case == 'paid_answer' and row.component == 'query_embedding':
            assert row.terminal_outcome == 'component_succeeded'
            assert row.charged_cost_usd == Decimal('0.000001')
            assert parent.total_charged_cost_usd == row.charged_cost_usd
        else:
            assert row.attempted is False
            assert row.dispatch_count == 0
            assert row.charged_cost_usd == 0
            assert row.dispatch_state == (
                'terminal' if failure in {'blocked', 'unknown_ack', 'after_barrier'} else 'not_attempted'
            )
    if failure == 'after_barrier':
        # Both snapshots are authentic and whole-set-consistent. Only the
        # changed envelope across the barrier prevents an acknowledged503.
        with transport._connection_factory() as connection:
            transport._safety._match_db_whole_set(
                connection, transport._safety._read_unlocked(),
            )


@pytest.mark.parametrize('damage', [
    None, 'missing_receipt', 'forged_receipt', 'missing_binding', 'forged_binding',
    'paid_row', 'authority_missing', 'authority_changed',
])
def test_failure_only_accounting_preserves_paid_row_but_never_relaxes_success(
    tmp_path, damage
):
    from backend.app.agent_runtime.rag_cost_ledger import RagCostPersistenceError
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
    from backend.app.models.rag_runtime import (
        AgentRunCostComponent,
        RagProviderSafetyAuthority,
    )
    from backend.tests.test_rag_task14_transport_delivery import _embedding_case

    ledger, authority, grant, dispatch, provider, _ = _embedding_case(tmp_path)
    authority.dispatch_and_finalize(grant=grant, prepared=dispatch)
    row = ledger._session.query(AgentRunCostComponent).filter_by(
        agent_run_id=141, component='query_embedding',
    ).one()
    before = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    with ledger.provider_connection_factory() as connection:
        ledger.provider_safety_authority.block_remediation(
            connection, 'answer_generation', agent_run_id=141,
            category='provider_safety_unavailable', input_tokens=0, output_tokens=0,
            cost_usd=Decimal('0'),
        )
    # The identical global/envelope drift must still refuse a success projection.
    with pytest.raises(RagProviderSafetyError):
        ledger.commit_embedding_only_pending(
            run_id=141, corpus_generation=1, vector_index_generation=1,
        )
    if damage == 'missing_receipt':
        ledger._terminal_cost_rows.pop((141, 'query_embedding'))
    elif damage == 'forged_receipt':
        ledger._terminal_cost_rows[(141, 'query_embedding')] = (('component', 'forged'),)
    elif damage == 'missing_binding':
        ledger._terminal_bindings.pop((141, 'query_embedding'))
    elif damage == 'forged_binding':
        ledger._terminal_bindings[(141, 'query_embedding')] = object()
    elif damage == 'paid_row':
        row.actual_input_tokens += 1
        ledger._session.commit()
    elif damage in {'authority_missing', 'authority_changed'}:
        authority_row = ledger._session.get(RagProviderSafetyAuthority, 1)
        if damage == 'authority_missing':
            ledger._session.delete(authority_row)
        else:
            authority_row.envelope_digest = 'f' * 64
        ledger._session.commit()
    if damage is not None:
        with pytest.raises(RagCostPersistenceError):
            ledger.finalize_inter_component_failure(
                run_id=141, outcome='provider_safety_unavailable',
            )
        assert ledger._session.get(AgentRun, 141).status == 'running'
    else:
        terminal = ledger.finalize_inter_component_failure(
            run_id=141, outcome='provider_safety_unavailable',
        )
        assert terminal.status == 'failed'
        assert terminal.run_record_phase == 'final'
        ledger._session.refresh(row)
        assert {column.name: getattr(row, column.name) for column in row.__table__.columns} == before
    assert len(provider.seen) == 1


@pytest.mark.parametrize('failure', ['rollback', 'dirty'])
def test_preclaim_read_transaction_release_refuses_uncertainty_without_discard(
    tmp_path, monkeypatch, failure
):
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    context, provider = _answer_context(tmp_path)
    context = _with_paid_embedding_answer(context, provider)
    ledger = context.services.cost_ledger
    transport = context.services.provider_transport
    claim = ledger.claim_component
    ready = transport._safety.require_ready
    rollbacks = []
    closures = []
    original_rollback = ledger._session.rollback

    def rollback():
        rollbacks.append('rollback')
        raise OSError('private rollback error')

    def dirty_ready(*args, **kwargs):
        try:
            return ready(*args, **kwargs)
        except RagProviderSafetyError:
            parent = ledger._session.get(AgentRun, 161)
            parent.metadata_ = {**parent.metadata_, 'pending_test_write': True}
            raise

    def block(**kwargs):
        if kwargs['component'] == 'answer_generation':
            with transport._connection_factory() as connection:
                transport._safety.block_remediation(
                    connection, 'answer_generation', category='provider_safety_unavailable',
                    agent_run_id=161, input_tokens=0, output_tokens=0, cost_usd=Decimal('0'),
                )
            if failure == 'rollback':
                monkeypatch.setattr(ledger._session, 'rollback', rollback)
            else:
                monkeypatch.setattr(transport._safety, 'require_ready', dirty_ready)
        return claim(**kwargs)

    monkeypatch.setattr(ledger, 'claim_component', block)
    monkeypatch.setattr(ledger, 'finalize_inter_component_failure', lambda **kwargs: closures.append(kwargs))
    with _graph_http(context) as http:
        response = _post(http, 'ask')
    assert response.status_code == 500
    assert response.json() == {'detail': {'code': 'persistence_failed'}}
    assert len(provider.seen) == 1
    assert closures == []
    assert rollbacks == (['rollback'] if failure == 'rollback' else [])
    assert ledger._session.in_transaction()
    if failure == 'dirty':
        assert ledger._session.dirty
        with ledger._session.no_autoflush:
            assert ledger._session.get(AgentRun, 161).metadata_['pending_test_write'] is True
    original_rollback()


@pytest.mark.parametrize('surface', ['ask', 'search'])
@pytest.mark.parametrize('fault', ['read', 'stat', 'close', 'unicode', 'json'])
def test_low_level_authority_io_is_not_an_authenticated_preclaim_refusal(
    tmp_path, monkeypatch, surface, fault
):
    import os

    context, provider = (_answer_context if surface == 'ask' else _paid_context)(tmp_path)
    ledger = context.services.cost_ledger
    authority = context.services.provider_transport._safety._authority
    identity = authority.path.stat()
    original_read, original_stat, original_close = os.read, os.fstat, os.close
    claim = ledger.claim_component
    faults = []

    def selected(descriptor):
        current = original_stat(descriptor)
        return (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)

    def read(descriptor, size):
        if not faults and selected(descriptor):
            faults.append(fault)
            if fault == 'unicode':
                return b'\xff'
            if fault == 'json':
                return b'{'
            raise OSError('private authority read failure')
        return original_read(descriptor, size)

    def stat(descriptor):
        if not faults and selected(descriptor):
            faults.append(fault)
            raise OSError('private authority stat failure')
        return original_stat(descriptor)

    def close(descriptor):
        target = not faults and selected(descriptor)
        original_close(descriptor)
        if target:
            faults.append(fault)
            raise OSError('private authority close acknowledgement failure')

    def claim_with_real_io_fault(**kwargs):
        name, function = {
            'read': ('read', read), 'stat': ('fstat', stat), 'close': ('close', close),
            'unicode': ('read', read), 'json': ('read', read),
        }[fault]
        monkeypatch.setattr(os, name, function)
        return claim(**kwargs)

    monkeypatch.setattr(ledger, 'claim_component', claim_with_real_io_fault)
    with _graph_http(context) as http:
        response = _post(http, surface)
    malformed = fault in {'unicode', 'json'}
    assert faults == [fault]
    assert response.status_code == (503 if malformed else 500)
    assert response.json() == {'detail': {'code': (
        'provider_safety_unavailable' if malformed else 'unexpected_internal_error'
    )}}
    assert provider.seen == []
    parent = ledger._session.query(AgentRun).one()
    assert parent.status == ('failed' if malformed else 'running')
    assert parent.run_record_phase == ('final' if malformed else 'admission')
    assert parent.metadata_.get('outcome') == ('provider_safety_unavailable' if malformed else None)
