from backend.app.agents.rag_orchestrator_agent import service as rag_service
from backend.app.agents.rag_orchestrator_agent.agent import (
    RagModelResponse,
    RagOrchestratorAgent,
)
from backend.app.connectors.mock import get_mock_connector
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import (
    AgentRun,
    DecisionRecord,
    ReviewItem,
    Source,
)


def _human_approve_synced_sources(db_session) -> None:
    for source in db_session.query(Source).all():
        db_session.add(
            ReviewItem(
                item_type='source_evidence',
                payload={'source_ids': [source.source_id]},
                source_links=[source.source_url],
                source_snippets=[source.title],
                confidence_score=1.0,
                permission_level=source.permission_level,
                status='approved',
                resolution_source='human',
            )
        )
    db_session.commit()


def test_ask_api_answers_with_visible_sources(client, db_session) -> None:
    sync_connector_events(db=db_session, connector=get_mock_connector('gmail'))
    _human_approve_synced_sources(db_session)

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis job state'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['agent_name'] == 'rag_orchestrator_agent'
    assert payload['answer']
    assert payload['source_links']
    assert payload['source_ids']
    assert payload['estimated_cost_usd'] > 0
    assert payload['token_usage']['total_tokens'] > 0
    agent_run = db_session.query(AgentRun).filter_by(agent_name='rag_orchestrator_agent').one()
    assert payload['agent_run_id'] == agent_run.id


def test_ask_api_respects_viewer_permissions(client, db_session) -> None:
    sync_connector_events(db=db_session, connector=get_mock_connector('drive'))
    _human_approve_synced_sources(db_session)

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'confidential pricing'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['source_links'] == []
    assert payload['source_ids'] == []
    assert payload['hidden_match_count'] == 1
    assert payload['permission_notice'] == 'Some sources may be hidden by permissions.'


def test_ask_api_answers_from_approved_knowledge(client, db_session) -> None:
    db_session.add(
        DecisionRecord(
            title='Use Redis for queues',
            decision_summary='Redis should power queue and job progress updates.',
            source_links=['https://knowledge.mock/redis-decision'],
            source_snippets=['Approved Redis decision snippet'],
            confidence_score=0.91,
            permission_level='internal',
            review_status='approved',
        )
    )
    db_session.commit()

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis queues'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['source_links'] == ['https://knowledge.mock/redis-decision']
    assert payload['source_ids'] == ['decision_record:1']
    assert payload['source_snippets'] == ['Approved Redis decision snippet']
    assert payload['hidden_match_count'] == 0


def test_ask_api_revalidates_exact_dependencies_after_provider_returns(
    client,
    db_session,
    monkeypatch,
) -> None:
    decision = DecisionRecord(
        title='Use Redis for queues',
        decision_summary='Redis should power queue and job progress updates.',
        source_links=['https://knowledge.mock/redis-decision'],
        source_snippets=['Approved Redis decision snippet'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    hidden_decision = DecisionRecord(
        title='Restricted Redis migration',
        decision_summary='Redis migration details remain restricted.',
        source_links=['https://knowledge.mock/restricted-redis'],
        source_snippets=['Restricted Redis migration evidence'],
        confidence_score=0.9,
        permission_level='restricted',
        review_status='approved',
    )
    db_session.add_all([decision, hidden_decision])
    db_session.commit()

    class RevokeDuringAnswer:
        def answer(self, question, packet):
            assert db_session.in_transaction() is False
            assert packet.source_ids == [f'decision_record:{decision.id}']
            current = db_session.get(DecisionRecord, decision.id)
            current.review_status = 'revoked'
            db_session.commit()
            return RagModelResponse(
                answer='provider generated stale answer',
                input_tokens=11,
                output_tokens=7,
                model_name='fake-blocking-rag-model',
            )

    monkeypatch.setattr(
        rag_service,
        'build_default_rag_orchestrator_agent',
        lambda _settings: RagOrchestratorAgent(model=RevokeDuringAnswer()),
    )

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis queues'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['answer'] == '이 답변의 근거를 더 이상 확인할 수 없습니다. 다시 생성해 주세요.'
    assert 'provider generated stale answer' not in response.text
    assert 'redis-decision' not in response.text
    assert 'Approved Redis decision snippet' not in response.text
    assert payload['source_ids'] == []
    assert payload['source_links'] == []
    assert payload['source_snippets'] == []
    assert payload['citations'] == []
    assert payload['permission_level'] is None
    assert payload['hidden_match_count'] == 0
    assert payload['permission_notice'] == 'evidence_unavailable'
    agent_run = db_session.query(AgentRun).one()
    assert agent_run.metadata_['source_count'] == 0
    assert agent_run.metadata_['hidden_match_count'] == 0
    assert agent_run.metadata_['evidence_status'] == 'evidence_unavailable'


def test_ask_api_recomputes_hidden_count_after_hidden_only_provider_mutation(
    client,
    db_session,
    monkeypatch,
) -> None:
    visible = DecisionRecord(
        title='Use Redis for queues',
        decision_summary='Redis should power queue and job progress updates.',
        source_links=['https://knowledge.mock/redis-decision'],
        source_snippets=['Approved Redis decision snippet'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    hidden = DecisionRecord(
        title='Restricted Redis migration',
        decision_summary='Redis migration details remain restricted.',
        source_links=['https://knowledge.mock/restricted-redis'],
        source_snippets=['Restricted Redis migration evidence'],
        confidence_score=0.9,
        permission_level='restricted',
        review_status='approved',
    )
    db_session.add_all([visible, hidden])
    db_session.commit()

    class RevokeHiddenDuringAnswer:
        def answer(self, question, packet):
            assert db_session.in_transaction() is False
            assert packet.source_ids == [f'decision_record:{visible.id}']
            current_hidden = db_session.get(DecisionRecord, hidden.id)
            current_hidden.review_status = 'revoked'
            db_session.commit()
            return RagModelResponse(
                answer='provider answer remains supported by visible evidence',
                input_tokens=11,
                output_tokens=7,
                model_name='fake-blocking-rag-model',
            )

    monkeypatch.setattr(
        rag_service,
        'build_default_rag_orchestrator_agent',
        lambda _settings: RagOrchestratorAgent(model=RevokeHiddenDuringAnswer()),
    )

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis queues'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['answer'] == 'provider answer remains supported by visible evidence'
    assert payload['source_ids'] == [f'decision_record:{visible.id}']
    assert payload['hidden_match_count'] == 0
    assert payload['permission_notice'] is None
    agent_run = db_session.query(AgentRun).one()
    assert agent_run.metadata_['source_count'] == 1
    assert agent_run.metadata_['hidden_match_count'] == 0
    assert 'evidence_status' not in agent_run.metadata_


def test_ask_api_fails_closed_when_hidden_count_recomputation_fails(
    client,
    db_session,
    monkeypatch,
) -> None:
    visible = DecisionRecord(
        title='Use Redis for queues',
        decision_summary='Redis should power queue and job progress updates.',
        source_links=['https://knowledge.mock/redis-decision'],
        source_snippets=['Approved Redis decision snippet'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    hidden = DecisionRecord(
        title='Restricted Redis migration',
        decision_summary='Redis migration details remain restricted.',
        source_links=['https://knowledge.mock/restricted-redis'],
        source_snippets=['Restricted Redis migration evidence'],
        confidence_score=0.9,
        permission_level='restricted',
        review_status='approved',
    )
    db_session.add_all([visible, hidden])
    db_session.commit()

    original_retrieve = rag_service.retrieve_matching_evidence_candidates
    retrieval_calls = 0

    def fail_second_retrieval(*, db, question):
        nonlocal retrieval_calls
        retrieval_calls += 1
        if retrieval_calls == 2:
            raise RuntimeError('synthetic hidden authority lookup failure')
        return original_retrieve(db=db, question=question)

    class AnswerWithoutMutation:
        def answer(self, question, packet):
            assert db_session.in_transaction() is False
            return RagModelResponse(
                answer='provider answer cannot retain stale hidden metadata',
                input_tokens=11,
                output_tokens=7,
                model_name='fake-blocking-rag-model',
            )

    monkeypatch.setattr(
        rag_service,
        'retrieve_matching_evidence_candidates',
        fail_second_retrieval,
    )
    monkeypatch.setattr(
        rag_service,
        'build_default_rag_orchestrator_agent',
        lambda _settings: RagOrchestratorAgent(model=AnswerWithoutMutation()),
    )

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis queues'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert retrieval_calls == 2
    assert payload['answer'] == '이 답변의 근거를 더 이상 확인할 수 없습니다. 다시 생성해 주세요.'
    assert payload['source_ids'] == []
    assert payload['hidden_match_count'] == 0
    assert payload['permission_notice'] == 'evidence_unavailable'
