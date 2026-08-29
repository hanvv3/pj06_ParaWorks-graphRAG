from backend.app.api.v1.knowledge import _eligible_for_actor
from backend.app.core.demo_auth import USERS
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibility,
)
from backend.app.models import DecisionRecord, HistoryEvent, TimelineEvent, Todo


def test_knowledge_visibility_projection_does_not_mutate_persisted_permission() -> None:
    record = DecisionRecord(
        id=41,
        title='Permission drift',
        decision_summary='Current evidence narrowed this record.',
        source_links=['https://source.mock/41'],
        source_snippets=['Current evidence'],
        confidence_score=0.9,
        permission_level='public',
        review_status='approved',
    )

    class InternalEligibility:
        def for_knowledge(self, knowledge_type: str, knowledge_id: int):
            return TrustedServingEligibility(True, 'internal')

    projected = _eligible_for_actor(
        InternalEligibility(),
        'decision_record',
        [record],
        USERS['viewer'],
    )

    assert record.permission_level == 'public'
    assert projected[0].permission_level == 'internal'


def test_knowledge_api_returns_approved_company_memory(client, db_session) -> None:
    decision = DecisionRecord(
        title='Use Redis for queues',
        decision_summary='Redis should power queue and job progress updates.',
        source_links=['https://slack.mock/redis'],
        source_snippets=['Redis source snippet'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    history = HistoryEvent(
        title='Project Beta scope changed',
        reason='Advanced diff UI moved out of MVP.',
        source_links=['https://gmail.mock/scope'],
        source_snippets=['Scope source snippet'],
        confidence_score=0.84,
        permission_level='internal',
        review_status='approved',
    )
    timeline = TimelineEvent(
        title='Launch QA completed',
        result_summary='QA passed and launch preparation moved to review.',
        source_links=['https://calendar.mock/launch'],
        source_snippets=['Timeline source snippet'],
        confidence_score=0.86,
        permission_level='internal',
        review_status='approved',
    )
    todo = Todo(
        title='Verify evidence inspection before launch',
        priority='high',
        priority_reason='Evidence must be checked before launch readiness review.',
        source_links=['https://drive.mock/evidence'],
        source_snippets=['Todo source snippet'],
        confidence_score=0.8,
        permission_level='restricted',
        review_status='approved',
    )
    db_session.add_all([decision, history, timeline, todo])
    db_session.commit()

    response = client.get('/api/v1/knowledge')

    assert response.status_code == 200
    payload = response.json()
    assert payload['counts'] == {
        'decisions': 1,
        'history_events': 1,
        'timeline_events': 1,
        'todos': 1,
    }
    assert payload['decisions'][0]['title'] == 'Use Redis for queues'
    assert payload['decisions'][0]['summary'] == 'Redis should power queue and job progress updates.'
    assert payload['decisions'][0]['source_links'] == ['https://slack.mock/redis']
    assert payload['decisions'][0]['confidence_score'] == 0.91
    assert payload['history_events'][0]['summary'] == 'Advanced diff UI moved out of MVP.'
    assert payload['timeline_events'][0]['title'] == 'Launch QA completed'
    assert payload['timeline_events'][0]['summary'] == 'QA passed and launch preparation moved to review.'
    assert payload['todos'][0]['summary'] == 'Evidence must be checked before launch readiness review.'
    assert payload['todos'][0]['permission_level'] == 'restricted'
    assert payload['todos'][0]['review_status'] == 'approved'


def test_knowledge_map_returns_memory_to_evidence_graph(client, db_session) -> None:
    shared_source = 'https://slack.mock/project-alpha/decision'
    decision = DecisionRecord(
        title='Keep Project Alpha in MVP',
        decision_summary='Project Alpha stays in scope because customer demos depend on it.',
        source_links=[shared_source],
        source_snippets=['Alpha decision source snippet'],
        confidence_score=0.92,
        permission_level='internal',
        review_status='approved',
    )
    timeline = TimelineEvent(
        title='Project Alpha demo confirmed',
        result_summary='Demo date was confirmed after the scope decision.',
        source_links=[shared_source],
        source_snippets=['Alpha timeline source snippet'],
        confidence_score=0.87,
        permission_level='restricted',
        review_status='approved',
    )
    db_session.add_all([decision, timeline])
    db_session.commit()

    response = client.get('/api/v1/knowledge/map')

    assert response.status_code == 200
    payload = response.json()
    assert payload['counts'] == {
        'memory_nodes': 2,
        'evidence_nodes': 1,
        'edges': 2,
        'permission_levels': {'internal': 1, 'restricted': 1},
    }
    assert payload['cost_policy'] == {
        'paid_llm_calls': False,
        'embedding_calls': False,
        'sync_jobs_triggered': False,
        'strategy': 'approved_memory_source_link_graph',
    }

    node_ids = {node['id'] for node in payload['nodes']}
    assert {'decision:1', 'timeline_event:1', 'evidence_source:https://slack.mock/project-alpha/decision'} <= node_ids
    evidence_node = next(node for node in payload['nodes'] if node['type'] == 'evidence_source')
    assert evidence_node['label'] == 'slack.mock/project-alpha/decision'
    assert evidence_node['connected_memory_count'] == 2
    assert evidence_node['permission_level'] == 'restricted'

    assert {
        (edge['source'], edge['target'], edge['relationship'], edge['permission_level']) for edge in payload['edges']
    } == {
        ('decision:1', 'evidence_source:https://slack.mock/project-alpha/decision', 'supported_by', 'internal'),
        ('timeline_event:1', 'evidence_source:https://slack.mock/project-alpha/decision', 'supported_by', 'restricted'),
    }
