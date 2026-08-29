from backend.app.models import AgentRun, AgentWorkflowThread, ReviewItem


def test_notifications_api_returns_review_and_agent_run_alerts(client, db_session) -> None:
    db_session.add_all(
        [
            AgentWorkflowThread(
                thread_id='viewer-failed-thread',
                workflow_name='review_sources',
                graph_version='v1',
                checkpoint_thread_id='checkpoint:viewer-failed-thread',
                checkpoint_store='database',
                owner_subject_id='employee-mina',
                security_scope_id='default',
                input_hash='a' * 64,
                evidence_version_hash='b' * 64,
            ),
            AgentWorkflowThread(
                thread_id='other-failed-thread',
                workflow_name='review_sources',
                graph_version='v1',
                checkpoint_thread_id='checkpoint:other-failed-thread',
                checkpoint_store='database',
                owner_subject_id='somebody-else',
                security_scope_id='default',
                input_hash='c' * 64,
                evidence_version_hash='d' * 64,
            ),
        ]
    )
    db_session.add_all(
        [
            ReviewItem(
                item_type='decision_record',
                payload={'title': 'Redis decision'},
                source_links=['https://slack.mock/redis'],
                source_snippets=['Redis decision source'],
                confidence_score=0.82,
                permission_level='internal',
                status='pending_review',
            ),
            ReviewItem(
                item_type='history_event',
                payload={'title': 'Scope history'},
                source_links=['https://gmail.mock/scope'],
                source_snippets=['Scope source'],
                confidence_score=0.62,
                permission_level='internal',
                status='needs_more_evidence',
            ),
            AgentRun(
                agent_name='slack_agent',
                prompt_version='slack-timeline:v1',
                status='failed',
                source_window='slack:test',
                cache_key='failed-run-cache',
                model_name='fake-model',
                permission_level='internal',
                metadata_={'failure_reason': 'provider timeout'},
                workflow_thread_id='viewer-failed-thread',
            ),
            AgentRun(
                agent_name='document_agent',
                prompt_version='document:v1',
                status='failed',
                source_window='drive:secret',
                cache_key='other-failed-run-cache',
                model_name='fake-model',
                permission_level='internal',
                metadata_={'failure_reason': 'raw secret: do not expose'},
                workflow_thread_id='other-failed-thread',
            ),
        ]
    )
    db_session.commit()

    response = client.get(
        '/api/v1/notifications', headers={'X-Demo-User': 'viewer'}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['counts'] == {
        'total': 3,
        'review': 2,
        'agent_runs': 1,
    }
    assert [item['category'] for item in payload['notifications']] == [
        'review',
        'review',
        'agent_run',
    ]
    assert payload['notifications'][0]['action_href'] == '/review'
    assert payload['notifications'][1]['severity'] == 'warning'
    assert payload['notifications'][2]['message'] == 'Agent run failed'
    assert all(
        'raw secret' not in item['message']
        for item in payload['notifications']
    )
