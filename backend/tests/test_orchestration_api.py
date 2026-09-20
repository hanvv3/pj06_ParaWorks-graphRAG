from sqlalchemy.orm import Session

from backend.app.models import Document, DocumentChunk, DocumentVersion, Source


def seed_chunk(db: Session, source_type: str, source_id: str, text: str, permission_level: str = 'internal') -> None:
    source = Source(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://{source_type}.mock/{source_id}',
        title=f'{source_type} evidence',
        author='owner@example.com',
        permission_level=permission_level,
        raw_metadata={'ts': '2026-05-02T09:00:00+09:00'},
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()

    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            chunk_index=0,
            text=text,
            source_snippet=text[:240],
            permission_level=permission_level,
            metadata_={'source_url': source.source_url, 'source_type': source_type},
        )
    )
    db.commit()


def test_legacy_company_memory_status_is_metadata_only(client) -> None:
    response = client.get('/api/v1/orchestration/company-memory')

    assert response.status_code == 200
    payload = response.json()
    assert payload['backend'] == 'langgraph'
    assert payload['workflow_name'] == 'company_memory'
    assert payload['node_names'] == [
        'collect_evidence',
        'draft_review_candidates',
        'retrieve_company_memory',
        'answer_with_rag',
    ]
    assert 'collect_evidence' in payload['graph_mermaid']
    assert payload['cost_policy'] == {
        'delta_sync': True,
        'source_hash_skip': True,
        'evidence_cache_reuse': True,
        'evidence_token_budget': True,
        'per_run_budget_usd': 0.001,
        'budget_actions': ['run', 'skip', 'use_cache'],
        'paid_llm_calls_in_status_api': False,
        'requires_explicit_run': True,
        'hitl_checkpointing': False,
        'checkpoint_store': 'none',
        'review_boundary': 'metadata_only',
        'trusted_knowledge_requires_approval': True,
    }


def test_company_memory_orchestration_api_runs_deterministic_dry_run(client) -> None:
    response = client.post(
        '/api/v1/orchestration/company-memory/dry-run',
        json={'objective': 'answer_from_company_memory', 'question': 'Redis queue state'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['backend'] == 'langgraph'
    assert payload['objective'] == 'answer_from_company_memory'
    assert payload['completed_nodes'] == [
        'collect_evidence',
        'draft_review_candidates',
        'retrieve_company_memory',
        'answer_with_rag',
    ]
    assert payload['outputs']['review_boundary'] == 'human_approval_required'
    assert payload['outputs']['token_budget_policy'] == 'delta_sync_hash_skip_evidence_budget'
    assert payload['token_cost_usd'] == 0


def test_company_memory_orchestration_api_runs_agent_services(client, db_session) -> None:
    seed_chunk(
        db_session,
        'slack',
        'slack-api-redis',
        'Use Redis to support queue and job progress workflows.',
    )
    seed_chunk(
        db_session,
        'gmail',
        'gmail-api-redis',
        'PostgreSQL remains durable while Redis handles transient job state.',
    )

    response = client.post(
        '/api/v1/orchestration/company-memory/run',
        json={'question': 'Redis job state'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['backend'] == 'langgraph'
    assert payload['completed_nodes'] == [
        'collect_evidence',
        'draft_review_candidates',
        'retrieve_company_memory',
        'answer_with_rag',
    ]
    assert payload['outputs']['slack_review_items_created'] == 1
    assert payload['outputs']['mail_document_review_items_created'] == 1
    assert payload['outputs']['review_boundary'] == 'metadata_only'
    assert payload['outputs']['hitl_checkpoint']['status'] == 'metadata_only'
    assert payload['outputs']['hitl_checkpoint']['checkpoint_type'] == 'review_queue_metadata'
    assert len(payload['outputs']['hitl_checkpoint']['review_item_ids']) == 6
    assert payload['outputs']['rag_agent_run_created'] is True
    assert payload['cost_policy']['requires_explicit_run'] is True
    assert payload['cost_policy']['hitl_checkpointing'] is False
    assert payload['cost_policy']['per_run_budget_usd'] == 0.001
