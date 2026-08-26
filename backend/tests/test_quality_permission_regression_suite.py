from sqlalchemy.orm import Session

from backend.app.agent_runtime.company_memory import (
    run_company_memory_agent_orchestration,
)
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AgentRun,
    Document,
    DocumentChunk,
    DocumentVersion,
    ReviewItem,
    Source,
)


def seed_chunk(
    db: Session,
    *,
    source_type: str,
    source_id: str,
    text: str,
    permission_level: str = 'internal',
) -> None:
    source = Source(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://{source_type}.mock/{source_id}',
        title=f'{source_type} quality fixture',
        author='quality@example.com',
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


def test_quality_suite_rejects_source_less_review_approval(client, db_session) -> None:
    item = ReviewItem(
        item_type='decision_record',
        payload={'title': 'Source-less AI decision', 'decision_summary': 'This must not be trusted.'},
        source_links=[],
        source_snippets=[],
        confidence_score=0.99,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    response = client.post(f'/api/v1/review/{item.id}/approve')

    assert response.status_code == 400
    assert response.json()['detail'] == 'Review item requires source evidence'


def test_quality_suite_viewer_rag_reports_hidden_restricted_match_without_leaking_content(
    client,
    db_session,
) -> None:
    seed_chunk(
        db_session,
        source_type='gmail',
        source_id='gmail-visible-redis-quality',
        text='Redis queue progress is safe to discuss with the delivery team.',
    )
    seed_chunk(
        db_session,
        source_type='drive',
        source_id='drive-restricted-pricing-quality',
        text='Redis queue pricing is confidential and should not be exposed to employees.',
        permission_level='restricted',
    )

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis queue pricing progress'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['hidden_match_count'] == 1
    assert payload['permission_notice'] == 'Some sources may be hidden by permissions.'
    assert all('confidential' not in snippet.lower() for snippet in payload['source_snippets'])
    assert all(citation['permission_level'] != 'restricted' for citation in payload['citations'])


def test_quality_suite_company_memory_emits_review_checkpoint_without_paid_calls(db_session) -> None:
    seed_chunk(
        db_session,
        source_type='slack',
        source_id='slack-quality-redis',
        text='Redis should support queue and job progress workflows.',
    )
    seed_chunk(
        db_session,
        source_type='gmail',
        source_id='gmail-quality-redis',
        text='PostgreSQL remains durable while Redis handles transient job state.',
    )

    result = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='Redis job state',
    )

    checkpoint = result.outputs['hitl_checkpoint']
    assert checkpoint['status'] == 'metadata_only'
    assert checkpoint['checkpoint_type'] == 'review_queue_metadata'
    assert checkpoint['trusted_knowledge_requires_approval'] is True
    assert checkpoint['paid_llm_calls'] is False
    assert len(checkpoint['review_item_ids']) == 6
    assert db_session.query(ReviewItem).filter(ReviewItem.status == 'pending_review').count() == 6


def test_quality_suite_cache_hit_does_not_duplicate_agent_runs_or_review_items(db_session) -> None:
    seed_chunk(
        db_session,
        source_type='slack',
        source_id='slack-quality-cache',
        text='Redis should support queue and job progress workflows.',
    )
    seed_chunk(
        db_session,
        source_type='gmail',
        source_id='gmail-quality-cache',
        text='PostgreSQL remains durable while Redis handles transient job state.',
    )

    first = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='Redis job state',
    )
    second = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='Redis job state',
    )

    assert first.outputs['hitl_checkpoint']['status'] == 'metadata_only'
    assert second.outputs['hitl_checkpoint']['status'] == 'metadata_only'
    assert second.outputs['cost_plan']['slack_agent']['action'] == 'use_cache'
    assert second.outputs['cost_plan']['mail_document_agent']['action'] == 'use_cache'
    assert second.outputs['cost_plan']['rag_orchestrator_agent']['action'] == 'use_cache'
    assert db_session.query(AgentRun).count() == 7
    assert db_session.query(ReviewItem).count() == 6
