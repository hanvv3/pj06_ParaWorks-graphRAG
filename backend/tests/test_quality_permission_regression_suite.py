import hashlib

from sqlalchemy.orm import Session

from backend.app.agent_runtime.company_memory import (
    run_company_memory_agent_orchestration,
)
from backend.app.core.demo_auth import USERS
from backend.app.ingestion.source_content_signature import (
    SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
    server_parser_policy_for_source,
)
from backend.app.models import (
    AgentRun,
    Document,
    DocumentChunk,
    DocumentParserRun,
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
    signature = hashlib.sha256(f'{source_type}:{source_id}:{text}'.encode()).hexdigest()
    raw_metadata = {'ts': '2026-05-02T09:00:00+09:00'}
    if source_type == 'drive':
        raw_metadata['mime_type'] = 'text/plain'
    source = Source(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://{source_type}.mock/{source_id}',
        title=f'{source_type} quality fixture',
        author='quality@example.com',
        permission_level=permission_level,
        raw_metadata=raw_metadata,
        server_content_signature_schema=(
            SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA
            if source_type != 'slack'
            else None
        ),
        server_content_signature=signature if source_type != 'slack' else None,
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()

    parser_run = None
    if source_type != 'slack':
        policy = server_parser_policy_for_source(
            source_type,
            mime_type=raw_metadata.get('mime_type'),
        )
        parser_run = DocumentParserRun(
            document_id=document.id,
            document_version_id=version.id,
            source_id=source.id,
            parser_name=policy.parser_name,
            parser_status='parsed',
            parser_status_reason=None,
            mime_type=policy.mime_type,
            document_version_label='v1',
            content_signature=signature,
            server_content_signature_schema=SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
            server_content_signature=signature,
            parser_policy_version=policy.parser_policy_version,
            parser_version=policy.parser_version,
            chunk_policy_version=policy.chunk_policy_version,
            chunk_count=1,
        )
        db.add(parser_run)
        db.flush()
    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=parser_run.id if parser_run is not None else None,
            chunk_index=0,
            text=text,
            source_snippet=text[:240],
            permission_level=permission_level,
            metadata_={'source_url': source.source_url, 'source_type': source_type},
        )
    )
    document.current_document_version_id = version.id
    db.add(
        ReviewItem(
            item_type='source_evidence',
            payload={'source_ids': [source_id]},
            source_links=[source.source_url],
            source_snippets=[text[:240]],
            confidence_score=1.0,
            permission_level=permission_level,
            status='approved',
            resolution_source='human',
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
        text='Use Redis to support queue and job progress workflows.',
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
        text='Use Redis to support queue and job progress workflows.',
    )
    seed_chunk(
        db_session,
        source_type='gmail',
        source_id='gmail-quality-cache',
        text='PostgreSQL remains durable while Redis handles transient job state.',
    )
    initial_review_count = db_session.query(ReviewItem).count()

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
    assert db_session.query(ReviewItem).count() == initial_review_count + 6
