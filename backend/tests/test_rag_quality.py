import hashlib

from sqlalchemy.orm import Session

from backend.app.ingestion.source_content_signature import (
    SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
    server_parser_policy_for_source,
)
from backend.app.models import (
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
    title: str,
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
        title=title,
        author='owner@example.com',
        permission_level=permission_level,
        raw_metadata=raw_metadata,
        server_content_signature_schema=SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
        server_content_signature=signature,
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()

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
            parser_run_id=parser_run.id,
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


def test_search_returns_ranked_citations_with_matched_terms(client, db_session) -> None:
    seed_chunk(
        db_session,
        source_type='gmail',
        source_id='gmail-redis-queue',
        title='Redis queue progress decision',
        text='Redis should store transient queue state and job progress updates.',
    )
    seed_chunk(
        db_session,
        source_type='drive',
        source_id='drive-redis-standup',
        title='Redis standup',
        text='Redis was mentioned during standup, but no queue decision was made.',
    )

    response = client.post(
        '/api/v1/search',
        headers={'X-Demo-User': 'viewer'},
        json={'query': 'Redis queue progress'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [result['source_id'] for result in payload['results']] == [
        'gmail-redis-queue',
        'drive-redis-standup',
    ]
    first = payload['results'][0]
    assert first['relevance_score'] > payload['results'][1]['relevance_score']
    assert first['matched_terms'] == ['redis', 'queue', 'progress']
    assert first['citation']['source_id'] == 'gmail-redis-queue'
    assert first['citation']['source_url'] == 'https://gmail.mock/gmail-redis-queue'
    assert first['citation']['source_type'] == 'gmail'
    assert first['citation']['permission_level'] == 'internal'
    assert first['citation']['source_snippet'] == 'Redis should store transient queue state and job progress updates.'


def test_ask_response_includes_ranked_citations_and_hides_restricted_matches(client, db_session) -> None:
    seed_chunk(
        db_session,
        source_type='gmail',
        source_id='gmail-visible-redis',
        title='Redis queue progress',
        text='Redis should store transient queue state and job progress updates.',
        permission_level='internal',
    )
    seed_chunk(
        db_session,
        source_type='drive',
        source_id='drive-restricted-pricing',
        title='Restricted Redis pricing',
        text='Redis queue pricing is confidential and requires finance approval.',
        permission_level='restricted',
    )

    response = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Redis queue progress pricing'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['hidden_match_count'] == 1
    assert payload['citations'][0]['source_id'] == 'gmail-visible-redis'
    assert payload['citations'][0]['source_url'] == 'https://gmail.mock/gmail-visible-redis'
    assert payload['citations'][0]['matched_terms'] == ['redis', 'queue', 'progress']
    assert payload['citations'][0]['relevance_score'] > 0
    assert all(citation['permission_level'] != 'restricted' for citation in payload['citations'])
