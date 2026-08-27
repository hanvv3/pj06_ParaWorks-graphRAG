from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.models.source import (
    Document,
    DocumentParserRun,
    DocumentVersion,
    Source,
)


def test_list_documents(client: TestClient, db_session: Session) -> None:
    source = Source(source_type='drive', source_id='test-doc-1', source_url='http://example.com/1', title='Test Spec', permission_level='public')
    db_session.add(source)
    db_session.flush()

    doc = Document(source_id=source.id, title='Test Spec', current_version='v1')
    db_session.add(doc)
    db_session.flush()

    doc_version = DocumentVersion(document_id=doc.id, version='v1', body='body')
    db_session.add(doc_version)
    db_session.flush()

    parser_run = DocumentParserRun(
        document_id=doc.id,
        document_version_id=doc_version.id,
        source_id=source.id,
        parser_name='google_docs_export',
        parser_status='parsed',
        parser_status_reason='',
        revision_id='rev-1',
        chunk_count=2,
    )
    db_session.add(parser_run)
    db_session.commit()

    response = client.get('/api/v1/documents')
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]['title'] == 'Test Spec'
    assert data[0]['parser_status'] == 'parsed'
    assert data[0]['chunk_count'] == 2


def test_get_document_versions(client: TestClient, db_session: Session) -> None:
    source = Source(source_type='drive', source_id='test-doc-2', source_url='http://example.com/2', title='Test Spec', permission_level='public')
    db_session.add(source)
    db_session.flush()

    doc = Document(source_id=source.id, title='Test Spec', current_version='v2')
    db_session.add(doc)
    db_session.flush()

    doc_version1 = DocumentVersion(document_id=doc.id, version='v1', body='body1')
    doc_version2 = DocumentVersion(document_id=doc.id, version='v2', body='body2')
    db_session.add_all([doc_version1, doc_version2])
    db_session.flush()

    parser_run1 = DocumentParserRun(
        document_id=doc.id,
        document_version_id=doc_version1.id,
        source_id=source.id,
        parser_name='google_docs_export',
        parser_status='parsed',
        parser_status_reason='',
        revision_id='rev-1',
        chunk_count=1,
    )
    parser_run2 = DocumentParserRun(
        document_id=doc.id,
        document_version_id=doc_version2.id,
        source_id=source.id,
        parser_name='google_docs_export',
        parser_status='parsed',
        parser_status_reason='',
        revision_id='rev-2',
        chunk_count=3,
    )
    db_session.add_all([parser_run1, parser_run2])
    db_session.commit()

    response = client.get(f'/api/v1/documents/{doc.id}/versions')
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]['revision_id'] == 'rev-2'
    assert data[1]['revision_id'] == 'rev-1'


def test_get_parser_stats(client: TestClient, db_session: Session) -> None:
    source = Source(source_type='drive', source_id='test-doc-3', source_url='http://example.com/3', title='Spec', permission_level='public')
    db_session.add(source)
    db_session.flush()

    doc = Document(source_id=source.id, title='Spec', current_version='v1')
    db_session.add(doc)
    db_session.flush()

    doc_version = DocumentVersion(document_id=doc.id, version='v1', body='body')
    db_session.add(doc_version)
    db_session.flush()

    db_session.add(DocumentParserRun(
        document_id=doc.id, document_version_id=doc_version.id, source_id=source.id,
        parser_name='pypdf', parser_status='parsed', parser_status_reason='',
        mime_type='application/pdf', revision_id='rev-1', chunk_count=3,
    ))
    db_session.add(DocumentParserRun(
        document_id=doc.id, document_version_id=doc_version.id, source_id=source.id,
        parser_name='google_drive_metadata', parser_status='unsupported',
        parser_status_reason='hwp_parser_not_decided',
        mime_type='application/x-hwp', revision_id='rev-2', chunk_count=0,
    ))
    db_session.commit()

    response = client.get('/api/v1/documents/parser-stats')
    assert response.status_code == 200
    body = response.json()
    assert 'stats' in body
    assert 'summary' in body
    assert body['summary']['total_documents'] == 2
    assert body['summary']['parsed_count'] == 1
    assert body['summary']['unsupported_count'] == 1
    pdf_stat = next((s for s in body['stats'] if s['mime_type'] == 'application/pdf'), None)
    assert pdf_stat is not None
    assert pdf_stat['parser_status'] == 'parsed'
    assert pdf_stat['total_chunks'] == 3


def test_get_document_parser_runs(client: TestClient, db_session: Session) -> None:
    source = Source(source_type='drive', source_id='test-doc-4', source_url='http://example.com/4', title='Doc', permission_level='public')
    db_session.add(source)
    db_session.flush()

    doc = Document(source_id=source.id, title='Doc', current_version='v2')
    db_session.add(doc)
    db_session.flush()

    dv1 = DocumentVersion(document_id=doc.id, version='v1', body='body1')
    dv2 = DocumentVersion(document_id=doc.id, version='v2', body='body2')
    db_session.add_all([dv1, dv2])
    db_session.flush()

    run1 = DocumentParserRun(
        document_id=doc.id, document_version_id=dv1.id, source_id=source.id,
        parser_name='pypdf', parser_status='parsed', parser_status_reason='',
        mime_type='application/pdf', revision_id='rev-1', chunk_count=1,
    )
    run2 = DocumentParserRun(
        document_id=doc.id, document_version_id=dv2.id, source_id=source.id,
        parser_name='pypdf', parser_status='parsed', parser_status_reason='',
        mime_type='application/pdf', revision_id='rev-2', chunk_count=5,
    )
    db_session.add_all([run1, run2])
    db_session.commit()

    response = client.get(f'/api/v1/documents/{doc.id}/parser-runs')
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 2
    assert runs[0]['revision_id'] == 'rev-2'
    assert runs[0]['chunk_count'] == 5
    assert runs[1]['revision_id'] == 'rev-1'
