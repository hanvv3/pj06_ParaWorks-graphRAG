import pytest

from backend.app.agent_runtime.canonical_sources import (
    ReviewWorkflowPreflightError,
    resolve_source_versions,
)
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_preflight import prepare_review_request
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_versions import SourceVersionRef
from backend.app.models.source import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)
from backend.app.schemas.review_workflow import ReviewWorkflowRunRequest


def _actor(
    subject_id: str = 'actor-1',
    *,
    permissions: set[str] | None = None,
) -> DemoUser:
    return DemoUser(
        id=subject_id,
        email=f'{subject_id}@example.test',
        role='employee',
        permission_levels=permissions or {'public', 'internal'},
        name=subject_id,
        title='Tester',
        department='Quality',
    )


def _registry(*names: str) -> AgentRegistry:
    registry = AgentRegistry()
    for name in names:
        registry.register(
            AgentManifest(
                name=name,
                owner='Test Owner',
                input_contract='PreparedReviewRequest',
                output_contract='AgentRunResult',
                prompt_versions=('test:v1',),
                supported_permissions=('public', 'internal', 'restricted'),
                capabilities=('review_draft',),
            )
        )
    return registry


def _source(
    db_session,
    *,
    source_type: str,
    source_id: str,
    signature: str,
    permission_level: str = 'internal',
) -> Source:
    source = Source(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://sensitive.example/{source_id}',
        title='Sensitive source title',
        permission_level=permission_level,
        raw_metadata={
            'content_signature': signature,
            'revision_id': 'external-revision-1',
            'source_snippet': 'sensitive source snippet',
        },
    )
    db_session.add(source)
    db_session.flush()
    return source


def _request(*refs: tuple[str, str, str]) -> ReviewWorkflowRunRequest:
    return ReviewWorkflowRunRequest(
        source_refs=[
            {
                'source_type': source_type,
                'source_id': source_id,
                'version_or_signature': signature,
            }
            for source_type, source_id, signature in refs
        ],
        agent_names=['mail_document_agent'],
    )


def _settings() -> Settings:
    return Settings(
        agent_runtime_security_scope_id='scope-1',
        agent_runtime_fingerprint_secret='test-review-preflight-secret',
        agent_runtime_fingerprint_key_version='test-v1',
    )


def _add_server_document_state(
    db_session,
    *,
    source: Source,
    signature: str,
    version_label: str = 'v1',
    parser_name: str | None = None,
    server_identity: bool = True,
    document: Document | None = None,
) -> tuple[Document, DocumentVersion, DocumentParserRun]:
    if document is None:
        document = Document(
            source_id=source.id,
            title=source.title,
            current_version=version_label,
        )
        db_session.add(document)
        db_session.flush()
    else:
        document.current_version = version_label
    version = DocumentVersion(
        document_id=document.id,
        version=version_label,
        body='canonical source body',
    )
    db_session.add(version)
    db_session.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name=parser_name or f'server_{source.source_type}_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type=(
            'message/rfc822'
            if source.source_type == 'gmail'
            else 'text/calendar'
            if source.source_type == 'calendar'
            else 'text/plain'
        ),
        document_version_label=version_label,
        revision_id='external-revision-1',
        content_signature=signature,
        server_content_signature_schema=(
            'server-source-content:v1' if server_identity else None
        ),
        server_content_signature=signature if server_identity else None,
        parser_policy_version=(
            'server-source-parser-policy:v1' if server_identity else None
        ),
        parser_version=(
            'source-event-paragraph-parser:v1' if server_identity else None
        ),
        chunk_policy_version=(
            'paragraph-chunks:1200:v1' if server_identity else None
        ),
        chunk_count=1,
    )
    db_session.add(parser_run)
    db_session.flush()
    db_session.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=parser_run.id,
            chunk_index=0,
            text='canonical source body',
            source_snippet='canonical source body',
            permission_level=source.permission_level,
            metadata_={},
        )
    )
    document.current_document_version_id = version.id
    db_session.flush()
    return document, version, parser_run


def _server_source(
    db_session,
    *,
    source_type: str = 'drive',
    source_id: str = 'drive:authority-test',
    signature: str = 'a' * 64,
) -> Source:
    source = Source(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://sensitive.example/{source_id}',
        title='Canonical source title',
        permission_level='internal',
        raw_metadata={
            'revision_id': 'external-revision-1',
            'mime_type': 'text/plain',
        },
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
    )
    db_session.add(source)
    db_session.flush()
    return source


def _resolve_one(db_session, source: Source, signature: str):
    return resolve_source_versions(
        db_session,
        refs=(
            SourceVersionRef(
                source_type=source.source_type,
                source_id=source.source_id,
                version_or_signature=signature,
            ),
        ),
        actor=_actor(),
        settings=_settings(),
    )[0]


def test_legacy_raw_only_signature_cannot_authorize_canonical_resolution(
    db_session,
) -> None:
    signature = 'b' * 64
    source = _source(
        db_session,
        source_type='drive',
        source_id='drive:legacy-raw-only',
        signature=signature,
    )

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        _resolve_one(db_session, source, signature)

    assert exc_info.value.code == 'evidence_changed'


def test_display_version_cannot_override_wrong_current_version_pointer(
    db_session,
) -> None:
    signature = 'c' * 64
    source = _server_source(db_session, signature=signature)
    document, pointed_version, _ = _add_server_document_state(
        db_session,
        source=source,
        signature=signature,
        version_label='pointed-v1',
    )
    db_session.query(DocumentChunk).delete()
    db_session.query(DocumentParserRun).delete()
    display_version = DocumentVersion(
        document_id=document.id,
        version='display-v2',
        body='display fallback must not authorize',
    )
    db_session.add(display_version)
    db_session.flush()
    display_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=display_version.id,
        source_id=source.id,
        parser_name='server_drive_source_event',
        parser_status='parsed',
        mime_type='text/plain',
        document_version_label='display-v2',
        revision_id='display-revision',
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version='server-source-parser-policy:v1',
        parser_version='source-event-paragraph-parser:v1',
        chunk_policy_version='paragraph-chunks:1200:v1',
        chunk_count=1,
    )
    db_session.add(display_run)
    db_session.flush()
    db_session.add(
        DocumentChunk(
            version_id=display_version.id,
            source_id=source.id,
            parser_run_id=display_run.id,
            chunk_index=0,
            text='display fallback must not authorize',
            source_snippet='display fallback must not authorize',
            permission_level='internal',
            metadata_={},
        )
    )
    document.current_version = 'display-v2'
    document.current_document_version_id = pointed_version.id
    db_session.flush()

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        _resolve_one(db_session, source, signature)

    assert exc_info.value.code == 'evidence_changed'


def test_ambiguous_current_parser_runs_fail_closed(db_session) -> None:
    signature = 'd' * 64
    source = _server_source(db_session, signature=signature)
    document, version, _ = _add_server_document_state(
        db_session,
        source=source,
        signature=signature,
    )
    duplicate_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_drive_source_event',
        parser_status='parsed',
        mime_type='text/plain',
        document_version_label=version.version,
        revision_id='duplicate-revision',
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version='server-source-parser-policy:v1',
        parser_version='source-event-paragraph-parser:v1',
        chunk_policy_version='paragraph-chunks:1200:v1',
        chunk_count=1,
    )
    db_session.add(duplicate_run)
    db_session.flush()

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        _resolve_one(db_session, source, signature)

    assert exc_info.value.code == 'evidence_changed'


def test_non_server_current_parser_run_fails_closed(db_session) -> None:
    signature = 'e' * 64
    source = _server_source(db_session, signature=signature)
    _add_server_document_state(
        db_session,
        source=source,
        signature=signature,
        parser_name='connector_owned_parser',
        server_identity=False,
    )

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        _resolve_one(db_session, source, signature)

    assert exc_info.value.code == 'evidence_changed'


def test_canonical_refs_sort_dedupe_and_require_prefixed_source_id(
    db_session,
) -> None:
    gmail_signature = '1' * 64
    drive_signature = '2' * 64
    gmail = _server_source(
        db_session,
        source_type='gmail',
        source_id='gmail:message-1',
        signature=gmail_signature,
    )
    drive = _server_source(
        db_session,
        source_type='drive',
        source_id='drive:file-1',
        signature=drive_signature,
    )
    _add_server_document_state(
        db_session,
        source=gmail,
        signature=gmail_signature,
    )
    _add_server_document_state(
        db_session,
        source=drive,
        signature=drive_signature,
    )
    registry = _registry('mail_document_agent')

    prepared = prepare_review_request(
        db_session,
        request=_request(
            ('gmail', gmail.source_id, gmail_signature),
            ('drive', drive.source_id, drive_signature),
            ('gmail', gmail.source_id, gmail_signature),
        ),
        actor=_actor(),
        registry=registry,
        settings=_settings(),
    )

    assert [ref.canonical_row_id for ref in prepared.source_refs] == [
        drive.id,
        gmail.id,
    ]

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        prepare_review_request(
            db_session,
            request=_request(('gmail', 'message-without-prefix', 'signature')),
            actor=_actor(),
            registry=registry,
            settings=_settings(),
        )
    assert exc_info.value.code == 'invalid_input'


def test_canonical_ref_rejects_source_type_mismatch(db_session) -> None:
    _source(
        db_session,
        source_type='drive',
        source_id='gmail:misclassified-row',
        signature='signature-1',
    )

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        prepare_review_request(
            db_session,
            request=_request(('gmail', 'gmail:misclassified-row', 'signature-1')),
            actor=_actor(),
            registry=_registry('mail_document_agent'),
            settings=_settings(),
        )

    assert exc_info.value.code == 'invalid_input'


def test_hidden_source_is_not_distinguishable_from_missing(db_session) -> None:
    _source(
        db_session,
        source_type='gmail',
        source_id='gmail:hidden-correct',
        signature='hidden-signature',
        permission_level='restricted',
    )
    _source(
        db_session,
        source_type='drive',
        source_id='gmail:hidden-wrong-type',
        signature='hidden-signature',
        permission_level='restricted',
    )
    _source(
        db_session,
        source_type='gmail',
        source_id='gmail:hidden-wrong-signature',
        signature='current-hidden-signature',
        permission_level='restricted',
    )
    actor = _actor(permissions={'public', 'internal'})
    registry = _registry('mail_document_agent')
    failures = []

    for source_id, signature in (
        ('gmail:hidden-correct', 'hidden-signature'),
        ('gmail:hidden-wrong-type', 'hidden-signature'),
        ('gmail:hidden-wrong-signature', 'requested-old-signature'),
        ('gmail:missing-message', 'missing-signature'),
    ):
        with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
            prepare_review_request(
                db_session,
                request=_request(('gmail', source_id, signature)),
                actor=actor,
                registry=registry,
                settings=_settings(),
            )
        failures.append((type(exc_info.value), exc_info.value.code, str(exc_info.value)))

    assert all(failure == failures[-1] for failure in failures)
    assert failures[0][1] == 'not_found'
    assert 'hidden' not in failures[0][2]


def test_changed_source_signature_raises_evidence_changed(db_session) -> None:
    _source(
        db_session,
        source_type='calendar',
        source_id='calendar:primary:event-1',
        signature='calendar-signature-current',
    )

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        prepare_review_request(
            db_session,
            request=_request(
                ('calendar', 'calendar:primary:event-1', 'calendar-signature-old')
            ),
            actor=_actor(),
            registry=_registry('mail_document_agent'),
            settings=_settings(),
        )

    assert exc_info.value.code == 'evidence_changed'


def test_parsed_drive_ref_rechecks_current_document_version_and_parser_revision(
    db_session,
) -> None:
    signature = '3' * 64
    source = _server_source(
        db_session,
        source_type='drive',
        source_id='drive:parsed-file',
        signature=signature,
    )
    document, version_1, parser_run_1 = _add_server_document_state(
        db_session,
        source=source,
        signature=signature,
        version_label='v1',
    )
    parser_run_1.revision_id = 'parser-revision-1'
    db_session.flush()
    request = _request(('drive', source.source_id, signature))
    registry = _registry('mail_document_agent')

    first = prepare_review_request(
        db_session,
        request=request,
        actor=_actor(),
        registry=registry,
        settings=_settings(),
    ).source_refs[0]

    _, version_2, parser_run_2 = _add_server_document_state(
        db_session,
        source=source,
        signature=signature,
        version_label='v2',
        document=document,
    )
    parser_run_2.revision_id = 'parser-revision-2'
    db_session.flush()

    second = prepare_review_request(
        db_session,
        request=request,
        actor=_actor(),
        registry=registry,
        settings=_settings(),
    ).source_refs[0]

    assert first.document_version_id == version_1.id
    assert first.external_revision == 'parser-revision-1'
    assert second.document_version_id == version_2.id
    assert second.external_revision == 'parser-revision-2'
    assert second.content_fingerprint != first.content_fingerprint
    assert len(second.content_fingerprint) == 64


@pytest.mark.parametrize(
    ('field', 'changed_value'),
    [
        ('document_version_label', 'parser-version-2'),
        ('revision_id', 'parser-revision-2'),
        ('content_signature', 'parser-signature-2'),
    ],
)
def test_parsed_drive_fingerprint_binds_each_parser_version_field_independently(
    db_session,
    field: str,
    changed_value: str,
) -> None:
    signature = '4' * 64
    source = _server_source(
        db_session,
        source_type='drive',
        source_id='drive:parser-provenance',
        signature=signature,
    )
    _, _, parser_run = _add_server_document_state(
        db_session,
        source=source,
        signature=signature,
        version_label='document-v1',
    )
    parser_run.revision_id = 'parser-revision-1'
    db_session.flush()
    request = _request(('drive', source.source_id, signature))
    registry = _registry('mail_document_agent')
    baseline = prepare_review_request(
        db_session,
        request=request,
        actor=_actor(),
        registry=registry,
        settings=_settings(),
    ).source_refs[0]

    setattr(parser_run, field, changed_value)
    db_session.flush()
    if field in {'document_version_label', 'content_signature'}:
        with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
            prepare_review_request(
                db_session,
                request=request,
                actor=_actor(),
                registry=registry,
                settings=_settings(),
            )
        assert exc_info.value.code == 'evidence_changed'
    else:
        changed = prepare_review_request(
            db_session,
            request=request,
            actor=_actor(),
            registry=registry,
            settings=_settings(),
        ).source_refs[0]
        assert changed.content_fingerprint != baseline.content_fingerprint
