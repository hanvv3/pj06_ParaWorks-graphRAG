import pytest

from backend.app.agent_runtime.canonical_sources import ReviewWorkflowPreflightError
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_preflight import prepare_review_request
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models.source import (
    Document,
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


def test_canonical_refs_sort_dedupe_and_require_prefixed_source_id(
    db_session,
) -> None:
    gmail = _source(
        db_session,
        source_type='gmail',
        source_id='gmail:message-1',
        signature='gmail-signature-1',
    )
    drive = _source(
        db_session,
        source_type='drive',
        source_id='drive:file-1',
        signature='drive-signature-1',
    )
    registry = _registry('mail_document_agent')

    prepared = prepare_review_request(
        db_session,
        request=_request(
            ('gmail', gmail.source_id, 'gmail-signature-1'),
            ('drive', drive.source_id, 'drive-signature-1'),
            ('gmail', gmail.source_id, 'gmail-signature-1'),
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
    source = _source(
        db_session,
        source_type='drive',
        source_id='drive:parsed-file',
        signature='drive:parsed-file:stable-source-signature',
    )
    document = Document(
        source_id=source.id,
        title='Parsed file',
        current_version='v1',
    )
    db_session.add(document)
    db_session.flush()
    version_1 = DocumentVersion(document_id=document.id, version='v1', body='secret-v1')
    db_session.add(version_1)
    db_session.flush()
    db_session.add(
        DocumentParserRun(
            document_id=document.id,
            document_version_id=version_1.id,
            source_id=source.id,
            parser_name='pypdf',
            parser_status='parsed',
            document_version_label='v1',
            revision_id='parser-revision-1',
            content_signature='parser-signature-1',
            chunk_count=1,
        )
    )
    db_session.flush()
    request = _request(
        ('drive', source.source_id, 'drive:parsed-file:stable-source-signature')
    )
    registry = _registry('mail_document_agent')

    first = prepare_review_request(
        db_session,
        request=request,
        actor=_actor(),
        registry=registry,
        settings=_settings(),
    ).source_refs[0]

    version_2 = DocumentVersion(document_id=document.id, version='v2', body='secret-v2')
    db_session.add(version_2)
    db_session.flush()
    db_session.add(
        DocumentParserRun(
            document_id=document.id,
            document_version_id=version_2.id,
            source_id=source.id,
            parser_name='pypdf',
            parser_status='parsed',
            document_version_label='v2',
            revision_id='parser-revision-2',
            content_signature='parser-signature-2',
            chunk_count=1,
        )
    )
    document.current_version = 'v2'
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
    source = _source(
        db_session,
        source_type='drive',
        source_id='drive:parser-provenance',
        signature='drive:parser-provenance:stable-source-signature',
    )
    document = Document(
        source_id=source.id,
        title='Parser provenance',
        current_version='document-v1',
    )
    db_session.add(document)
    db_session.flush()
    document_version = DocumentVersion(
        document_id=document.id,
        version='document-v1',
        body='sensitive parsed content',
    )
    db_session.add(document_version)
    db_session.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=document_version.id,
        source_id=source.id,
        parser_name='pypdf',
        parser_status='parsed',
        document_version_label='parser-version-1',
        revision_id='parser-revision-1',
        content_signature='parser-signature-1',
        chunk_count=1,
    )
    db_session.add(parser_run)
    db_session.flush()
    request = _request(
        (
            'drive',
            source.source_id,
            'drive:parser-provenance:stable-source-signature',
        )
    )
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
    changed = prepare_review_request(
        db_session,
        request=request,
        actor=_actor(),
        registry=registry,
        settings=_settings(),
    ).source_refs[0]

    assert changed.content_fingerprint != baseline.content_fingerprint
