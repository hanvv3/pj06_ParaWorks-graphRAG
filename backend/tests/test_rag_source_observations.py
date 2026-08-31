from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
)
from backend.app.models import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    ReviewItem,
    Source,
    VectorServingTombstone,
)
from backend.app.rag.serving_contracts import (
    RawServingVersionEnvelope,
    build_model_content_hmac,
)
from backend.app.rag.source_observations import (
    CanonicalSourceObservationEligibilityService,
    CanonicalSourceObservationResolver,
)
from backend.app.rag.trusted_evidence import ServingEvidenceResolver


def _settings(*, secret: str = 'task-4-source-secret-at-least-32-bytes') -> Settings:
    return Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret=secret,
        agent_runtime_fingerprint_key_version='task4-source-v1',
    )


def _scope(
    *,
    source_ids: tuple[int, ...] = (),
    project_keys: tuple[str, ...] = (),
    permissions: tuple[str, ...] = ('public', 'internal', 'restricted'),
) -> SecurityScope:
    source_constraints = tuple(f'source_pk:{value}' for value in source_ids)
    project_constraints = tuple(f'project_key:{value}' for value in project_keys)
    constrained = bool(source_constraints or project_constraints)
    return SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='task4-actor',
        workspace_scope_id='workspace-a',
        resource_scope_mode='constrained' if constrained else 'all_current_scope',
        project_constraints=project_constraints,
        source_constraints=source_constraints,
        allowed_permission_levels=permissions,
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )


def _canonical_snippet(value: str) -> str:
    return ' '.join(value.split())[:240]


def _seed_source_chunk(
    db: Session,
    *,
    source_type: str = 'gmail',
    permission_level: str = 'internal',
    chunk_permission: str | None = None,
    text: str = 'Exact  current\nsource observation',
    signature: str = 'a' * 64,
    revision_id: str = 'revision-1',
) -> tuple[Source, Document, DocumentVersion, DocumentParserRun, DocumentChunk]:
    source = Source(
        source_type=source_type,
        source_id=f'{source_type}:source-1',
        source_url='https://mail.example.test/messages/1',
        title='Canonical source',
        author='owner@example.test',
        permission_level=permission_level,
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
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
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name=f'server_{source_type}_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type=(
            'message/rfc822'
            if source_type == 'gmail'
            else 'text/calendar'
            if source_type == 'calendar'
            else 'application/octet-stream'
        ),
        document_version_label=version.version,
        revision_id=revision_id,
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        chunk_count=1,
    )
    db.add(parser_run)
    db.flush()
    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        parser_run_id=parser_run.id,
        chunk_index=0,
        text=text,
        source_snippet=_canonical_snippet(text),
        permission_level=chunk_permission or permission_level,
        metadata_={},
    )
    db.add(chunk)
    document.current_document_version_id = version.id
    db.commit()
    return source, document, version, parser_run, chunk


@pytest.mark.parametrize(
    'source_type',
    ['gmail', 'gmail_attachment', 'drive', 'calendar'],
)
def test_current_signed_google_chunk_resolves_one_exact_raw_serving_identity(
    db_session: Session,
    source_type: str,
) -> None:
    source, document, version, parser_run, chunk = _seed_source_chunk(
        db_session,
        source_type=source_type,
        permission_level='public',
        chunk_permission='restricted',
    )

    observation = CanonicalSourceObservationResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index(chunk.id)

    assert observation is not None
    assert observation.identity.serving_document_id == f'chunk:{chunk.id}'
    assert observation.identity.serving_kind == 'raw_chunk'
    assert observation.identity.public_source_id == source.source_id
    assert observation.identity.public_source_type == source_type
    assert observation.identity.effective_permission == 'restricted'
    assert isinstance(observation.identity.version_envelope, RawServingVersionEnvelope)
    assert observation.identity.version_envelope is observation.raw_version
    assert observation.evidence.version_envelope is observation.raw_version
    assert observation.evidence.provenance.raw_version is observation.raw_version
    assert observation.evidence.serving_kind == 'raw_chunk'
    assert observation.evidence.support_mode == 'source_observation'
    assert observation.evidence.model_content == chunk.text
    assert observation.raw_version == RawServingVersionEnvelope(
        serving_document_id=f'chunk:{chunk.id}',
        source_row_id=source.id,
        public_source_id=source.source_id,
        document_id=document.id,
        document_version_id=version.id,
        current_document_version_id=version.id,
        document_chunk_id=chunk.id,
        parser_run_id=parser_run.id,
        external_revision='revision-1',
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='a' * 64,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        model_content_hmac=observation.evidence.model_content_hmac,
        canonical_citation_projection_hmac=(
            observation.evidence.canonical_citation_projection_hmac
        ),
        effective_permission='restricted',
    )
    assert observation.identity.model_content_hmac == build_model_content_hmac(
        serving_kind='raw_chunk',
        model_content=chunk.text,
        settings=_settings(),
    )
    assert observation.evidence.serving_version_fingerprint == (
        observation.identity.serving_version_fingerprint
    )


@pytest.mark.parametrize(
    ('mutation', 'expected_none'),
    [
        ('slack', True),
        ('bad_url', True),
        ('snippet_drift', True),
        ('stale_version', True),
        ('parser_signature_drift', True),
        ('unknown_permission', True),
        ('tombstoned', True),
    ],
)
def test_raw_resolver_rejects_unsupported_stale_or_noncanonical_evidence(
    db_session: Session,
    mutation: str,
    expected_none: bool,
) -> None:
    source, document, _, parser_run, chunk = _seed_source_chunk(db_session)
    if mutation == 'slack':
        source.source_type = 'slack'
        source.source_id = 'slack:source-1'
    elif mutation == 'bad_url':
        source.source_url = 'javascript:alert(1)'
    elif mutation == 'snippet_drift':
        chunk.source_snippet = 'not the canonical snippet'
    elif mutation == 'stale_version':
        replacement = DocumentVersion(
            document_id=document.id,
            version='v2',
            body='replacement',
        )
        db_session.add(replacement)
        db_session.flush()
        document.current_document_version_id = replacement.id
    elif mutation == 'parser_signature_drift':
        parser_run.server_content_signature = 'b' * 64
    elif mutation == 'unknown_permission':
        chunk.permission_level = 'unknown'
    elif mutation == 'tombstoned':
        item = ReviewItem(
            item_type='history_event',
            payload={'title': 'revoked'},
            source_links=[source.source_url],
            source_snippets=[chunk.source_snippet],
            confidence_score=1.0,
            permission_level='internal',
            status='approved',
        )
        db_session.add(item)
        db_session.flush()
        db_session.add(
            VectorServingTombstone(
                document_id=f'chunk:{chunk.id}',
                source_review_item_id=item.id,
                reason_code='source_invalidated',
            )
        )
    db_session.commit()

    resolved = CanonicalSourceObservationResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index(chunk.id)

    assert (resolved is None) is expected_none


def test_raw_access_classification_keeps_scope_and_permission_separate(
    db_session: Session,
) -> None:
    source, _, _, _, chunk = _seed_source_chunk(
        db_session,
        permission_level='restricted',
    )
    observation = CanonicalSourceObservationResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index(chunk.id)
    assert observation is not None
    classifier = CanonicalSourceObservationEligibilityService()

    visible = classifier.classify_access(
        _scope(source_ids=(source.id,), permissions=('restricted',)),
        observation,
    )
    denied = classifier.classify_access(
        _scope(source_ids=(source.id,), permissions=('public', 'internal')),
        observation,
    )
    outside = classifier.classify_access(
        _scope(source_ids=(source.id + 1,), permissions=('restricted',)),
        observation,
    )
    invalid = classifier.classify_access(
        _scope(project_keys=('project-a',), permissions=('restricted',)),
        observation,
    )

    assert visible == replace(
        visible,
        global_eligibility='eligible',
        resource_scope='in_scope',
        permission_visibility='visible',
    )
    assert denied.permission_visibility == 'denied_known'
    assert denied.resource_scope == 'in_scope'
    assert outside.resource_scope == 'out_of_scope'
    assert invalid.resource_scope == 'invalid_scope'


def test_final_candidate_resolution_rejects_vector_identity_mismatch_and_denied_scope(
    db_session: Session,
) -> None:
    source, _, _, _, chunk = _seed_source_chunk(db_session)
    canonical = CanonicalSourceObservationResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index(chunk.id)
    assert canonical is not None
    resolver = ServingEvidenceResolver(settings=_settings())

    visible = resolver.resolve_candidate(
        db=db_session,
        identity=canonical.identity,
        scope=_scope(source_ids=(source.id,)),
    )
    wrong_document = resolver.resolve_candidate(
        db=db_session,
        identity=replace(
            canonical.identity,
            serving_document_id=f'chunk:{chunk.id + 100}',
        ),
        scope=_scope(source_ids=(source.id,)),
    )
    denied = resolver.resolve_candidate(
        db=db_session,
        identity=canonical.identity,
        scope=_scope(source_ids=(source.id,), permissions=('public',)),
    )

    assert visible == canonical.evidence
    assert wrong_document is None
    assert denied is None


def test_model_content_hmac_preserves_utf8_bytes_and_rotates_with_key_material() -> None:
    composed = build_model_content_hmac(
        serving_kind='raw_chunk',
        model_content='é 🧭',
        settings=_settings(),
    )
    decomposed = build_model_content_hmac(
        serving_kind='raw_chunk',
        model_content='é 🧭',
        settings=_settings(),
    )
    rotated = build_model_content_hmac(
        serving_kind='raw_chunk',
        model_content='é 🧭',
        settings=_settings(secret='task-4-rotated-secret-at-least-32-bytes'),
    )

    assert composed == 'd32bb863591391514bd34305de68ef2fd1e644dd62147c5adba79592b1a653c9'
    assert decomposed == '55811b95c551147851562170b158ecbae672a927ebc1d87799e8955faffadee7'
    assert rotated == '54896efde556b73ff1df8b6ea20e58ac0a8c527faef49d009e1de41e8a13dc77'
