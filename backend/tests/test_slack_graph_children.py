"""Approved-only synthetic Slack graph children; raw serving stays closed."""

from dataclasses import replace

import pytest
from sqlalchemy import select

from backend.app.agent_runtime.rag_v2_identity import ServerRagSecurityScopeResolver
from backend.app.core.config import get_settings
from backend.app.core.demo_auth import USERS
from backend.app.ingestion.source_authority import resolve_exact_source_authority
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import (
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)
from backend.app.rag.graph_projection import read_projection_page
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import ServingEvidenceResolver
from backend.tests.slack_synthetic_fixture import SyntheticSlackClient
from backend.tests.test_slack_synthetic_authority import (
    RecordingSlackModel,
    approve,
    draft,
    local_connector,
)


def prepared(db):
    settings = get_settings()
    scope = ServerRagSecurityScopeResolver(settings).resolve(
        db=db, actor=USERS['viewer']
    )
    fake = SyntheticSlackClient()
    sync_connector_events(db, local_connector(fake))
    fake.add_late_reply()
    sync_connector_events(db, local_connector(fake))
    item = draft(db, RecordingSlackModel(), source_ids=[f'CPUBLIC:{fake.reply_ts}'])[0]
    source = db.scalar(
        select(Source).where(Source.source_id == f'CPUBLIC:{fake.parent_ts}')
    )
    chunk = resolve_exact_source_authority(db, source=source).chunks[0]
    resolver = CanonicalSourceObservationResolver(db=db, settings=settings)
    return settings, scope, fake, item, source, chunk, resolver


def test_raw_slack_stays_hidden_before_and_after_human_approval(db_session):
    settings, scope, fake, item, source, chunk, resolver = prepared(db_session)
    assert resolver.resolve_for_index_strict(chunk.id) is None
    assert resolver.resolve_projection_for_scope_strict(chunk.id, scope=scope) is None
    assert (
        resolver.resolve_approved_slack_child_for_scope_strict(chunk.id, scope=scope)
        is None
    )
    assert not read_projection_page(db_session, settings=settings, scope=scope).edges
    approve(db_session, item)
    assert resolver.resolve_for_index_strict(chunk.id) is None
    assert resolver.resolve_projection_for_scope_strict(chunk.id, scope=scope) is None
    child = resolver.resolve_approved_slack_child_for_scope_strict(
        chunk.id, scope=scope
    )
    assert child is not None and child.source_snippet == chunk.source_snippet
    assert child.evidence.support_mode == 'source_observation'
    from backend.app.rag.evidence_projection import _validate_row
    from backend.app.rag.retrieval import EvidenceSlot

    _validate_row(
        EvidenceSlot('E1', 'source_observation', child.evidence, 1.0, ()),
        child,
        settings=settings,
    )
    assert (
        len(read_projection_page(db_session, settings=settings, scope=scope).edges) == 4
    )
    assert (
        resolver.resolve_approved_slack_child_for_scope_strict(
            chunk.id, scope=replace(scope, workspace_scope_id='other-workspace')
        )
        is None
    )
    assert (
        resolver.resolve_approved_slack_child_for_scope_strict(
            chunk.id, scope=replace(scope, allowed_permission_levels=('public',))
        )
        is None
    )


@pytest.mark.parametrize(
    'corruption', ['inactive', 'signature', 'snippet', 'pending', 'wrong_source']
)
def test_forged_or_revoked_approval_storage_cannot_authorize_child(
    db_session, corruption
):
    settings, scope, fake, item, source, chunk, resolver = prepared(db_session)
    approve(db_session, item)
    original = resolver.resolve_approved_slack_child_for_scope_strict(
        chunk.id, scope=scope
    )
    assert original is not None
    # Negative storage drift only: never used to create a valid approval.
    links = list(
        db_session.scalars(
            select(TrustedKnowledgeApprovalLink).where(
                TrustedKnowledgeApprovalLink.review_item_id == item.id
            )
        )
    )
    if corruption == 'inactive':
        for link in links:
            link.active = False
    elif corruption == 'pending':
        item.status = 'pending_review'
    elif corruption == 'snippet':
        item.source_snippets = ['forged snippet'] * len(item.source_snippets)
    else:
        children = list(
            db_session.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id.in_(
                        [link.id for link in links]
                    )
                )
            )
        )
        for child in children:
            if corruption == 'signature':
                child.canonical_version_or_signature = '0' * 64
            else:
                child.canonical_source_id = '999999'
    db_session.flush()
    assert (
        resolver.resolve_approved_slack_child_for_scope_strict(chunk.id, scope=scope)
        is None
    )
    assert not read_projection_page(db_session, settings=settings, scope=scope).edges
    serving = ServingEvidenceResolver(settings=settings)
    assert (
        serving.resolve_candidate(
            db=db_session, identity=original.identity, scope=scope
        )
        is None
    )
    assert (
        serving.resolve_projection_candidate_strict(
            db=db_session, identity=original.identity, scope=scope
        )
        is None
    )


def test_unsigned_labeled_slack_has_no_graph_child_authority(db_session):
    from backend.app.models import DocumentChunk

    fake = SyntheticSlackClient()
    sync_connector_events(db_session, fake.connector())
    source = db_session.scalar(
        select(Source).where(Source.source_id == f'CPUBLIC:{fake.parent_ts}')
    )
    chunk = db_session.scalar(
        select(DocumentChunk).where(DocumentChunk.source_id == source.id)
    )
    settings = get_settings()
    scope = ServerRagSecurityScopeResolver(settings).resolve(
        db=db_session, actor=USERS['admin']
    )
    assert (
        CanonicalSourceObservationResolver(
            db=db_session, settings=settings
        ).resolve_approved_slack_child_for_scope_strict(chunk.id, scope=scope)
        is None
    )
