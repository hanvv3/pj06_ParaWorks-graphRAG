"""Approved legacy-only integrity: actual new writes, current authority and bytes."""

import copy
import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from backend.app.agents.rag_orchestrator_agent.service import (
    RagEvidenceCandidate,
    build_serving_dependency_snapshot,
    citation_from_candidate,
)
from backend.app.assistant.service import (
    append_assistant_message,
    create_conversation,
    eligible_context_messages,
    serialize_message,
)
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
)
from backend.tests.test_assistant_service import _seed_explicit_history
from backend.tests.test_rag_v2_runtime_migration import sqlite_migration  # noqa: F401


def write_explicit(
    db, *, email=False, resolution='auto_policy', output_permission='internal'
):
    history, item, source, approval = _seed_explicit_history(
        db, resolution_source=resolution
    )
    candidate = RagEvidenceCandidate(
        source_id=f'history_event:{history.id}',
        source_url=history.source_links[0],
        text=f'{history.title}\n{history.reason}',
        source_snippet=history.source_snippets[0],
        author=None,
        timestamp=history.created_at.isoformat(),
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    dependency = build_serving_dependency_snapshot(db, candidate)
    conversation = create_conversation(db, USERS['viewer'])
    metadata = {
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v1',
    }
    if email:
        metadata.update(
            action_type='email_draft',
            status='pending_approval',
            email_draft={
                'to': ['partner@example.com'],
                'subject': '회의 안내',
                'body': '정확한 근거 내용',
            },
        )
    message = append_assistant_message(
        db,
        USERS['viewer'],
        conversation,
        content='Bound answer',
        citations=[] if email else [citation_from_candidate(candidate)],
        source_ids=[] if email else [candidate.source_id],
        source_links=[] if email else [candidate.source_url],
        source_snippets=[] if email else [candidate.source_snippet],
        permission_level=None if email else output_permission,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata=metadata,
        serving_dependencies=(dependency,),
        evidence_derived=email,
    )
    return message, history, item, source, approval


@pytest.mark.parametrize('field', ('agent_name', 'effective_backend'))
def test_persisted_origin_metadata_cannot_be_removed_and_defaulted(db_session, field):
    message, *_ = write_explicit(db_session)
    metadata = dict(message.metadata_)
    del metadata[field]
    message.metadata_ = metadata
    db_session.commit()
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['metadata'][
            'status'
        ]
        == 'evidence_unavailable'
    )


def test_new_legacy_output_cannot_broaden_current_evidence_permission(db_session):
    with pytest.raises(ValueError, match='permission'):
        write_explicit(db_session, output_permission='public')
    assert db_session.query(AssistantMessage).count() == 0


@pytest.mark.parametrize(
    'drift',
    (
        'claim',
        'review_bytes',
        'evidence_hash',
        'source_version',
        'permission',
        'inactive',
        'revoked',
    ),
)
def test_exact_trusted_current_authority_drift_redacts(db_session, drift):
    from datetime import UTC, datetime

    from backend.app.models import TrustedKnowledgeEvidenceLink

    message, _, item, source, approval = write_explicit(db_session)
    if drift == 'claim':
        approval.claim_fingerprint = 'b' * 64
    elif drift == 'review_bytes':
        item.source_snippets = ['changed review bytes']
    elif drift == 'evidence_hash':
        db_session.scalar(select(TrustedKnowledgeEvidenceLink)).evidence_hash = 'b' * 64
    elif drift == 'source_version':
        source.server_content_signature = 'b' * 64
    elif drift == 'permission':
        source.permission_level = 'restricted'
    elif drift == 'inactive':
        approval.active = False
    else:
        approval.revoked_at = datetime.now(UTC)
    db_session.commit()
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['metadata'][
            'status'
        ]
        == 'evidence_unavailable'
    )
    assert eligible_context_messages(db_session, USERS['viewer'], [message]) == []


@pytest.mark.parametrize(
    'mutation', ('influence_content', 'public_order', 'child_order')
)
def test_ordered_selected_and_uncited_legacy_dependencies_remain_bound(
    db_session, mutation
):
    from backend.app.agents.rag_orchestrator_agent.service import (
        retrieve_matching_evidence_candidates,
    )
    from backend.app.models import AutoReviewRuntimeKeyState, DocumentChunk
    from backend.tests.test_rag_orchestrator_service import seed_chunk

    for identifier in ('first', 'second', 'uncited'):
        # This single-source fixture seeds its own key row. Reset only that
        # fixture row before the next source, before any message is written.
        db_session.query(AutoReviewRuntimeKeyState).delete()
        db_session.commit()
        seed_chunk(
            db_session,
            'gmail',
            identifier,
            f'Ordered evidence {identifier}',
            'internal',
        )
    candidates = retrieve_matching_evidence_candidates(
        db=db_session, question='Ordered'
    )
    assert len(candidates) == 3
    selected = [
        citation_from_candidate(candidate) for candidate in reversed(candidates[:2])
    ]
    conversation = create_conversation(db_session, USERS['viewer'])
    message = append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='Ordered answer',
        citations=selected,
        source_ids=[value['source_id'] for value in selected],
        source_links=[value['source_url'] for value in selected],
        source_snippets=[value['source_snippet'] for value in selected],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
        serving_dependencies=tuple(
            build_serving_dependency_snapshot(db_session, candidate)
            for candidate in candidates
        ),
    )
    children = list(
        db_session.scalars(
            select(AssistantMessageEvidenceDependency).order_by(
                AssistantMessageEvidenceDependency.candidate_ordinal
            )
        )
    )
    assert [child.dependency_role for child in children] == [
        'selected_citation',
        'selected_citation',
        'legacy_evidence_influence',
    ]
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['content']
        == 'Ordered answer'
    )
    if mutation == 'influence_content':
        db_session.get(
            DocumentChunk, children[-1].document_chunk_id
        ).text = 'changed uncited evidence'
    elif mutation == 'public_order':
        message.citations = list(reversed(message.citations))
        message.source_ids = list(reversed(message.source_ids))
        message.source_links = list(reversed(message.source_links))
        message.source_snippets = list(reversed(message.source_snippets))
    else:
        fields = [
            column.name
            for column in AssistantMessageEvidenceDependency.__table__.columns
            if column.name not in {'id', 'candidate_ordinal', 'created_at'}
        ]
        first, second = [
            {field: getattr(child, field) for field in fields} for child in children[:2]
        ]
        # Move the first canonical id aside within this transaction to preserve
        # its unique constraint while swapping complete payloads.
        children[0].serving_document_id = 'chunk:999999'
        db_session.flush([children[0]])
        for field, value in first.items():
            setattr(children[1], field, value)
        db_session.flush([children[1]])
        for child, values in ((children[0], second), (children[1], first)):
            for field, value in values.items():
                setattr(child, field, value)
    db_session.commit()
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['metadata'][
            'status'
        ]
        == 'evidence_unavailable'
    )


def test_legacy_v3_sqlite_model_and_frozen_migration_parity():
    from backend.app.db.assistant_legacy_guards import sqlite_guard_statements
    from backend.app.models import AssistantMessageEvidenceDependency

    migration = importlib.import_module(
        'backend.migrations.versions.c6f7a8b9c0d1_harden_legacy_v3_database_guards'
    )
    assert tuple(migration._sqlite_statements()) == sqlite_guard_statements()
    constraints = {
        constraint.name: str(constraint.sqltext)
        for table in (
            AssistantMessage.__table__,
            AssistantMessageEvidenceDependency.__table__,
        )
        for constraint in table.constraints
        if hasattr(constraint, 'sqltext')
    }
    assert all(
        constraints[name] == value for name, value in migration.NEW_CHECKS.items()
    )


@pytest.mark.parametrize('foreign_keys', (False, True))
def test_published_v3_parent_delete_rejects_surviving_authority(
    db_session, foreign_keys
):
    message, *_ = write_explicit(db_session)
    identifier = message.id
    db_session.commit()
    db_session.connection().exec_driver_sql(
        f'PRAGMA foreign_keys={int(foreign_keys)}'
    )

    with pytest.raises(IntegrityError):
        db_session.execute(
            delete(AssistantMessage).where(AssistantMessage.id == identifier)
        )
        db_session.commit()


@pytest.mark.parametrize('foreign_keys', (False, True))
def test_new_reference_must_match_signed_dependency_actual_owner_and_effect(
    db_session, foreign_keys
):
    message, *_ = write_explicit(db_session)
    unsigned = AssistantMessage(
        conversation_id=message.conversation_id,
        role='assistant',
        content='independent',
    )
    db_session.add(unsigned)
    db_session.commit()
    db_session.connection().exec_driver_sql(
        f'PRAGMA foreign_keys={int(foreign_keys)}'
    )
    ref = db_session.scalar(select(AssistantMessageKnowledgeEvidenceRef))
    db_session.add(
        AssistantMessageKnowledgeEvidenceRef(
            dependency_id=ref.dependency_id,
            assistant_message_id=unsigned.id,
            approval_link_id=ref.approval_link_id,
            trusted_knowledge_evidence_link_id=999999,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_postgres_successor_uses_insert_registration_not_tuple_xmin(monkeypatch):
    migration = importlib.import_module(
        'backend.migrations.versions.c6f7a8b9c0d1_harden_legacy_v3_database_guards'
    )
    emitted = []
    monkeypatch.setattr(
        migration,
        'op',
        SimpleNamespace(
            get_bind=lambda: SimpleNamespace(
                dialect=SimpleNamespace(name='postgresql'), scalar=lambda sql: 0
            ),
            batch_alter_table=lambda name: nullcontext(SimpleNamespace()),
            execute=lambda sql: emitted.append(str(sql)),
            create_table=lambda *args, **kwargs: None,
            drop_table=lambda *args, **kwargs: None,
        ),
    )

    migration.upgrade()
    sql = '\n'.join(emitted)
    assert 'xmin' not in migration.PG_STAGING
    assert 'assistant_legacy_v3_parent_staging' in sql
    assert 'assistant_legacy_v3_dependency_staging' in sql
    assert 'FORCE ROW LEVEL SECURITY' in sql
    assert 'pg_trigger_depth() > 0' in sql
    assert 'REVOKE ALL' in sql and 'FROM PUBLIC' in sql
    assert sql.count('SECURITY DEFINER SET search_path FROM CURRENT') >= 6
    assert 'AFTER INSERT ON assistant_messages' in sql
    assert 'AFTER INSERT ON assistant_message_evidence_dependencies' in sql
    assert sql.count('DEFERRABLE INITIALLY DEFERRED') == 2
    assert 'parent_id=OLD.assistant_message_id' in migration.PG_STAGING
    assert 'dependency_id=OLD.id' in migration.PG_STAGING
    assert migration.PG_STAGING.count('transaction_id=txid_current()') >= 2
    assert 'DELETE FROM assistant_legacy_v3_parent_staging' in sql
    mutable = migration.PG_STAGING.split('ARRAY[', 1)[1].split('];', 1)[0]
    assert {item.strip().strip("'") for item in mutable.split(',')} == {
        'dependency_set_hmac',
        'dependency_serving_scope',
        'dependency_role',
        'dependency_child_hmac',
        'legacy_dependency_identity_hmac',
        'model_content_hmac',
        'canonical_citation_projection_hmac',
        'selected_v1_citation_projection_hmac',
        'approval_provenance_hmac',
        'evidence_link_set_hmac',
    }

    migration.downgrade()
    downgraded = '\n'.join(emitted)
    assert migration.predecessor.PG_STAGING in downgraded
    assert downgraded.index(
        'DROP TABLE assistant_legacy_v3_dependency_staging'
    ) < downgraded.index('DROP TABLE assistant_legacy_v3_parent_staging')


def test_legacy_v3_postgres_upgrade_emits_deferred_parent_child_ref_contract(
    monkeypatch,
):
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    from backend.app.db.assistant_legacy_guards import invalid_v3_sql

    migration = importlib.import_module(
        'backend.migrations.versions.b5e6f7a8b9c0_add_legacy_v3_integrity'
    )
    emitted, constraints = [], []
    batch = SimpleNamespace(
        drop_constraint=lambda *a, **kw: None,
        create_check_constraint=lambda name, sql: constraints.append((name, sql)),
    )
    monkeypatch.setattr(
        migration,
        'op',
        SimpleNamespace(
            get_bind=lambda: SimpleNamespace(
                dialect=SimpleNamespace(name='postgresql'), scalar=lambda sql: 0
            ),
            batch_alter_table=lambda name: nullcontext(batch),
            execute=lambda sql: emitted.append(str(sql)),
        ),
    )
    migration.upgrade()
    assert dict(constraints) == migration.NEW_CHECKS
    assert invalid_v3_sql('postgresql') in emitted[0]
    assert "dependency_kind IS DISTINCT FROM 'legacy_unbound'" in emitted[0]
    assert 'DEFERRABLE INITIALLY DEFERRED' in emitted[1]
    assert (
        'AFTER INSERT OR UPDATE OR DELETE ON assistant_message_knowledge_evidence_refs'
        in emitted[1]
    )
    joined = '\n'.join(emitted)
    assert 'enforce_assistant_legacy_v3_staging' in joined
    assert 'xmin::text' in joined and 'txid_current()' in joined
    assert 'genuine legacy authority required' in joined
    assert 'published legacy v3 marker is immutable' in joined
    assert 'DROP TRIGGER trg_assistant_dependency_immutable' in joined
    migration.downgrade()
    assert migration.PG_OLD in emitted


def test_postgres_successor_preserves_old_exactness_and_limits_staging_updates():
    import ast
    import importlib
    from pathlib import Path

    migration = importlib.import_module(
        'backend.migrations.versions.b5e6f7a8b9c0_add_legacy_v3_integrity'
    )
    source = Path(
        'backend/migrations/versions/7c5a2e9f4b10_add_auto_review_trust_promotion.py'
    ).read_text(encoding='utf-8')
    original = next(
        node.value.strip()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip().startswith(
            'CREATE OR REPLACE FUNCTION enforce_assistant_dependency_exactness()'
        )
    )
    assert migration.PG_EXACTNESS_OLD.strip() == original
    start = migration.PG_EXACTNESS_NEW.index(
        "          IF dep.dependency_serving_scope = 'legacy_v1_only'"
    )
    end = migration.PG_EXACTNESS_NEW.index(
        "          IF dep.dependency_kind = 'raw_chunk' THEN", start
    )
    assert (
        migration.PG_EXACTNESS_NEW[:start] + migration.PG_EXACTNESS_NEW[end:]
    ).strip() == original
    staging = migration.PG_STAGING
    assert "TG_OP='UPDATE' AND OLD.dependency_serving_scope IS NULL" in staging
    assert "NEW.dependency_serving_scope='legacy_v1_only'" in staging
    assert 'p.content_write_mode IS NULL' in staging
    assert staging.count('xmin::text=(txid_current() % 4294967296)::text') == 2
    assert '(to_jsonb(NEW)-mutable_columns)=(to_jsonb(OLD)-mutable_columns)' in staging
    mutable = staging.split('ARRAY[', 1)[1].split('];', 1)[0]
    assert {item.strip().strip("'") for item in mutable.split(',')} == {
        'dependency_set_hmac',
        'dependency_serving_scope',
        'dependency_role',
        'dependency_child_hmac',
        'legacy_dependency_identity_hmac',
        'model_content_hmac',
        'canonical_citation_projection_hmac',
        'selected_v1_citation_projection_hmac',
        'approval_provenance_hmac',
        'evidence_link_set_hmac',
    }
    assert (
        "THEN RETURN NEW; END IF;\n RAISE EXCEPTION 'C.5 audit/evidence row is append-only';"
        in staging
    )


def test_published_legacy_marker_cannot_be_stripped_to_historical_null(db_session):
    from sqlalchemy.exc import IntegrityError

    message, *_ = write_explicit(db_session)
    for field in (
        'content_write_mode',
        'content_hmac_schema_version',
        'assistant_message_content_hmac',
        'content_hmac_key_version',
        'content_hmac_key_material_verifier',
        'content_origin',
        'content_origin_hmac',
        'rag_result_hmac',
        'linked_agent_run_id',
        'dependency_set_hmac_schema_version',
        'dependency_set_hmac',
        'parent_selected_evidence_projection_hmac',
        'model_influence_set_hmac',
    ):
        setattr(message, field, None)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_published_legacy_signature_cannot_be_reissued_or_mutated_by_read(db_session):
    from backend.app.assistant.legacy_evidence import sign_legacy_message
    from backend.app.core.config import get_settings

    message, *_ = write_explicit(db_session)
    before = (
        message.assistant_message_content_hmac,
        message.dependency_set_hmac,
        message.content,
    )
    with pytest.raises(ValueError, match='new unsigned staging row'):
        sign_legacy_message(
            db=db_session,
            message=message,
            settings=get_settings(),
            actor=USERS['viewer'],
        )
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['content']
        == 'Bound answer'
    )
    assert (
        message.assistant_message_content_hmac,
        message.dependency_set_hmac,
        message.content,
    ) == before
    assert not db_session.dirty and not db_session.new and not db_session.deleted


def test_new_legacy_signature_does_not_create_d_envelope_index_or_retrieval(
    client, db_session
):
    from backend.app.core.config import get_settings
    from backend.app.models import DecisionRecord, VectorIndexState
    from backend.app.rag.indexing import build_rag_v2_index_documents
    from backend.app.rag.search_store import (
        SqlAlchemyKeywordSearchStore,
        _resolve_canonical_projection,
    )
    from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver
    from backend.tests.test_assistant_task18 import (
        test_disabled_evidence_write_is_keyed_and_still_readable,
    )
    from backend.tests.test_rag_v2_keyword_retriever import _request

    test_disabled_evidence_write_is_keyed_and_still_readable(
        client, db_session, 'legacy_knowledge'
    )
    target = db_session.scalar(select(DecisionRecord))
    settings = get_settings()
    request = _request(query='Redis')
    assert (
        TrustedServingEnvelopeResolver(
            db=db_session, settings=settings
        ).resolve_for_index('decision_record', target.id)
        is None
    )
    assert build_rag_v2_index_documents(db_session, settings=settings) == []
    assert db_session.query(VectorIndexState).count() == 0
    assert (
        SqlAlchemyKeywordSearchStore(db=db_session, settings=settings).search(request)
        == ()
    )
    # Exact production pgvector canonicalization boundary: a legacy identity in
    # a vector hit cannot be authorized even if a ranker supplies it.
    assert (
        _resolve_canonical_projection(
            db_session,
            settings,
            {'serving_document_id': f'decision_record:{target.id}'},
            request=request,
        )
        is None
    )


@pytest.mark.parametrize('provenance', ('genuine_unbound', 'legacy_human'))
def test_new_genuine_unbound_snapshot_is_signed_without_inventing_provenance(
    db_session,
    provenance,
):
    from backend.app.agents.rag_orchestrator_agent.service import (
        ServingDependencySnapshot,
    )
    from backend.app.assistant.legacy_evidence import current_legacy_evidence
    from backend.app.core.config import get_settings
    from backend.app.models import DecisionRecord, ReviewItem
    from backend.tests.assistant_evidence_helpers import ensure_fingerprint_runtime

    ensure_fingerprint_runtime(db_session, get_settings())
    target = DecisionRecord(
        title='Old decision',
        decision_summary='Known old evidence',
        source_links=['https://example.test/old'],
        source_snippets=['Known old evidence'],
        permission_level='internal',
        review_status='approved',
        confidence_score=1,
    )
    db_session.add(target)
    db_session.commit()
    if provenance == 'legacy_human':
        item = ReviewItem(
            item_type='decision',
            payload={},
            source_links=list(target.source_links),
            source_snippets=list(target.source_snippets),
            permission_level='internal',
            confidence_score=1,
            status='approved',
            resolution_source='human',
        )
        db_session.add(item)
        db_session.flush()
        target.source_review_item_id = item.id
        db_session.commit()
    identifier = f'decision_record:{target.id}'
    public, kind, text, links, snippets, permission, fingerprint, _ = (
        current_legacy_evidence(db_session, identifier)
    )
    candidate = RagEvidenceCandidate(
        source_id=public,
        source_url=links[0],
        text=text,
        source_snippet=snippets[0],
        author=None,
        timestamp=None,
        permission_level=permission,
        metadata={'source_type': kind},
    )
    snapshot = ServingDependencySnapshot(
        serving_document_id=identifier,
        dependency_kind='legacy_unbound',
        serving_content_hash=fingerprint,
        permission_level=permission,
    )
    if provenance == 'legacy_human':
        snapshot = build_serving_dependency_snapshot(db_session, candidate)
    conversation = create_conversation(db_session, USERS['viewer'])
    message = append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='Old bound answer',
        citations=[citation_from_candidate(candidate)],
        source_ids=[public],
        source_links=links,
        source_snippets=snippets,
        permission_level=permission,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
        serving_dependencies=(snapshot,),
    )
    child = db_session.scalar(select(AssistantMessageEvidenceDependency))
    assert child.approval_link_id is None
    if provenance == 'genuine_unbound':
        assert child.dependency_kind == 'legacy_unbound'
        assert child.knowledge_id is None and not child.legacy_human_base
    else:
        assert child.dependency_kind == 'trusted_knowledge' and child.legacy_human_base
        assert child.legacy_source_review_item_id == item.id
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['content']
        == 'Old bound answer'
    )
    if provenance == 'legacy_human':
        item.source_snippets = ['changed exact human evidence']
        db_session.commit()
        assert (
            serialize_message(message, db=db_session, user=USERS['viewer'])['metadata'][
                'status'
            ]
            == 'evidence_unavailable'
        )


def test_legacy_v3_frozen_golden_domains_and_utf8_vectors(db_session, monkeypatch):
    import hashlib
    import hmac
    import json

    from backend.app.assistant import legacy_integrity_v3

    captured = {}
    original = legacy_integrity_v3.keyed_fingerprint

    def fingerprint(value, *, secret, schema_version, policy_version):
        result = original(
            value,
            secret=secret,
            schema_version=schema_version,
            policy_version=policy_version,
        )
        # Independent standard-library implementation checks canonical wire bytes.
        canonical = json.dumps(
            {
                'domain': 'paraworks:keyed-fingerprint:v1',
                'schema_version': schema_version,
                'policy_version': policy_version,
                'value': value,
            },
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        ).encode('utf-8')
        assert result == hmac.new(secret, canonical, hashlib.sha256).hexdigest()
        captured[schema_version] = result
        return result

    monkeypatch.setattr(legacy_integrity_v3, 'keyed_fingerprint', fingerprint)
    write_explicit(db_session, email=True, resolution='human')
    assert captured == {
        'assistant-dependency-child-hmac:v3': '6701c60c7245e887ddd2e37ce8bb947b61176df68107443e6df0e65bfbccbaff',
        'assistant-dependency-set-hmac:v3': '6ba828ce9e08e987f5fa8e0c6f9e3e0ed79d8af37203e8741279aa8f556645d2',
        'assistant-legacy-action-payload:v1': '772792aa8c99eb640419afe802f46182ef30607751aa2db485896f634e80c96f',
        'assistant-legacy-approval-provenance:v1': 'ac0177018c4e90a1587826f80c1654324002099e5c4551d0b3e8a01a84b25345',
        'assistant-legacy-content-bytes:v1': '7c90fd5d54b130f95654adf0f254ab01e04c6c3a0cd74f76f733f573453cba39',
        'assistant-legacy-dependency-snapshot:v2': '22a48184a1c70684676c2b4192ca80202e2fc485291f80d567f9cc215941425c',
        'assistant-legacy-evidence-links:v1': '71ff2a33ec60a48eebd3c001b59df22bdc3f9ba34c81b07976b1725a2f83da36',
        'assistant-legacy-evidence-origin:v2': '55e45545d57d181e7135fad602b834f35773580f5128464f30a0df0aa7c2d891',
        'assistant-message-content-hmac:v1': '0c5b36731318de1e3fced0eb16b6c3111275cc4847de8371f99fc5ea79df47fe',
    }
    assert legacy_integrity_v3.exact_object(
        'e\u0301'
    ) != legacy_integrity_v3.exact_object('\u00e9')


@pytest.mark.parametrize('resolution', ('human', 'auto_policy'))
@pytest.mark.parametrize('email', (False, True))
def test_every_future_explicit_legacy_write_preserves_exact_signed_provenance(
    db_session, resolution, email
):
    message, _, _, _, approval = write_explicit(
        db_session, email=email, resolution=resolution
    )
    assert (
        message.dependency_set_hmac_schema_version == 'assistant-dependency-set-hmac:v3'
    )
    child = db_session.scalar(select(AssistantMessageEvidenceDependency))
    assert child.dependency_kind == 'trusted_knowledge'
    assert child.approval_link_id == approval.id
    assert child.dependency_serving_scope == 'legacy_v1_only'
    assert child.dependency_role == (
        'legacy_evidence_influence' if email else 'selected_citation'
    )
    assert (
        child.serving_identity_hmac is None
        and child.serving_version_fingerprint is None
    )
    assert (
        child.support_mode is None
        and child.approval_provenance_hmac
        and child.evidence_link_set_hmac
    )
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['content']
        == 'Bound answer'
    )


@pytest.mark.parametrize(
    'field', ('content', 'to', 'subject', 'body', 'evidence_derived')
)
def test_future_signed_email_tamper_redacts_get_context_and_send(
    client, db_session, monkeypatch, field
):
    from backend.app.api.v1 import assistant

    message, *_ = write_explicit(db_session, email=True)
    if field == 'content':
        message.content = 'forged content'
    else:
        metadata = copy.deepcopy(message.metadata_)
        if field == 'evidence_derived':
            metadata[field] = False
        else:
            metadata['email_draft'][field] = (
                ['attacker@example.com'] if field == 'to' else 'forged bytes'
            )
        message.metadata_ = metadata
    if field == 'evidence_derived':
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()
        # Prove the reader also rejects persisted corruption, independently of
        # the DB guard. Restore this exact trigger immediately after injection.
        connection = db_session.connection()
        trigger = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE name='legacy_v3_assistant_messages_update'"
        ).scalar_one()
        connection.exec_driver_sql('DROP TRIGGER legacy_v3_assistant_messages_update')
        message.metadata_ = metadata
        db_session.commit()
        db_session.connection().exec_driver_sql(trigger)
        db_session.commit()
    else:
        db_session.commit()
    sent = []

    class Sender:
        def __init__(self, **kwargs):
            pass

        def send(self, **kwargs):
            sent.append(kwargs)
            raise AssertionError('unverified draft must never send')

    monkeypatch.setattr(assistant, 'GmailDraftSender', Sender)
    view = serialize_message(message, db=db_session, user=USERS['viewer'])
    assert view['metadata']['status'] == 'evidence_unavailable'
    assert eligible_context_messages(db_session, USERS['viewer'], [message]) == []
    fetched = client.get(
        f'/api/v1/assistant/conversations/{message.conversation_id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )
    assert fetched.json()['messages'][0]['metadata']['status'] == 'evidence_unavailable'
    response = client.post(
        f'/api/v1/assistant/messages/{message.id}/email/send',
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 409 and sent == []


def test_future_answer_content_tamper_is_not_historical_compatibility(db_session):
    message, *_ = write_explicit(db_session)
    message.content = 'forged stored answer'
    db_session.commit()
    assert (
        serialize_message(message, db=db_session, user=USERS['viewer'])['content']
        != 'forged stored answer'
    )


@pytest.mark.parametrize('shape', ('citations', 'metadata_flag'))
def test_new_evidence_shaped_write_without_dependency_is_rejected(db_session, shape):
    conversation = create_conversation(db_session, USERS['viewer'])
    with pytest.raises(ValueError, match='dependenc'):
        append_assistant_message(
            db_session,
            USERS['viewer'],
            conversation,
            content='unsigned evidence',
            citations=[{'source_id': 'unsafe'}] if shape == 'citations' else [],
            source_ids=['unsafe'] if shape == 'citations' else [],
            source_links=['https://example.test/source']
            if shape == 'citations'
            else [],
            source_snippets=['evidence'] if shape == 'citations' else [],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={'evidence_derived': True} if shape == 'metadata_flag' else {},
        )
    assert db_session.query(AssistantMessage).count() == 0


def test_signing_failure_rolls_back_entire_new_legacy_product(db_session, monkeypatch):
    from backend.app.assistant import legacy_evidence

    def fail(**kwargs):
        raise ValueError('signature unavailable')

    monkeypatch.setattr(legacy_evidence, 'sign_legacy_message', fail)
    with pytest.raises(ValueError, match='signature unavailable'):
        write_explicit(db_session)
    assert db_session.query(AssistantMessage).count() == 0


@pytest.mark.parametrize(
    'mutation',
    ('count', 'child_delete', 'ref_delete', 'parent_hex', 'child_hex', 'v2_marker'),
)
def test_published_v3_structure_is_database_enforced(db_session, mutation):
    from sqlalchemy.exc import IntegrityError

    from backend.app.models import AssistantMessageKnowledgeEvidenceRef

    message, *_ = write_explicit(db_session)
    if mutation == 'count':
        message.serving_dependency_count += 1
    elif mutation == 'parent_hex':
        message.content_origin_hmac = 'F' * 64
    elif mutation == 'child_hex':
        db_session.scalar(
            select(AssistantMessageEvidenceDependency)
        ).approval_provenance_hmac = 'F' * 64
    elif mutation == 'v2_marker':
        message.dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2'
    elif mutation == 'child_delete':
        db_session.delete(db_session.scalar(select(AssistantMessageEvidenceDependency)))
    else:
        db_session.delete(
            db_session.scalar(select(AssistantMessageKnowledgeEvidenceRef))
        )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_legacy_v3_migration_upgrades_and_refuses_lossy_downgrade(sqlite_migration):  # noqa: F811
    from alembic import command
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    config, database_url = sqlite_migration
    command.upgrade(config, 'a4d5e6f7b8c9')
    engine = create_engine(database_url)
    with Session(engine) as db:
        conversation = create_conversation(db, USERS['viewer'])
        db.add(
            AssistantMessage(
                conversation_id=conversation.id, role='assistant', content='historical'
            )
        )
        db.commit()
    command.upgrade(config, 'head')
    with engine.connect() as db:
        assert (
            db.scalar(text('SELECT version_num FROM alembic_version')) == 'd7a8b9c0d1e2'
        )
        assert (
            db.scalar(text('SELECT content_write_mode FROM assistant_messages')) is None
        )
    command.downgrade(config, 'a4d5e6f7b8c9')
    command.upgrade(config, 'head')
    with Session(engine) as db:
        write_explicit(db)
    with pytest.raises(ValueError, match='v3'):
        command.downgrade(config, 'a4d5e6f7b8c9')
    with engine.connect() as db:
        assert (
            db.scalar(text('SELECT version_num FROM alembic_version')) == 'd7a8b9c0d1e2'
        )
