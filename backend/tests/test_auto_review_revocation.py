from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.db.base import Base
from backend.app.models import (
    AutoReviewPostAudit,
    AutoReviewRuntimeKeyState,
    HistoryEvent,
    ReviewItem,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.vector_store import InMemoryVectorStore, VectorDocument
from backend.app.review.actors import human_review_actor
from backend.tests.test_auto_review_source_reconciliation import (
    _seed_explicit_history,
)


def _seed_runtime(db: Session, settings: Settings, *, generation: int = 1) -> None:
    db.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
            generation=generation,
            ready=True,
        )
    )
    db.commit()


def _other_session() -> Session:
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_reindex_and_revoke_use_the_same_document_advisory_key_and_session(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        reindex_context = manager.acquire_documents(
            key_context, ['history_event:7', 'history_event:7']
        )
        revoke_context = manager.acquire_documents(
            key_context, ['history_event:7']
        )

    assert reindex_context.session_identity == id(db_session)
    assert revoke_context.session_identity == id(db_session)
    assert reindex_context.document_ids == ('history_event:7',)
    assert reindex_context.advisory_keys == revoke_context.advisory_keys


def test_nested_vector_mutation_validates_already_held_context_without_reacquiring_runtime(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        locked = manager.acquire_documents(key_context, ['todo:9'])
        manager.validate_locked_context(locked, ['todo:9'])


def test_wrong_session_generation_or_document_set_rejects_vector_locked_context(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        locked = manager.acquire_documents(key_context, ['todo:9'])

        try:
            manager.validate_locked_context(locked, ['todo:10'])
        except TypeError as exc:
            assert str(exc) == 'Vector-serving lock context does not cover every document'
        else:
            raise AssertionError('an incomplete document lock context must fail')

    other = _other_session()
    try:
        other_manager = VectorServingLockManager(db=other, settings=settings)
        try:
            other_manager.validate_locked_context(locked, ['todo:9'])
        except TypeError as exc:
            assert str(exc) == 'Vector-serving lock context belongs to another session'
        else:
            raise AssertionError('a wrong-session lock context must fail')
    finally:
        other.close()

    runtime = db_session.scalar(select(AutoReviewRuntimeKeyState))
    runtime.generation += 1
    db_session.commit()
    try:
        manager.validate_locked_context(locked, ['todo:9'])
    except TypeError as exc:
        assert str(exc) == 'Vector-serving lock context generation is stale'
    else:
        raise AssertionError('a stale-generation lock context must fail')


def test_serving_mutation_refuses_same_version_with_different_key_material(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    current_settings = Settings(database_url='sqlite://')
    mismatched_settings = Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret='another-development-secret',
        agent_runtime_fingerprint_key_version=(
            current_settings.agent_runtime_fingerprint_key_version
        ),
    )
    _seed_runtime(db_session, current_settings)
    manager = VectorServingLockManager(db=db_session, settings=mismatched_settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        try:
            manager.acquire_documents(key_context, ['todo:9'])
        except TypeError as exc:
            assert str(exc) == 'Configured vector-serving key material is stale'
        else:
            raise AssertionError('different key material under one version must fail')


def test_in_memory_delete_runs_after_commit_and_is_discarded_on_rollback(
    db_session: Session,
) -> None:
    store = InMemoryVectorStore(session=db_session)
    document = VectorDocument(
        document_id='history_event:3',
        text='Rollback-safe revocation evidence',
        source_url='knowledge://history_event:3',
        source_snippet='Rollback-safe revocation evidence',
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    store.upsert(document)

    assert store.delete_many([document.document_id]) == 1
    assert store.search(query='revocation evidence', user=USERS['viewer']).matches
    db_session.rollback()
    assert store.search(query='revocation evidence', user=USERS['viewer']).matches

    assert store.delete_many([document.document_id]) == 1
    db_session.commit()
    assert store.search(query='revocation evidence', user=USERS['viewer']).matches == []


def _seed_revoke_runtime(db: Session, settings: Settings) -> None:
    if db.query(AutoReviewRuntimeKeyState).count() == 0:
        _seed_runtime(db, settings)


def test_human_or_incomplete_auto_item_cannot_be_revoked(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_revoke import (
        AutoReviewRevokeRefused,
        AutoReviewRevokeService,
    )

    settings = Settings(database_url='sqlite://')
    human_history, human_item, _, _ = _seed_explicit_history(
        db_session, resolution_source='human'
    )
    del human_history
    _seed_revoke_runtime(db_session, settings)
    service = AutoReviewRevokeService(db_session, settings=settings)
    actor = human_review_actor(USERS['admin'])

    for review_item_id in (human_item.id,):
        try:
            service.revoke(
                review_item_id=review_item_id,
                actor=actor,
                reason_code='business_withdrawal',
            )
        except AutoReviewRevokeRefused as exc:
            assert exc.code == 'unsupported_transition'
        else:
            raise AssertionError('human approval must not be auto-revocable')

    auto_history, auto_item, _, _ = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='b' * 64,
        evidence_signature='b' * 64,
    )
    del auto_history
    auto_item.auto_validation_id = None
    db_session.commit()
    try:
        service.revoke(
            review_item_id=auto_item.id,
            actor=actor,
            reason_code='business_withdrawal',
        )
    except AutoReviewRevokeRefused as exc:
        assert exc.code == 'incomplete_auto_approval'
    else:
        raise AssertionError('incomplete auto approval must not be revocable')


def test_revoke_marks_only_the_selected_reaffirmation_link(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_revoke import AutoReviewRevokeService

    settings = Settings(database_url='sqlite://')
    history, item, source, selected = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    other_item = ReviewItem(
        item_type='history_event',
        payload={'title': history.title},
        source_links=history.source_links,
        source_snippets=history.source_snippets,
        confidence_score=1.0,
        permission_level='internal',
        status='approved',
        candidate_contract_version='c5-v1',
        resolution_source='human',
    )
    db_session.add(other_item)
    db_session.flush()
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    other_link = TrustedKnowledgeApprovalLink(
        knowledge_type='history_event',
        knowledge_id=history.id,
        review_item_id=other_item.id,
        security_scope_id='workspace-a',
        promotion_effect_kind='primary',
        resolution_source='human',
        claim_fingerprint='d' * 64,
        permission_level='internal',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier=verifier,
        active=True,
    )
    db_session.add(other_link)
    db_session.flush()
    db_session.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=other_link.id,
            canonical_source_kind='gmail',
            canonical_source_id=str(source.id),
            canonical_version_or_signature='a' * 64,
            evidence_hash='1' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier=verifier,
        )
    )
    db_session.commit()
    _seed_revoke_runtime(db_session, settings)

    result = AutoReviewRevokeService(db_session, settings=settings).revoke(
        review_item_id=item.id,
        actor=human_review_actor(USERS['admin']),
        reason_code='business_withdrawal',
    )

    assert result.knowledge_remains_trusted is True
    assert result.revoked_document_count == 0
    assert db_session.get(TrustedKnowledgeApprovalLink, selected.id).active is False
    assert db_session.get(TrustedKnowledgeApprovalLink, other_link.id).active is True
    assert db_session.get(HistoryEvent, history.id).review_status == 'approved'


def test_last_provenance_revokes_primary_companion_and_exact_documents(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_revoke import AutoReviewRevokeService

    settings = Settings(database_url='sqlite://')
    history, item, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    document_id = f'history_event:{history.id}'
    db_session.add(
        VectorIndexState(
            document_id=document_id,
            embedding_model='deterministic-hash:v1',
            embedding_dimensions=2,
            content_hash='c' * 64,
            status='indexed',
        )
    )
    db_session.commit()
    _seed_revoke_runtime(db_session, settings)
    store = InMemoryVectorStore(session=db_session)
    store.upsert(
        VectorDocument(
            document_id=document_id,
            text=history.reason,
            source_url=history.source_links[0],
            source_snippet=history.source_snippets[0],
            permission_level='internal',
            metadata={'source_type': 'history_event'},
        )
    )

    result = AutoReviewRevokeService(
        db_session, settings=settings, vector_writer=store
    ).revoke(
        review_item_id=item.id,
        actor=human_review_actor(USERS['admin']),
        reason_code='business_withdrawal',
    )

    assert result.status == 'revoked'
    assert result.knowledge_remains_trusted is False
    assert result.revoked_document_count == 1
    assert db_session.get(HistoryEvent, history.id).review_status == 'revoked'
    assert db_session.query(VectorServingTombstone).one().document_id == document_id
    assert db_session.query(VectorIndexState).count() == 0
    assert store.search(query='Exact current evidence', user=USERS['admin']).matches == []


def test_revoke_replay_returns_the_same_bounded_result(db_session: Session) -> None:
    from backend.app.review.auto_review_revoke import AutoReviewRevokeService

    settings = Settings(database_url='sqlite://')
    _, item, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    _seed_revoke_runtime(db_session, settings)
    service = AutoReviewRevokeService(db_session, settings=settings)
    actor = human_review_actor(USERS['admin'])
    first = service.revoke(
        review_item_id=item.id,
        actor=actor,
        reason_code='business_withdrawal',
    )

    replay = service.revoke(
        review_item_id=item.id,
        actor=actor,
        reason_code='business_withdrawal',
    )

    assert replay.replayed is True
    assert replay.knowledge_remains_trusted == first.knowledge_remains_trusted
    assert replay.revoked_document_count == first.revoked_document_count


def test_pending_remediation_or_critical_audit_blocks_normal_direct_revoke(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_revoke import (
        AutoReviewRevokeRefused,
        AutoReviewRevokeService,
    )

    settings = Settings(database_url='sqlite://')
    _, item, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    db_session.add(
        AutoReviewPostAudit(
            review_item_id=item.id,
            promotion_decision_id=1,
            sample_cohort='first_50',
            status='pending',
            outcome=None,
        )
    )
    db_session.commit()
    _seed_revoke_runtime(db_session, settings)

    try:
        AutoReviewRevokeService(db_session, settings=settings).revoke(
            review_item_id=item.id,
            actor=human_review_actor(USERS['admin']),
            reason_code='business_withdrawal',
        )
    except AutoReviewRevokeRefused as exc:
        assert exc.code == 'audit_required'
    else:
        raise AssertionError('pending audit must block direct revoke')

    assert db_session.get(ReviewItem, item.id).status == 'approved'
