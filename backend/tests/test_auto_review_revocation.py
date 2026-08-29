from datetime import UTC, datetime

import pytest
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
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewRevocationAssessment,
    AutoReviewRuntimeKeyState,
    HistoryEvent,
    ReviewItem,
    Source,
    TimelineEvent,
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


def test_consumed_key_context_refuses_second_document_lock_acquisition(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        serving_key_context = manager.bind_transaction(key_context)
        reindex_context = manager.acquire_documents(
            serving_key_context, ['history_event:7', 'history_event:7']
        )
        with pytest.raises(TypeError, match='already consumed'):
            manager.acquire_documents(
                serving_key_context, ['history_event:7']
            )

    assert reindex_context.session_identity == id(db_session)
    assert reindex_context.document_ids == ('history_event:7',)


def test_shared_key_context_refuses_second_transaction_binding(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        first = manager.bind_transaction(key_context)

        with pytest.raises(TypeError, match='already bound'):
            manager.bind_transaction(key_context)

    assert first.session_identity == id(db_session)


def test_shared_key_context_cannot_rebind_after_its_root_transaction_ends(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        manager.bind_transaction(key_context)
        original_transaction = db_session.get_transaction()
        assert original_transaction is not None
        db_session.commit()

        assert db_session.scalar(
            select(AutoReviewRuntimeKeyState.generation)
        ) == 1
        assert db_session.get_transaction() is not original_transaction
        with pytest.raises(TypeError, match='already bound'):
            manager.bind_transaction(key_context)
        db_session.rollback()

    with KeyedMutationGuard.generation_barrier(db_session):
        fresh_key_context = lock_runtime_state(db_session)
        fresh_bound = manager.bind_transaction(fresh_key_context)

    assert fresh_bound.session_identity == id(db_session)


def test_key_context_cannot_mint_document_locks_after_its_transaction_commits(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        serving_key_context = manager.bind_transaction(key_context)
        db_session.commit()

        with pytest.raises(TypeError, match='transaction is no longer active'):
            manager.acquire_documents(
                serving_key_context, ['history_event:7']
            )


def test_global_serving_coordinator_locks_exact_source_and_provenance_plan(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import (
        ServingMutationLockCoordinator,
        build_serving_lock_plan,
    )

    settings = Settings(database_url='sqlite://')
    history, item, source, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    document_id = f'history_event:{history.id}'
    plan = build_serving_lock_plan(db_session, [document_id])
    _seed_revoke_runtime(db_session, settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        locked = ServingMutationLockCoordinator(
            db=db_session, settings=settings
        ).acquire(key_context=key_context, plan=plan)

    assert plan.document_ids == (document_id,)
    assert plan.source_ids == (source.id,)
    assert plan.review_item_ids == (item.id,)
    assert plan.approval_link_ids == (link.id,)
    assert plan.targets == (('history_event', history.id),)
    assert locked.document_ids == (document_id,)


def test_nested_vector_mutation_validates_already_held_context_without_reacquiring_runtime(
    db_session: Session,
) -> None:
    from backend.app.rag.serving_locks import VectorServingLockManager

    settings = Settings(database_url='sqlite://')
    _seed_runtime(db_session, settings)
    manager = VectorServingLockManager(db=db_session, settings=settings)

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        locked = manager.acquire_documents(
            manager.bind_transaction(key_context), ['todo:9']
        )
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
        locked = manager.acquire_documents(
            manager.bind_transaction(key_context), ['todo:9']
        )

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
            manager.bind_transaction(key_context)
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


def test_in_memory_composed_mutations_observe_transaction_shadow(
    db_session: Session,
) -> None:
    store = InMemoryVectorStore(session=db_session)
    deleted = VectorDocument(
        document_id='history_event:31',
        text='Composed delete evidence',
        source_url='knowledge://history_event:31',
        source_snippet='Composed delete evidence',
        permission_level='public',
        metadata={},
    )
    narrowed = VectorDocument(
        document_id='history_event:32',
        text='Composed narrow evidence',
        source_url='knowledge://history_event:32',
        source_snippet='Composed narrow evidence',
        permission_level='public',
        metadata={},
    )
    store.upsert_many([deleted, narrowed])

    assert store.delete_many([deleted.document_id]) == 1
    assert store.narrow_permissions([deleted.document_id], 'restricted') == 0
    assert store.narrow_permissions([narrowed.document_id], 'restricted') == 1
    with pytest.raises(ValueError, match='broadening'):
        store.narrow_permissions([narrowed.document_id], 'internal')

    db_session.commit()
    exported = {row['document_id']: row for row in store.export_documents()}
    assert deleted.document_id not in exported
    assert exported[narrowed.document_id]['permission_level'] == 'restricted'


def test_in_memory_nested_rollback_discards_only_savepoint_mutations(
    db_session: Session,
) -> None:
    store = InMemoryVectorStore(session=db_session)
    kept = VectorDocument(
        document_id='history_event:33',
        text='Nested rollback evidence',
        source_url='knowledge://history_event:33',
        source_snippet='Nested rollback evidence',
        permission_level='public',
        metadata={},
    )
    root_deleted = VectorDocument(
        document_id='history_event:34',
        text='Root delete evidence',
        source_url='knowledge://history_event:34',
        source_snippet='Root delete evidence',
        permission_level='public',
        metadata={},
    )
    store.upsert_many([kept, root_deleted])
    assert store.delete_many([root_deleted.document_id]) == 1

    savepoint = db_session.begin_nested()
    assert store.narrow_permissions([kept.document_id], 'restricted') == 1
    savepoint.rollback()
    db_session.commit()

    exported = {row['document_id']: row for row in store.export_documents()}
    assert root_deleted.document_id not in exported
    assert exported[kept.document_id]['permission_level'] == 'public'


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


def test_broken_active_row_is_not_surviving_provenance(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_revoke import AutoReviewRevokeService

    settings = Settings(database_url='sqlite://')
    history, item, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    invalid_source = Source(
        source_type='gmail',
        source_id='gmail:invalid-other-effect',
        source_url='https://gmail.mock/invalid-other-effect',
        title='Invalid other evidence',
        permission_level='internal',
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='f' * 64,
    )
    other_item = ReviewItem(
        item_type='history_event',
        payload={'title': history.title},
        source_links=[invalid_source.source_url],
        source_snippets=['invalid other evidence'],
        confidence_score=1.0,
        permission_level='internal',
        status='approved',
        candidate_contract_version='c5-v1',
        resolution_source='human',
    )
    db_session.add_all([invalid_source, other_item])
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
        claim_fingerprint='9' * 64,
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
            canonical_source_id=str(invalid_source.id),
            canonical_version_or_signature='e' * 64,
            evidence_hash='8' * 64,
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

    assert result.knowledge_remains_trusted is False
    assert result.revoked_document_count == 1
    assert history.review_status == 'revoked'


@pytest.mark.parametrize('surviving_target', ['primary', 'companion'])
def test_primary_and_companion_recompute_provenance_independently(
    db_session: Session,
    surviving_target: str,
) -> None:
    from backend.app.review.auto_review_revoke import AutoReviewRevokeService

    settings = Settings(database_url='sqlite://')
    primary, item, source, selected_primary = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    companion = TimelineEvent(
        project_key='project-companion',
        title='Companion history',
        result_summary='Companion exact evidence',
        source_links=[source.source_url],
        source_snippets=['Companion exact evidence'],
        confidence_score=0.99,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=item.id,
    )
    db_session.add(companion)
    db_session.flush()
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    selected_companion = TrustedKnowledgeApprovalLink(
        knowledge_type='timeline_event',
        knowledge_id=companion.id,
        review_item_id=item.id,
        security_scope_id='workspace-a',
        promotion_effect_kind='companion',
        resolution_source='auto_policy',
        claim_fingerprint='7' * 64,
        permission_level='internal',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier=verifier,
        active=True,
    )
    other_item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Independent reaffirmation'},
        source_links=[source.source_url],
        source_snippets=['Exact current evidence'],
        confidence_score=1.0,
        permission_level='internal',
        status='approved',
        candidate_contract_version='c5-v1',
        resolution_source='human',
    )
    db_session.add_all([selected_companion, other_item])
    db_session.flush()
    db_session.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=selected_companion.id,
            canonical_source_kind='gmail',
            canonical_source_id=str(source.id),
            canonical_version_or_signature='a' * 64,
            evidence_hash='6' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier=verifier,
        )
    )
    survivor = primary if surviving_target == 'primary' else companion
    other_link = TrustedKnowledgeApprovalLink(
        knowledge_type=(
            'history_event' if surviving_target == 'primary' else 'timeline_event'
        ),
        knowledge_id=survivor.id,
        review_item_id=other_item.id,
        security_scope_id='workspace-a',
        promotion_effect_kind='primary',
        resolution_source='human',
        claim_fingerprint='5' * 64,
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
            evidence_hash='4' * 64,
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

    db_session.refresh(primary)
    db_session.refresh(companion)
    assert result.knowledge_remains_trusted is True
    assert result.revoked_document_count == 1
    assert (primary.review_status, companion.review_status) == (
        ('approved', 'revoked')
        if surviving_target == 'primary'
        else ('revoked', 'approved')
    )
    assert selected_primary.active is False
    assert selected_companion.active is False


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


def test_revoke_rolls_back_all_relational_changes_when_vector_delete_fails(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_revoke import AutoReviewRevokeService

    class FailingWriter:
        def delete_many(self, document_ids):
            raise RuntimeError('bounded vector failure')

    settings = Settings(database_url='sqlite://')
    history, item, _, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    _seed_revoke_runtime(db_session, settings)

    with pytest.raises(RuntimeError, match='bounded vector failure'):
        AutoReviewRevokeService(
            db_session,
            settings=settings,
            vector_writer=FailingWriter(),
        ).revoke(
            review_item_id=item.id,
            actor=human_review_actor(USERS['admin']),
            reason_code='business_withdrawal',
        )

    db_session.expire_all()
    assert db_session.get(ReviewItem, item.id).status == 'approved'
    assert db_session.get(TrustedKnowledgeApprovalLink, link.id).active is True
    assert db_session.get(HistoryEvent, history.id).review_status == 'approved'
    assert db_session.query(VectorServingTombstone).count() == 0
    assert db_session.query(AutoReviewRevocationAssessment).count() == 0


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


def test_later_critical_audit_correction_blocks_confirmed_business_revoke(
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
    audit = AutoReviewPostAudit(
        review_item_id=item.id,
        promotion_decision_id=1,
        sample_cohort='manual',
        status='completed',
        outcome='confirmed',
        auditor_subject_hmac='1' * 64,
        auditor_fingerprint_key_version='v1',
        auditor_fingerprint_key_material_verifier='2' * 64,
        audit_reason='Initially confirmed',
        audited_at=datetime.now(UTC),
    )
    assessment = AutoReviewRevocationAssessment(
        review_item_id=item.id,
        reason_code='business_withdrawal',
        actor_subject_hmac='3' * 64,
        actor_fingerprint_key_version='v1',
        actor_fingerprint_key_material_verifier='4' * 64,
    )
    db_session.add_all([audit, assessment])
    db_session.flush()
    db_session.add(
        AutoReviewAuditCorrection(
            post_audit_id=audit.id,
            review_item_id=item.id,
            assessment_id=assessment.id,
            effective_outcome='incorrect',
            status='remediation_required',
            system_resolution_code='revoke_pending',
            actor_subject_hmac='5' * 64,
            actor_fingerprint_key_version='v1',
            actor_fingerprint_key_material_verifier='6' * 64,
        )
    )
    db_session.commit()
    _seed_revoke_runtime(db_session, settings)

    with pytest.raises(AutoReviewRevokeRefused, match='audit_required'):
        AutoReviewRevokeService(db_session, settings=settings).revoke(
            review_item_id=item.id,
            actor=human_review_actor(USERS['admin']),
            reason_code='business_withdrawal',
        )

    assert item.status == 'approved'
