from __future__ import annotations

import os
from threading import Event, Thread
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.models import (
    AutoReviewRuntimeKeyState,
    RagServingCorpusGeneration,
)
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    ServingProjectionReadCoordinator,
    ServingProjectionTailLockedContext,
    build_serving_lock_plan,
)
from backend.tests.postgres_isolation import lease_postgres_schema
from backend.tests.test_auto_review_source_reconciliation import (
    _seed_explicit_history,
)


def _settings(database_url: str = 'sqlite://') -> Settings:
    return Settings(
        _env_file=None,
        database_url=database_url,
    )


def _seed_lock_prefix(db: Session, settings: Settings) -> None:
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db.add_all(
        [
            AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                fingerprint_key_version=(
                    settings.agent_runtime_fingerprint_key_version
                ),
                fingerprint_key_material_verifier=verifier,
                generation=1,
                ready=True,
            ),
            RagServingCorpusGeneration(
                id=1,
                corpus_generation=3,
                vector_index_generation=5,
                embedding_model=settings.openai_embedding_model,
                embedding_dimensions=settings.openai_embedding_dimensions,
                index_policy_version='rag-v2-serving-index:v1',
                pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
                fingerprint_key_version=(
                    settings.agent_runtime_fingerprint_key_version
                ),
                fingerprint_key_material_verifier=verifier,
            ),
        ]
    )
    db.commit()


def test_projection_reader_mints_exact_canonical_tail_after_prefix(
    db_session: Session,
) -> None:
    settings = _settings()
    history, item, source, link = _seed_explicit_history(
        db_session,
        resolution_source='human',
    )
    runtime = db_session.query(AutoReviewRuntimeKeyState).one()
    runtime.fingerprint_key_version = (
        settings.agent_runtime_fingerprint_key_version
    )
    runtime.fingerprint_key_material_verifier = (
        fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        )
    )
    db_session.add(
        RagServingCorpusGeneration(
            id=1,
            corpus_generation=3,
            vector_index_generation=5,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
            index_policy_version='rag-v2-serving-index:v1',
            pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
        )
    )
    db_session.commit()
    history_id = history.id
    item_id = item.id
    source_id = source.id
    link_id = link.id
    document_id = f'history_event:{history_id}'
    db_session.rollback()
    coordinator = ServingProjectionReadCoordinator(
        db=db_session,
        settings=settings,
    )

    with db_session.begin(), coordinator.acquire() as generations:
        locked = coordinator.lock_canonical_tail([document_id])
        coordinator.validate_tail_context(locked)

    assert generations == (3, 5)
    assert type(locked) is ServingProjectionTailLockedContext
    assert locked.plan.document_ids == (document_id,)
    assert locked.plan.source_ids == (source_id,)
    assert locked.plan.review_item_ids == (item_id,)
    assert locked.plan.approval_link_ids == (link_id,)
    assert locked.plan.targets == (('history_event', history_id),)
    with pytest.raises(TypeError, match='transaction is no longer active'):
        coordinator.validate_tail_context(locked)


def test_projection_reader_refuses_tail_before_prefix_and_second_tail(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_lock_prefix(db_session, settings)
    coordinator = ServingProjectionReadCoordinator(
        db=db_session,
        settings=settings,
    )

    with db_session.begin():
        with pytest.raises(TypeError, match='projection prefix'):
            coordinator.lock_canonical_tail(())
        with coordinator.acquire():
            coordinator.lock_canonical_tail(())
            with pytest.raises(TypeError, match='already locked'):
                coordinator.lock_canonical_tail(())


@pytest.mark.skipif(
    not os.environ.get('PARAWORKS_TEST_POSTGRES_URL'),
    reason='disposable PostgreSQL C.5 interleaving database is not configured',
)
def test_postgres_finalization_prefix_and_tail_serialize_c5_mutation() -> None:
    base_url = os.environ['PARAWORKS_TEST_POSTGRES_URL']
    with lease_postgres_schema(
        base_url,
        run_id=uuid4().hex[:12],
        scope_name='d13_tail',
    ) as lease:
        engine = create_engine(lease.database_url)
        # The lease has a local search_path while public may already be at head.
        # Materialize this test's tables in the leased schema, not public.
        Base.metadata.create_all(engine, checkfirst=False)
        settings = _settings(lease.database_url)
        session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        with session_local() as setup:
            _seed_lock_prefix(setup, settings)

        finalizer_holds_tail = Event()
        mutation_acquired = Event()
        worker_failed: list[BaseException] = []

        def mutate() -> None:
            try:
                with session_local() as db, db.begin():
                    plan = build_serving_lock_plan(db, ['history_event:999999'])
                    with KeyedMutationGuard.generation_barrier(db):
                        key_context = lock_runtime_state(db, mode='share')
                        ServingMutationLockCoordinator(
                            db=db,
                            settings=settings,
                        ).acquire(key_context=key_context, plan=plan)
                        mutation_acquired.set()
            except BaseException as exc:  # pragma: no cover - surfaced below
                worker_failed.append(exc)
                mutation_acquired.set()

        try:
            with session_local() as db, db.begin():
                coordinator = ServingProjectionReadCoordinator(
                    db=db,
                    settings=settings,
                )
                with coordinator.acquire():
                    locked = coordinator.lock_canonical_tail(
                        ['history_event:999999']
                    )
                    coordinator.validate_tail_context(locked)
                    finalizer_holds_tail.set()
                    worker = Thread(target=mutate, daemon=True)
                    worker.start()
                    assert finalizer_holds_tail.is_set()
                    assert mutation_acquired.wait(0.25) is False
            worker.join(timeout=5)
            assert worker.is_alive() is False
            assert worker_failed == []
            assert mutation_acquired.is_set()
        finally:
            engine.dispose()
