from __future__ import annotations

import inspect
import os
import sqlite3
from decimal import Decimal
from multiprocessing import get_context
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_finalization import (
    AssistantProjectionTarget,
    PreparedRagFinalization,
)
from backend.app.agent_runtime.rag_sqlite_smoke import (
    _PROCESS_LOCKS,
    SQLiteRagSmokeCoordinator,
    SQLiteRagSmokeUnavailable,
    sqlite_rag_smoke_mutex,
)
from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ValidatedAnswerBlock,
    ValidatedAnswerBlocks,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    prepare_assistant_request_text,
    prepare_direct_request_text,
)
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.models import (
    AgentRun,
    AgentRunCostComponent,
    AssistantConversation,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    RagLexicalServingProjection,
)
from backend.app.rag.evidence_projection import CanonicalEvidenceProjector
from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
from backend.app.rag.retrieval import RetrievalRequest, rank_evidence_slots
from backend.app.rag.search_store import SqlAlchemyKeywordSearchStore
from backend.tests.test_rag_v2_keyword_retriever import (
    _scope,
    _seed_sqlite_raw_projection,
    _seed_sqlite_trusted_projection,
    _settings,
)


def _engine(path: Path):
    return create_engine(f'sqlite+pysqlite:///{path.as_posix()}')


def _prepared_search(engine, *, settings: Settings) -> PreparedRagFinalization:
    scope = _scope(allowed_permission_levels=('public', 'internal'))
    with Session(engine) as db:
        text = prepare_direct_request_text('%_Case', key=b'k')
        request = RetrievalRequest(
            retrieval_query_text=text.retrieval_query_text,
            security_scope=scope,
            security_scope_fingerprint=security_scope_fingerprint(
                scope, settings=settings
            ),
            query_embedding_result=None,
            candidate_scan_limit=50,
            visible_limit=5,
            relevance_policy_version='rag-retrieval-policy:v2.0',
        )
        retrieval = KeywordEvidenceRetriever(
            store=SqlAlchemyKeywordSearchStore(db=db, settings=settings),
            settings=settings,
        ).invoke(request)
        return PreparedRagFinalization(
            product_kind='search',
            tentative_outcome='search_projected',
            prepared_text=text,
            security_scope=scope,
            query_embedding_result=None,
            retrieval_result=retrieval,
            evidence_slots=rank_evidence_slots(retrieval.visible),
            model_influence_observations=(),
            selected_slot_ids=(),
            validated_answer=None,
            canned_message_identity=None,
        )


def _second_process_lock_attempt(database_path: str, queue) -> None:
    engine = _engine(Path(database_path))
    try:
        SQLiteRagSmokeCoordinator(
            engine=engine,
            database_path=database_path,
            settings=_settings(),
        )
    except SQLiteRagSmokeUnavailable:
        queue.put('refused')
    else:
        queue.put('acquired')
    finally:
        engine.dispose()


def test_sqlite_smoke_uses_one_never_replaced_reentrant_mutex() -> None:
    assert sqlite_rag_smoke_mutex() is sqlite_rag_smoke_mutex()


def test_sqlite_coordinator_owns_concrete_keyword_finalization_without_callback(
    tmp_path: Path,
) -> None:
    signature = inspect.signature(SQLiteRagSmokeCoordinator)
    assert 'keyword_operation' not in signature.parameters
    assert 'automated_test' not in signature.parameters
    assert not hasattr(SQLiteRagSmokeCoordinator, 'run_atomic')

    db_path = (tmp_path / 'owned-smoke.db').absolute()
    engine = _engine(db_path)
    Base.metadata.create_all(engine)
    settings = _settings()
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
    prepared = _prepared_search(engine, settings=settings)

    coordinator = SQLiteRagSmokeCoordinator(
        engine=engine,
        database_path=db_path,
        settings=settings,
    )
    projection = coordinator.run_keyword(prepared)

    assert projection.outcome == 'search_projected'
    assert len(projection.evidence.search_results) == 1
    with Session(engine) as db:
        parent = db.scalar(select(AgentRun).where(AgentRun.run_contract_version == 'rag-run:v2'))
        assert parent is not None
        assert (parent.status, parent.run_record_phase) == ('complete', 'final')
        assert parent.projection_owner_fence_hmac is None
        children = tuple(
            db.scalars(
                select(AgentRunCostComponent)
                .where(AgentRunCostComponent.agent_run_id == parent.id)
                .order_by(AgentRunCostComponent.component_ordinal)
            )
        )
        assert tuple((row.component, row.dispatch_state) for row in children) == (
            ('query_embedding', 'terminal'),
            ('answer_generation', 'terminal'),
        )
        assert all(not row.attempted and row.dispatch_count == 0 for row in children)


def test_full_terminal_zero_invariant_rejects_omitted_field_mutation(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'invariant.db').absolute()
    engine = _engine(db_path)
    Base.metadata.create_all(engine)
    settings = _settings()
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
    projection = SQLiteRagSmokeCoordinator(
        engine=engine, database_path=db_path, settings=settings
    ).run_keyword(_prepared_search(engine, settings=settings))
    with Session(engine) as db:
        parent = db.scalar(select(AgentRun))
        assert parent is not None
        children = tuple(
            db.scalars(
                select(AgentRunCostComponent)
                .where(AgentRunCostComponent.agent_run_id == parent.id)
                .order_by(AgentRunCostComponent.component_ordinal)
            )
        )
        db.expunge_all()
    children[0].reserved_input_tokens = 1

    class InvariantSession:
        def get(self, model, identity):
            del model, identity
            return parent

        def scalars(self, statement):
            del statement
            return children

        def scalar(self, statement):
            del statement
            return None

    with pytest.raises(SQLiteRagSmokeUnavailable, match='exact-two'):
        SQLiteRagSmokeCoordinator._assert_final_invariants(
            InvariantSession(), projection  # type: ignore[arg-type]
        )


def test_in_memory_mode_requires_server_test_environment() -> None:
    with pytest.raises(SQLiteRagSmokeUnavailable, match='automated-test-only'):
        SQLiteRagSmokeCoordinator(
            engine=create_engine('sqlite+pysqlite:///:memory:'),
            database_path=':memory:',
            settings=_settings(),
        )
    coordinator = SQLiteRagSmokeCoordinator(
        engine=create_engine('sqlite+pysqlite:///:memory:'),
        database_path=':memory:',
        settings=Settings(paraworks_env='test'),
    )
    assert coordinator is not None


def test_fresh_retrieval_drift_commits_safe_outcome_with_zero_dispatch(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'drift.db').absolute()
    engine = _engine(db_path)
    Base.metadata.create_all(engine)
    settings = _settings()
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
    prepared = _prepared_search(engine, settings=settings)
    with Session(engine) as db:
        db.execute(delete(RagLexicalServingProjection))
        db.commit()

    projection = SQLiteRagSmokeCoordinator(
        engine=engine,
        database_path=db_path,
        settings=settings,
    ).run_keyword(prepared)

    assert projection.outcome == 'evidence_unavailable'
    assert projection.evidence.search_results == ()
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(AgentRun)) == 1
        assert db.scalar(select(func.count()).select_from(AgentRunCostComponent)) == 2


def test_unsealed_substantive_answer_is_rejected_before_any_mutation(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'assistant.db').absolute()
    engine = _engine(db_path)
    Base.metadata.create_all(engine)
    settings = _settings()
    scope = _scope(allowed_permission_levels=('public', 'internal'))
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
        _seed_sqlite_trusted_projection(db)
        conversation = AssistantConversation(user_id='user-1', title='RAG')
        db.add(conversation)
        db.flush()
        user_message = AssistantMessage(
            conversation_id=conversation.id,
            role='user',
            content='TrustedNeedle %_Case',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            metadata_={},
        )
        db.add(user_message)
        db.commit()
        conversation_id = conversation.id
        user_message_id = user_message.id

    with Session(engine) as db:
        text = prepare_assistant_request_text('TrustedNeedle %_Case', (), key=b'k')
        request = RetrievalRequest(
            retrieval_query_text=text.retrieval_query_text,
            security_scope=scope,
            security_scope_fingerprint=security_scope_fingerprint(
                scope, settings=settings
            ),
            query_embedding_result=None,
            candidate_scan_limit=50,
            visible_limit=8,
            relevance_policy_version='rag-retrieval-policy:v2.0',
        )
        retrieval = KeywordEvidenceRetriever(
            store=SqlAlchemyKeywordSearchStore(db=db, settings=settings),
            settings=settings,
        ).invoke(request)
        slots = rank_evidence_slots(retrieval.visible)
        assert len(slots) == 2
        influence = CanonicalEvidenceProjector(
            db=db, settings=settings
        ).prepare_model_influence(
            slots,
            scope=scope,
            prepared_corpus_generation=1,
            prepared_index_generation=None,
            prepared_readiness_hmac=None,
            rendered_input_hmac='d' * 64,
        )
        validated = ValidatedAnswerBlocks(
            blocks=(
                ValidatedAnswerBlock(
                    block_ordinal=0,
                    text='Trusted fact.',
                    evidence_slot_ids=('E1',),
                    support_mode='trusted_fact',
                    confidence_score=Decimal('0.95'),
                    uncertainty_reason=None,
                    block_result_hmac='a' * 64,
                ),
            ),
            insufficient_reason=None,
            selected_slot_ids=('E1',),
            assembled_answer='Trusted fact.',
            assembled_answer_hmac='b' * 64,
            answer_block_audit_set_hmac='c' * 64,
        )
        prepared = PreparedRagFinalization(
            product_kind='answer',
            tentative_outcome='supported',
            prepared_text=text,
            security_scope=scope,
            query_embedding_result=None,
            retrieval_result=retrieval,
            evidence_slots=slots,
            model_influence_observations=influence.observations,
            selected_slot_ids=('E1',),
            validated_answer=validated,
            canned_message_identity=None,
            prepared_model_influence=influence,
            rendered_input_hmac=influence.rendered_input_hmac,
            answer_model_config_snapshot_hmac='e' * 64,
        )

    with pytest.raises(SQLiteRagSmokeUnavailable, match='sealed.*authority'):
        SQLiteRagSmokeCoordinator(
            engine=engine, database_path=db_path, settings=settings
        ).run_keyword(
            prepared,
            assistant_target=AssistantProjectionTarget(
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                owner_user_id='user-1',
            ),
        )

    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(AgentRun)) == 0
        assert db.scalar(select(func.count()).select_from(AgentRunCostComponent)) == 0
        assert db.scalar(
            select(func.count())
            .select_from(AssistantMessage)
            .where(AssistantMessage.role == 'assistant')
        ) == 0
        assert db.scalar(
            select(func.count()).select_from(AssistantMessageEvidenceDependency)
        ) == 0


def test_changed_assistant_target_rolls_back_parent_and_exact_two_children(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'rollback.db').absolute()
    engine = _engine(db_path)
    Base.metadata.create_all(engine)
    settings = _settings()
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
    prepared = _prepared_search(engine, settings=settings)

    with pytest.raises(SQLiteRagSmokeUnavailable, match='target changed'):
        SQLiteRagSmokeCoordinator(
            engine=engine, database_path=db_path, settings=settings
        ).run_keyword(
            prepared,
            assistant_target=AssistantProjectionTarget(
                conversation_id=999,
                user_message_id=999,
                owner_user_id='user-1',
            ),
        )

    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(AgentRun)) == 0
        assert db.scalar(select(func.count()).select_from(AgentRunCostComponent)) == 0


@pytest.mark.parametrize(
    'settings',
    (
        Settings(rag_retrieval_backend='pgvector'),
        Settings(rag_use_pgvector_search=True),
        Settings(paraworks_env='production'),
    ),
)
def test_sqlite_smoke_refuses_non_smoke_modes_before_sidecar(
    tmp_path: Path, settings: Settings
) -> None:
    db_path = (tmp_path / 'refused.db').absolute()
    sqlite3.connect(db_path).close()
    with pytest.raises(SQLiteRagSmokeUnavailable):
        SQLiteRagSmokeCoordinator(
            engine=_engine(db_path), database_path=db_path, settings=settings
        )
    assert not db_path.with_name(db_path.name + '.rag-smoke-process.lock').exists()


def test_file_process_lock_identity_is_never_replaced(tmp_path: Path) -> None:
    db_path = (tmp_path / 'smoke.db').absolute()
    sqlite3.connect(db_path).close()
    engine = _engine(db_path)
    SQLiteRagSmokeCoordinator(engine=engine, database_path=db_path, settings=_settings())
    lock_path = db_path.with_name(db_path.name + '.rag-smoke-process.lock')
    first = _PROCESS_LOCKS[lock_path]
    SQLiteRagSmokeCoordinator(engine=engine, database_path=db_path, settings=_settings())
    assert _PROCESS_LOCKS[lock_path] is first


def test_cached_lock_authority_is_revalidated_before_reuse(tmp_path: Path) -> None:
    db_path = (tmp_path / 'cached.db').absolute()
    sqlite3.connect(db_path).close()
    engine = _engine(db_path)
    SQLiteRagSmokeCoordinator(engine=engine, database_path=db_path, settings=_settings())
    lock_path = db_path.with_name(db_path.name + '.rag-smoke-process.lock')
    lock_alias = tmp_path / 'lock-alias'
    lock_alias.hardlink_to(lock_path)

    with pytest.raises(SQLiteRagSmokeUnavailable, match='lock hardlink'):
        SQLiteRagSmokeCoordinator(
            engine=engine, database_path=db_path, settings=_settings()
        )


@pytest.mark.skipif(os.name != 'nt', reason='Windows case-folded path identity')
def test_case_folded_database_path_alias_is_rejected(tmp_path: Path) -> None:
    db_path = (tmp_path / 'CaseIdentity.db').absolute()
    sqlite3.connect(db_path).close()
    alias = db_path.with_name('CASEIDENTITY.DB')

    with pytest.raises(SQLiteRagSmokeUnavailable, match='ambiguous'):
        SQLiteRagSmokeCoordinator(
            engine=_engine(alias), database_path=alias, settings=_settings()
        )


def test_file_smoke_refuses_second_process_while_lifetime_lock_is_live(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'smoke.db').absolute()
    sqlite3.connect(db_path).close()
    engine = _engine(db_path)
    SQLiteRagSmokeCoordinator(engine=engine, database_path=db_path, settings=_settings())
    context = get_context('spawn')
    queue = context.Queue()
    process = context.Process(
        target=_second_process_lock_attempt, args=(str(db_path), queue)
    )
    process.start()
    process.join(10)
    assert not process.is_alive()
    assert process.exitcode == 0
    assert queue.get(timeout=2) == 'refused'


def test_preexisting_database_hardlink_is_rejected_before_alias_sidecar_creation(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'smoke.db').absolute()
    sqlite3.connect(db_path).close()
    alias = (tmp_path / 'alias.db').absolute()
    alias.hardlink_to(db_path)

    with pytest.raises(SQLiteRagSmokeUnavailable, match='hardlink'):
        SQLiteRagSmokeCoordinator(
            engine=_engine(alias),
            database_path=alias,
            settings=_settings(),
        )

    assert not alias.with_name(alias.name + '.rag-smoke-process.lock').exists()


def test_second_process_refuses_hardlink_alias_without_alias_sidecar(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'smoke.db').absolute()
    sqlite3.connect(db_path).close()
    engine = _engine(db_path)
    SQLiteRagSmokeCoordinator(engine=engine, database_path=db_path, settings=_settings())
    alias = (tmp_path / 'alias.db').absolute()
    alias.hardlink_to(db_path)
    context = get_context('spawn')
    queue = context.Queue()
    process = context.Process(
        target=_second_process_lock_attempt, args=(str(alias), queue)
    )
    process.start()
    process.join(10)
    assert not process.is_alive()
    assert process.exitcode == 0
    assert queue.get(timeout=2) == 'refused'
    assert not alias.with_name(alias.name + '.rag-smoke-process.lock').exists()


def test_hardlink_created_after_start_is_refused_before_begin_immediate(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'late-link.db').absolute()
    engine = _engine(db_path)
    Base.metadata.create_all(engine)
    settings = _settings()
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
    prepared = _prepared_search(engine, settings=settings)
    coordinator = SQLiteRagSmokeCoordinator(
        engine=engine, database_path=db_path, settings=settings
    )
    alias = (tmp_path / 'late-alias.db').absolute()
    alias.hardlink_to(db_path)

    with pytest.raises(SQLiteRagSmokeUnavailable, match='hardlink'):
        coordinator.run_keyword(prepared)

    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(AgentRun)) == 0
