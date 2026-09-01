from __future__ import annotations

import sqlite3
from multiprocessing import get_context
from pathlib import Path

import pytest

from backend.app.agent_runtime.rag_finalization import CanonicalRagProjection
from backend.app.agent_runtime.rag_sqlite_smoke import (
    _PROCESS_LOCKS,
    SQLiteRagSmokeCoordinator,
    SQLiteRagSmokeOperationResult,
    SQLiteRagSmokeUnavailable,
    sqlite_rag_smoke_mutex,
)
from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    prepare_direct_request_text,
)
from backend.app.rag.evidence_projection import V1EvidenceProjection


def _second_process_lock_attempt(database_path: str, queue) -> None:
    connection = sqlite3.connect(database_path)
    try:
        SQLiteRagSmokeCoordinator(
            connection=connection,
            database_path=database_path,
            configured_backend='keyword',
            provider_dispatch_count=0,
            production_mode=False,
        )
    except SQLiteRagSmokeUnavailable:
        queue.put('refused')
    else:
        queue.put('acquired')
    finally:
        connection.close()


def test_sqlite_smoke_uses_one_reentrant_mutex_and_begin_immediate(tmp_path: Path) -> None:
    db_path = tmp_path / 'smoke.db'
    connection = sqlite3.connect(db_path)
    seen: list[tuple[bool, bool]] = []
    coordinator = SQLiteRagSmokeCoordinator(
        connection=connection,
        database_path=db_path,
        configured_backend='keyword',
        provider_dispatch_count=0,
        production_mode=False,
    )

    result = coordinator.run_atomic(lambda db: (
        db.execute('CREATE TABLE proof(value INTEGER)'),
        db.execute('INSERT INTO proof VALUES (1)'),
        seen.append((db.in_transaction, sqlite_rag_smoke_mutex() is sqlite_rag_smoke_mutex())),
        'ok',
    )[-1])

    assert result == 'ok'
    assert seen == [(True, True)]
    assert connection.execute('SELECT value FROM proof').fetchone() == (1,)


def test_sqlite_smoke_crash_rolls_back_whole_product(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / 'smoke.db')
    coordinator = SQLiteRagSmokeCoordinator(
        connection=connection,
        database_path=tmp_path / 'smoke.db',
        configured_backend='keyword',
        provider_dispatch_count=0,
        production_mode=False,
    )
    with pytest.raises(RuntimeError, match='crash'):
        coordinator.run_atomic(lambda db: (
            db.execute('CREATE TABLE proof(value INTEGER)'),
            db.execute('INSERT INTO proof VALUES (1)'),
            (_ for _ in ()).throw(RuntimeError('crash')),
        ))
    assert connection.execute(
        "SELECT count(*) FROM sqlite_master WHERE name='proof'"
    ).fetchone() == (0,)


@pytest.mark.parametrize(
    ('backend', 'dispatches', 'production'),
    [('pgvector', 0, False), ('keyword', 1, False), ('keyword', 0, True)],
)
def test_sqlite_smoke_refuses_non_smoke_modes_before_mutation(
    tmp_path: Path, backend: str, dispatches: int, production: bool
) -> None:
    connection = sqlite3.connect(tmp_path / 'smoke.db')
    with pytest.raises(SQLiteRagSmokeUnavailable):
        SQLiteRagSmokeCoordinator(
            connection=connection,
            database_path=tmp_path / 'smoke.db',
            configured_backend=backend,
            provider_dispatch_count=dispatches,
            production_mode=production,
        )
    assert connection.execute('PRAGMA user_version').fetchone() == (0,)


def test_file_process_lock_identity_is_never_replaced(tmp_path: Path) -> None:
    db_path = (tmp_path / 'smoke.db').absolute()
    first_connection = sqlite3.connect(db_path)
    SQLiteRagSmokeCoordinator(
        connection=first_connection,
        database_path=db_path,
        configured_backend='keyword',
        provider_dispatch_count=0,
        production_mode=False,
    )
    lock_path = db_path.with_name(db_path.name + '.rag-smoke-process.lock')
    first = _PROCESS_LOCKS[lock_path]
    second_connection = sqlite3.connect(db_path)
    SQLiteRagSmokeCoordinator(
        connection=second_connection,
        database_path=db_path,
        configured_backend='keyword',
        provider_dispatch_count=0,
        production_mode=False,
    )
    assert _PROCESS_LOCKS[lock_path] is first


def test_file_smoke_refuses_hardlink_or_wrong_connection_path(tmp_path: Path) -> None:
    db_path = tmp_path / 'smoke.db'
    connection = sqlite3.connect(db_path)
    connection.execute('PRAGMA user_version')
    hardlink = tmp_path / 'alias.db'
    hardlink.hardlink_to(db_path)
    with pytest.raises(SQLiteRagSmokeUnavailable, match='identity mismatch'):
        SQLiteRagSmokeCoordinator(
            connection=connection,
            database_path=hardlink,
            configured_backend='keyword',
            provider_dispatch_count=0,
            production_mode=False,
        )


def test_file_smoke_refuses_second_process_while_lifetime_lock_is_live(
    tmp_path: Path,
) -> None:
    db_path = (tmp_path / 'smoke.db').absolute()
    connection = sqlite3.connect(db_path)
    SQLiteRagSmokeCoordinator(
        connection=connection,
        database_path=db_path,
        configured_backend='keyword',
        provider_dispatch_count=0,
        production_mode=False,
    )
    context = get_context('spawn')
    queue = context.Queue()
    process = context.Process(
        target=_second_process_lock_attempt,
        args=(str(db_path), queue),
    )
    process.start()
    process.join(10)
    assert not process.is_alive()
    assert process.exitcode == 0
    assert queue.get(timeout=2) == 'refused'


def test_run_keyword_asserts_exact_two_final_before_single_commit() -> None:
    connection = sqlite3.connect(':memory:')
    projection = CanonicalRagProjection(
        outcome='no_match',
        answer_text='근거 없음',
        evidence=V1EvidenceProjection((), (), (), (), (), 'a' * 64),
        model_influence=(),
        hidden_match_count=0,
        effective_backend='deterministic_lexical',
        result_hmac='b' * 64,
    )

    def operation(db, **_kwargs):
        db.execute(
            'CREATE TABLE agent_runs('
            'id INTEGER PRIMARY KEY,status TEXT,run_contract_version TEXT,'
            'run_record_phase TEXT,total_charged_cost_usd TEXT,'
            'projection_owner_fence_hmac TEXT)'
        )
        db.execute(
            'CREATE TABLE agent_run_cost_components('
            'agent_run_id INTEGER,component TEXT,component_ordinal INTEGER,'
            'dispatch_state TEXT,attempted INTEGER,dispatch_count INTEGER,'
            'charged_cost_usd TEXT)'
        )
        db.execute(
            "INSERT INTO agent_runs VALUES(1,'complete','rag-run:v2','final',"
            "'0.000000',NULL)"
        )
        db.executemany(
            'INSERT INTO agent_run_cost_components VALUES(?,?,?,?,?,?,?)',
            (
                (1, 'query_embedding', 0, 'terminal', 0, 0, '0.000000'),
                (1, 'answer_generation', 1, 'terminal', 0, 0, '0.000000'),
            ),
        )
        return SQLiteRagSmokeOperationResult(1, projection)

    coordinator = SQLiteRagSmokeCoordinator(
        connection=connection,
        database_path=':memory:',
        configured_backend='keyword',
        provider_dispatch_count=0,
        production_mode=False,
        keyword_operation=operation,
    )
    request = prepare_direct_request_text('질문', key=b'k')
    scope = SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='owner',
        workspace_scope_id='workspace',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=('public',),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )
    assert coordinator.run_keyword(request, scope=scope) == projection
    assert connection.in_transaction is False


def test_run_keyword_rolls_back_when_exact_two_invariant_is_missing() -> None:
    connection = sqlite3.connect(':memory:')
    projection = CanonicalRagProjection(
        outcome='no_match',
        answer_text='근거 없음',
        evidence=V1EvidenceProjection((), (), (), (), (), 'a' * 64),
        model_influence=(),
        hidden_match_count=0,
        effective_backend='deterministic_lexical',
        result_hmac='b' * 64,
    )

    def incomplete(db, **_kwargs):
        db.execute(
            'CREATE TABLE agent_runs('
            'id INTEGER PRIMARY KEY,status TEXT,run_contract_version TEXT,'
            'run_record_phase TEXT,total_charged_cost_usd TEXT,'
            'projection_owner_fence_hmac TEXT)'
        )
        db.execute(
            'CREATE TABLE agent_run_cost_components('
            'agent_run_id INTEGER,component TEXT,component_ordinal INTEGER,'
            'dispatch_state TEXT,attempted INTEGER,dispatch_count INTEGER,'
            'charged_cost_usd TEXT)'
        )
        db.execute(
            "INSERT INTO agent_runs VALUES(1,'running','rag-run:v2',"
            "'cost_finalized_pending_projection','0.000000','fence')"
        )
        return SQLiteRagSmokeOperationResult(1, projection)

    coordinator = SQLiteRagSmokeCoordinator(
        connection=connection,
        database_path=':memory:',
        configured_backend='keyword',
        provider_dispatch_count=0,
        production_mode=False,
        keyword_operation=incomplete,
    )
    request = prepare_direct_request_text('질문', key=b'k')
    scope = SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='owner',
        workspace_scope_id='workspace',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=('public',),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )
    with pytest.raises(SQLiteRagSmokeUnavailable, match='exact-two'):
        coordinator.run_keyword(request, scope=scope)
    assert connection.execute(
        "SELECT count(*) FROM sqlite_master WHERE name='agent_runs'"
    ).fetchone() == (0,)
