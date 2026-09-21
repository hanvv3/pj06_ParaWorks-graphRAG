from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Lock
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.checkpointing import CheckpointUnavailableError
from backend.app.agent_runtime.retention import (
    CheckpointPruneResult,
    prune_expired_checkpoints,
)
from backend.app.core.config import Settings
from backend.app.models import AgentWorkflowThread
from scripts.admin import prune_langgraph_checkpoints as prune_command

_NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
_CUTOFF = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRUNE_SCRIPT = _REPO_ROOT / 'scripts' / 'admin' / 'prune_langgraph_checkpoints.py'


class _DeleteResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeCheckpointConnection:
    def __init__(
        self,
        events: list[str],
        rowcounts: dict[str, int],
        *,
        execute_error: Exception | None = None,
        commit_error: Exception | None = None,
    ) -> None:
        self._events = events
        self._rowcounts = rowcounts
        self._execute_error = execute_error
        self._commit_error = commit_error
        self.statements: list[tuple[str, tuple[list[str]]]] = []

    def __enter__(self) -> _FakeCheckpointConnection:
        self._events.append('connection_enter')
        return self

    def __exit__(self, *_args: object) -> None:
        self._events.append('connection_exit')

    def transaction(self) -> _FakeCheckpointTransaction:
        return _FakeCheckpointTransaction(
            self._events,
            commit_error=self._commit_error,
        )

    def execute(
        self,
        statement: str,
        parameters: tuple[list[str]],
    ) -> _DeleteResult:
        normalized = ' '.join(statement.split())
        if normalized == 'SELECT pg_advisory_xact_lock(%s)':
            self._events.append('advisory_lock')
            if self._execute_error is not None:
                raise self._execute_error
            return _DeleteResult(0)
        if normalized.startswith('SELECT DISTINCT thread_id FROM'):
            candidate_ids = parameters[0]
            assert isinstance(candidate_ids, list)
            assert parameters == (candidate_ids, candidate_ids, candidate_ids)
            self._events.append('select_remaining')
            return _TransactionalQueryResult(
                rows=[{'thread_id': thread_id} for thread_id in candidate_ids]
            )
        table = normalized.split()[2]
        self._events.append(f'delete:{table}')
        self.statements.append((normalized, parameters))
        if self._execute_error is not None:
            raise self._execute_error
        return _DeleteResult(self._rowcounts.get(table, 0))

    def commit(self) -> None:
        self._events.append('commit')
        if self._commit_error is not None:
            raise self._commit_error


class _FakeCheckpointTransaction:
    def __init__(
        self,
        events: list[str],
        *,
        commit_error: Exception | None,
    ) -> None:
        self._events = events
        self._commit_error = commit_error

    def __enter__(self) -> _FakeCheckpointTransaction:
        self._events.append('transaction_enter')
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        if exc_type is not None:
            self._events.append('transaction_rollback')
            return
        self._events.append('transaction_commit')
        if self._commit_error is not None:
            raise self._commit_error


class _FakeCheckpointPool:
    def __init__(
        self,
        rowcounts: dict[str, int] | None = None,
        *,
        open_error: Exception | None = None,
        execute_error: Exception | None = None,
        commit_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.events: list[str] = []
        self.open_error = open_error
        self.close_error = close_error
        self.connection_value = _FakeCheckpointConnection(
            self.events,
            rowcounts or {},
            execute_error=execute_error,
            commit_error=commit_error,
        )
        self.close_calls = 0

    def open(self, *, wait: bool) -> None:
        assert wait is True
        self.events.extend(['open', 'wait'])
        if self.open_error is not None:
            raise self.open_error

    def connection(self) -> _FakeCheckpointConnection:
        self.events.append('connection')
        return self.connection_value

    def close(self) -> None:
        self.close_calls += 1
        self.events.append('close')
        if self.close_error is not None:
            raise self.close_error


class _TransactionalQueryResult:
    def __init__(
        self,
        *,
        rowcount: int = 0,
        rows: list[dict[str, str]] | None = None,
    ) -> None:
        self.rowcount = rowcount
        self._rows = rows or []

    def fetchall(self) -> list[dict[str, str]]:
        return self._rows


class _TransactionalCheckpointStore:
    def __init__(self, checkpoint_thread_ids: list[str]) -> None:
        self.rows = {
            table: set(checkpoint_thread_ids)
            for table in ('checkpoint_writes', 'checkpoint_blobs', 'checkpoints')
        }
        self.transaction_lock = Lock()
        self.delete_batches: list[list[str]] = []

    def snapshot(self) -> dict[str, set[str]]:
        return {table: set(rows) for table, rows in self.rows.items()}


class _FakeTransaction:
    def __init__(self, connection: _TransactionalCheckpointConnection) -> None:
        self._connection = connection

    def __enter__(self) -> _FakeTransaction:
        store = self._connection.store
        store.transaction_lock.acquire()
        self._connection.events.append('transaction_enter')
        self._connection.transaction_rows = store.snapshot()
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        connection = self._connection
        if exc_type is None:
            assert connection.transaction_rows is not None
            connection.store.rows = connection.transaction_rows
            connection.events.append('transaction_commit')
        else:
            connection.events.append('transaction_rollback')
        connection.transaction_rows = None
        connection.store.transaction_lock.release()


class _TransactionalCheckpointConnection:
    def __init__(
        self,
        store: _TransactionalCheckpointStore,
        events: list[str],
        *,
        fail_after_table: str | None = None,
    ) -> None:
        self.store = store
        self.events = events
        self.fail_after_table = fail_after_table
        self.transaction_rows: dict[str, set[str]] | None = None

    def __enter__(self) -> _TransactionalCheckpointConnection:
        self.events.append('connection_enter')
        return self

    def __exit__(self, *_args: object) -> None:
        self.events.append('connection_exit')

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    def execute(
        self,
        statement: str,
        parameters: tuple[object, ...],
    ) -> _TransactionalQueryResult:
        normalized = ' '.join(statement.split())
        rows = self.transaction_rows or self.store.rows
        if normalized == 'SELECT pg_advisory_xact_lock(%s)':
            self.events.append('advisory_lock')
            return _TransactionalQueryResult()
        if normalized.startswith('SELECT DISTINCT thread_id FROM'):
            candidate_ids = parameters[0]
            assert isinstance(candidate_ids, list)
            assert parameters == (candidate_ids, candidate_ids, candidate_ids)
            remaining = set().union(*rows.values())
            self.events.append('select_remaining')
            return _TransactionalQueryResult(
                rows=[
                    {'thread_id': thread_id}
                    for thread_id in candidate_ids
                    if thread_id in remaining
                ]
            )

        table = normalized.split()[2]
        assert table in rows
        selected_ids = parameters[0]
        assert isinstance(selected_ids, list)
        self.events.append(f'delete:{table}')
        if table == 'checkpoint_writes':
            self.store.delete_batches.append(list(selected_ids))
        deleted_count = len(rows[table].intersection(selected_ids))
        rows[table].difference_update(selected_ids)
        if self.fail_after_table == table:
            raise RuntimeError('credential-marker')
        return _TransactionalQueryResult(rowcount=deleted_count)

    def commit(self) -> None:
        self.events.append('explicit_commit')


class _TransactionalCheckpointPool:
    def __init__(
        self,
        store: _TransactionalCheckpointStore,
        *,
        fail_after_table: str | None = None,
    ) -> None:
        self.events: list[str] = []
        self.connection_value = _TransactionalCheckpointConnection(
            store,
            self.events,
            fail_after_table=fail_after_table,
        )
        self.close_calls = 0

    def open(self, *, wait: bool) -> None:
        assert wait is True
        self.events.extend(['open', 'wait'])

    def connection(self) -> _TransactionalCheckpointConnection:
        self.events.append('connection')
        return self.connection_value

    def close(self) -> None:
        self.close_calls += 1
        self.events.append('close')


def _settings(*, database_url: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=None,
        database_url=database_url
        or 'postgresql+psycopg://runtime:credential-marker@db/checkpoints',
        langgraph_checkpoint_retention_days=30,
    )


def _thread(
    thread_id: str,
    *,
    status: str,
    updated_at: datetime,
    completed_at: datetime | None = None,
    cancelled_at: datetime | None = None,
    checkpoint_store: str = 'postgres',
) -> AgentWorkflowThread:
    return AgentWorkflowThread(
        thread_id=thread_id,
        workflow_name='company-memory-review-v2',
        graph_version='company-memory-review-v2.0',
        checkpoint_thread_id=f'checkpoint:{thread_id}',
        checkpoint_store=checkpoint_store,
        owner_subject_id='owner-1',
        security_scope_id='default',
        client_request_id=None,
        input_hash='a' * 64,
        evidence_version_hash='b' * 64,
        status=status,
        updated_at=updated_at,
        completed_at=completed_at,
        cancelled_at=cancelled_at,
    )


@pytest.fixture
def session_factory() -> Callable[[], Session]:
    engine = create_engine('sqlite://')
    AgentWorkflowThread.__table__.create(engine)
    return sessionmaker(bind=engine)


def _seed(
    session_factory: Callable[[], Session],
    *threads: AgentWorkflowThread,
) -> None:
    with session_factory() as session:
        session.add_all(threads)
        session.commit()


def test_prune_selects_only_expired_postgres_terminal_threads_in_oldest_order(
    session_factory: Callable[[], Session],
) -> None:
    old = _CUTOFF - timedelta(days=10)
    prunable = [
        _thread(
            'completed',
            status='completed',
            updated_at=old - timedelta(days=20),
            completed_at=_CUTOFF - timedelta(days=4),
        ),
        _thread(
            'needs-more-evidence',
            status='needs_more_evidence',
            updated_at=old - timedelta(days=20),
            completed_at=_CUTOFF - timedelta(days=3),
        ),
        _thread(
            'failed',
            status='failed',
            updated_at=_CUTOFF - timedelta(days=2),
        ),
        _thread(
            'cancelled',
            status='cancelled',
            updated_at=old - timedelta(days=20),
            cancelled_at=_CUTOFF - timedelta(days=1),
        ),
    ]
    excluded_statuses = (
        'created',
        'drafting',
        'checkpoint_pending',
        'awaiting_human_review',
        'resuming',
        'checkpoint_failed',
    )
    excluded = [
        _thread(name, status=name, updated_at=old) for name in excluded_statuses
    ]
    excluded.extend(
        [
            _thread(
                'at-cutoff',
                status='completed',
                updated_at=old,
                completed_at=_CUTOFF,
            ),
            _thread(
                'recent',
                status='failed',
                updated_at=_CUTOFF + timedelta(seconds=1),
            ),
            _thread(
                'memory-terminal',
                status='failed',
                updated_at=old,
                checkpoint_store='memory',
            ),
        ]
    )
    _seed(session_factory, *prunable, *excluded)
    pool = _FakeCheckpointPool(
        {
            'checkpoint_writes': 8,
            'checkpoint_blobs': 4,
            'checkpoints': 5,
        }
    )

    result = prune_expired_checkpoints(
        _settings(),
        session_factory=session_factory,
        pool_factory=lambda dsn: _assert_dsn_and_return_pool(dsn, pool),
        now=lambda: _NOW,
    )

    expected_ids = [
        'checkpoint:completed',
        'checkpoint:needs-more-evidence',
        'checkpoint:failed',
        'checkpoint:cancelled',
    ]
    assert result == CheckpointPruneResult(4, 8, 4, 5, _CUTOFF)
    assert pool.events == [
        'open',
        'wait',
        'connection',
        'connection_enter',
        'transaction_enter',
        'advisory_lock',
        'select_remaining',
        'delete:checkpoint_writes',
        'delete:checkpoint_blobs',
        'delete:checkpoints',
        'transaction_commit',
        'connection_exit',
        'close',
    ]
    assert pool.connection_value.statements == [
        (
            'DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)',
            (expected_ids,),
        ),
        (
            'DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)',
            (expected_ids,),
        ),
        (
            'DELETE FROM checkpoints WHERE thread_id = ANY(%s)',
            (expected_ids,),
        ),
    ]
    all_sql = ' '.join(statement for statement, _ in pool.connection_value.statements)
    for forbidden_table in (
        'checkpoint_migrations',
        'agent_workflow_threads',
        'audit_logs',
        'review_items',
        'decision_records',
        'history_events',
        'timeline_events',
        'todos',
    ):
        assert forbidden_table not in all_sql
    with session_factory() as session:
        persisted = session.scalars(
            select(AgentWorkflowThread).order_by(AgentWorkflowThread.thread_id)
        ).all()
    assert len(persisted) == len(prunable) + len(excluded)
    assert {thread.status for thread in persisted} == {
        'completed',
        'needs_more_evidence',
        'failed',
        'cancelled',
        'created',
        'drafting',
        'checkpoint_pending',
        'awaiting_human_review',
        'resuming',
        'checkpoint_failed',
    }


def _assert_dsn_and_return_pool(
    dsn: str,
    pool: _FakeCheckpointPool,
) -> _FakeCheckpointPool:
    assert dsn == 'postgresql://runtime:credential-marker@db/checkpoints'
    return pool


def test_prune_limit_selects_at_most_the_100_oldest_terminal_threads(
    session_factory: Callable[[], Session],
) -> None:
    threads = [
        _thread(
            f'thread-{index:03d}',
            status='failed',
            updated_at=_CUTOFF - timedelta(days=200 - index),
        )
        for index in range(105)
    ]
    _seed(session_factory, *threads)
    pool = _FakeCheckpointPool()

    result = prune_expired_checkpoints(
        _settings(),
        session_factory=session_factory,
        pool_factory=lambda _dsn: pool,
        now=lambda: _NOW,
        limit=100,
    )

    selected_ids = pool.connection_value.statements[0][1][0]
    assert result.selected_thread_count == 100
    assert selected_ids == [f'checkpoint:thread-{index:03d}' for index in range(100)]
    assert 'checkpoint:thread-100' not in selected_ids


def test_no_expired_threads_returns_zero_counts_without_opening_checkpoint_store(
    session_factory: Callable[[], Session],
) -> None:
    _seed(
        session_factory,
        _thread('waiting', status='awaiting_human_review', updated_at=_CUTOFF - timedelta(days=1)),
    )
    pool_calls: list[str] = []

    result = prune_expired_checkpoints(
        _settings(),
        session_factory=session_factory,
        pool_factory=lambda dsn: pool_calls.append(dsn),  # type: ignore[arg-type]
        now=lambda: _NOW,
    )

    assert result == CheckpointPruneResult(0, 0, 0, 0, _CUTOFF)
    assert pool_calls == []


@pytest.mark.parametrize('limit', [0, -1, 1001])
def test_invalid_limit_is_rejected_before_application_or_checkpoint_access(
    limit: int,
) -> None:
    calls: list[str] = []

    with pytest.raises(
        ValueError,
        match='^checkpoint prune limit must be between 1 and 1000$',
    ):
        prune_expired_checkpoints(
            _settings(),
            session_factory=lambda: calls.append('session'),  # type: ignore[arg-type]
            pool_factory=lambda _dsn: calls.append('pool'),  # type: ignore[arg-type]
            now=lambda: _NOW,
            limit=limit,
        )

    assert calls == []


def test_repeat_prune_keeps_application_threads_and_returns_actual_rowcounts(
    session_factory: Callable[[], Session],
) -> None:
    _seed(
        session_factory,
        _thread('repeatable', status='failed', updated_at=_CUTOFF - timedelta(days=1)),
    )
    store = _TransactionalCheckpointStore(['checkpoint:repeatable'])

    def pool_factory(_dsn: str) -> _TransactionalCheckpointPool:
        return _TransactionalCheckpointPool(store)

    first = prune_expired_checkpoints(
        _settings(),
        session_factory=session_factory,
        pool_factory=pool_factory,  # type: ignore[arg-type]
        now=lambda: _NOW,
    )
    second = prune_expired_checkpoints(
        _settings(),
        session_factory=session_factory,
        pool_factory=pool_factory,  # type: ignore[arg-type]
        now=lambda: _NOW,
    )

    assert first == CheckpointPruneResult(1, 1, 1, 1, _CUTOFF)
    assert second == CheckpointPruneResult(0, 0, 0, 0, _CUTOFF)
    with session_factory() as session:
        retained_thread = session.get(AgentWorkflowThread, 'repeatable')
    assert retained_thread is not None
    assert retained_thread.status == 'failed'


def test_sequential_limited_runs_reach_all_remaining_checkpoint_threads(
    session_factory: Callable[[], Session],
) -> None:
    checkpoint_thread_ids = [
        f'checkpoint:thread-{index:03d}' for index in range(105)
    ]
    _seed(
        session_factory,
        *[
            _thread(
                f'thread-{index:03d}',
                status='failed',
                updated_at=_CUTOFF - timedelta(days=200 - index),
            )
            for index in range(105)
        ],
    )
    store = _TransactionalCheckpointStore(checkpoint_thread_ids)

    def pool_factory(_dsn: str) -> _TransactionalCheckpointPool:
        return _TransactionalCheckpointPool(store)

    results = [
        prune_expired_checkpoints(
            _settings(),
            session_factory=session_factory,
            pool_factory=pool_factory,  # type: ignore[arg-type]
            now=lambda: _NOW,
            limit=100,
        )
        for _ in range(3)
    ]

    assert [result.selected_thread_count for result in results] == [100, 5, 0]
    assert store.delete_batches == [checkpoint_thread_ids[:100], checkpoint_thread_ids[100:]]
    assert store.snapshot() == {
        'checkpoint_writes': set(),
        'checkpoint_blobs': set(),
        'checkpoints': set(),
    }
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AgentWorkflowThread)) == 105


def test_concurrent_prune_runs_claim_disjoint_remaining_checkpoint_threads(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / 'retention-concurrency.db'
    engine = create_engine(
        f'sqlite:///{database_path.as_posix()}',
        connect_args={'check_same_thread': False},
    )
    AgentWorkflowThread.__table__.create(engine)
    concurrent_session_factory = sessionmaker(bind=engine)
    checkpoint_thread_ids = [
        f'checkpoint:thread-{index:03d}' for index in range(150)
    ]
    _seed(
        concurrent_session_factory,
        *[
            _thread(
                f'thread-{index:03d}',
                status='failed',
                updated_at=_CUTOFF - timedelta(days=250 - index),
            )
            for index in range(150)
        ],
    )
    store = _TransactionalCheckpointStore(checkpoint_thread_ids)
    both_selected_application_rows = Barrier(2)

    def pool_factory(_dsn: str) -> _TransactionalCheckpointPool:
        both_selected_application_rows.wait(timeout=5)
        return _TransactionalCheckpointPool(store)

    def prune() -> CheckpointPruneResult:
        return prune_expired_checkpoints(
            _settings(),
            session_factory=concurrent_session_factory,
            pool_factory=pool_factory,  # type: ignore[arg-type]
            now=lambda: _NOW,
            limit=100,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: prune(), range(2)))

    first_batch, second_batch = store.delete_batches
    assert sorted(result.selected_thread_count for result in results) == [50, 100]
    assert set(first_batch).isdisjoint(second_batch)
    assert set(first_batch).union(second_batch) == set(checkpoint_thread_ids)
    assert store.snapshot() == {
        'checkpoint_writes': set(),
        'checkpoint_blobs': set(),
        'checkpoints': set(),
    }


@pytest.mark.parametrize(
    'fail_after_table',
    ['checkpoint_writes', 'checkpoint_blobs', 'checkpoints'],
)
def test_delete_failure_rolls_back_every_checkpoint_table_without_partial_effects(
    session_factory: Callable[[], Session],
    fail_after_table: str,
) -> None:
    checkpoint_thread_id = 'checkpoint:expired'
    _seed(
        session_factory,
        _thread('expired', status='failed', updated_at=_CUTOFF - timedelta(days=1)),
    )
    store = _TransactionalCheckpointStore([checkpoint_thread_id])
    before = store.snapshot()
    pool = _TransactionalCheckpointPool(
        store,
        fail_after_table=fail_after_table,
    )

    with pytest.raises(
        CheckpointUnavailableError,
        match='^checkpoint_unavailable$',
    ):
        prune_expired_checkpoints(
            _settings(),
            session_factory=session_factory,
            pool_factory=lambda _dsn: pool,  # type: ignore[arg-type]
            now=lambda: _NOW,
        )

    assert store.snapshot() == before
    assert 'transaction_rollback' in pool.events
    assert 'transaction_commit' not in pool.events
    assert pool.close_calls == 1


@pytest.mark.parametrize(
    ('pool_kwargs', 'expected_events'),
    [
        (
            {'open_error': RuntimeError('credential-marker')},
            ['open', 'wait', 'close'],
        ),
        (
            {'execute_error': RuntimeError('credential-marker')},
            [
                'open',
                'wait',
                'connection',
                'connection_enter',
                'transaction_enter',
                'advisory_lock',
                'transaction_rollback',
                'connection_exit',
                'close',
            ],
        ),
        (
            {'commit_error': RuntimeError('credential-marker')},
            [
                'open',
                'wait',
                'connection',
                'connection_enter',
                'transaction_enter',
                'advisory_lock',
                'select_remaining',
                'delete:checkpoint_writes',
                'delete:checkpoint_blobs',
                'delete:checkpoints',
                'transaction_commit',
                'connection_exit',
                'close',
            ],
        ),
    ],
)
def test_checkpoint_failures_are_sanitized_and_pool_is_closed(
    session_factory: Callable[[], Session],
    pool_kwargs: dict[str, Exception],
    expected_events: list[str],
) -> None:
    _seed(
        session_factory,
        _thread('expired', status='failed', updated_at=_CUTOFF - timedelta(days=1)),
    )
    pool = _FakeCheckpointPool(**pool_kwargs)

    with pytest.raises(CheckpointUnavailableError) as exc_info:
        prune_expired_checkpoints(
            _settings(),
            session_factory=session_factory,
            pool_factory=lambda _dsn: pool,
            now=lambda: _NOW,
        )

    assert str(exc_info.value) == 'checkpoint_unavailable'
    assert 'credential-marker' not in str(exc_info.value)
    assert pool.events == expected_events
    assert pool.close_calls == 1


def test_pool_close_failure_replaces_success_with_sanitized_error(
    session_factory: Callable[[], Session],
) -> None:
    _seed(
        session_factory,
        _thread('expired', status='failed', updated_at=_CUTOFF - timedelta(days=1)),
    )
    pool = _FakeCheckpointPool(close_error=RuntimeError('credential-marker'))

    with pytest.raises(CheckpointUnavailableError) as exc_info:
        prune_expired_checkpoints(
            _settings(),
            session_factory=session_factory,
            pool_factory=lambda _dsn: pool,
            now=lambda: _NOW,
        )

    assert str(exc_info.value) == 'checkpoint_unavailable'
    assert 'credential-marker' not in str(exc_info.value)
    assert pool.close_calls == 1


def test_non_postgres_configuration_is_rejected_without_pool_or_secret(
    session_factory: Callable[[], Session],
) -> None:
    _seed(
        session_factory,
        _thread('expired', status='failed', updated_at=_CUTOFF - timedelta(days=1)),
    )
    pool_calls: list[str] = []
    database_url = 'sqlite:///credential-marker.db'

    with pytest.raises(CheckpointUnavailableError) as exc_info:
        prune_expired_checkpoints(
            _settings(database_url=database_url),
            session_factory=session_factory,
            pool_factory=lambda dsn: pool_calls.append(dsn),  # type: ignore[arg-type]
            now=lambda: _NOW,
        )

    assert str(exc_info.value) == 'checkpoint_unavailable'
    assert database_url not in str(exc_info.value)
    assert 'credential-marker' not in str(exc_info.value)
    assert pool_calls == []


def test_operator_prints_only_cutoff_and_row_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = CheckpointPruneResult(4, 8, 4, 5, _CUTOFF)
    calls: list[tuple[Settings, int]] = []
    settings = _settings()

    monkeypatch.setattr(prune_command, 'get_settings', lambda: settings)
    monkeypatch.setattr(
        prune_command,
        'prune_expired_checkpoints',
        lambda settings_arg, *, limit: calls.append((settings_arg, limit)) or result,
    )

    assert prune_command.main(['--limit', '23']) == 0

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f'cutoff: {_CUTOFF.isoformat()}',
        'selected threads: 4',
        'deleted writes: 8',
        'deleted blobs: 4',
        'deleted checkpoints: 5',
    ]
    assert captured.err == ''
    assert calls == [(settings, 23)]
    for secret in (
        'credential-marker',
        'postgresql://',
        'runtime',
        'checkpoint:completed',
        'source-id',
    ):
        assert secret not in captured.out


def test_operator_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(prune_command, 'get_settings', lambda: _settings())

    def fail(*_args: Any, **_kwargs: Any) -> CheckpointPruneResult:
        raise RuntimeError(
            'postgresql://runtime:credential-marker@db/checkpoints checkpoint:thread'
        )

    monkeypatch.setattr(prune_command, 'prune_expired_checkpoints', fail)

    assert prune_command.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == 'checkpoint prune failed\n'
    assert 'credential-marker' not in captured.err
    assert 'checkpoint:thread' not in captured.err


def test_operator_rejects_invalid_arguments_before_settings_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings_calls: list[str] = []
    monkeypatch.setattr(
        prune_command,
        'get_settings',
        lambda: settings_calls.append('settings'),
    )

    with pytest.raises(SystemExit) as exc_info:
        prune_command.main(['--limit', 'credential-marker'])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert captured.out == ''
    assert captured.err.endswith('invalid arguments\n')
    assert 'credential-marker' not in captured.err
    assert settings_calls == []


def test_operator_import_and_invalid_arguments_do_not_evaluate_settings() -> None:
    environment = os.environ.copy()
    environment['PARAWORKS_DEMO_MODE'] = 'credential-marker'

    completed = subprocess.run(
        [sys.executable, str(_PRUNE_SCRIPT), '--unknown', 'credential-marker'],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ''
    assert completed.stderr.endswith('invalid arguments\n')
    assert 'credential-marker' not in completed.stderr
    assert 'Traceback' not in completed.stderr
