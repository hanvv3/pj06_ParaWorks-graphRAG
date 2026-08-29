# Database Storage Initialization Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the key-admin CLI's mixed `db.session` import classifier with one DB-owned typed SQLAlchemy runtime initializer while preserving every existing application session consumer and producing bounded, privacy-safe CLI outcomes.

**Architecture:** Add a leaf `backend.app.db.initialization` module that accepts only a resolved URL and returns an owned `DatabaseRuntime` containing an `Engine` and bound `sessionmaker`. Keep `backend.app.db.session` as the process-global compatibility adapter, while the key-admin CLI creates and disposes its own runtime and classifies configuration, storage availability, operation, and cleanup failures by phase and explicit SQLAlchemy precedence. Initialization remains connection-lazy; real connection failures are classified at command execution.

**Tech Stack:** Python 3.12, SQLAlchemy >=2.0.45, Pydantic 2, FastAPI, PostgreSQL + pgvector, SQLite smoke mode, pytest, Ruff, PowerShell, Docker Compose

**Spec:** `docs/superpowers/specs/2026-08-29-database-storage-initialization-boundary-design.md`

**Status:** Planning complete — user-approved design; independent Spec and Execution reviews PASS; actual implementation authorization is still pending.

## Global Constraints

- Do not change dependency bounds, Alembic schema, LangChain/LangGraph code, Review Queue trust rules, permission policy, token/cost policy, output schemas, or duplicate-resolution rules.
- The initializer public contracts are exactly `DatabaseRuntime`, `initialize_database_runtime(database_url: str) -> DatabaseRuntime`, `DatabaseConfigurationError`, and `DatabaseInitializationError`.
- `DatabaseConfigurationError.code` is exactly `database_configuration_invalid`; `DatabaseInitializationError.code` is exactly `database_initialization_failed`. Instances carry no message or copied original exception.
- Preserve `create_engine(database_url, pool_pre_ping=True)` and `sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=True)`.
- Initializer import and execution must not read `Settings` or environment state and must not call `connect()`, checkout/ping, schema inspection, or SQL.
- Preserve public `engine`, `SessionLocal`, and `get_db` imports, `SessionLocal.kw['bind'] is engine`, request-session close semantics, resolved-URL precedence, and the process-lifetime global application runtime.
- The key-admin CLI must not initialize storage through `db.session`. Its owned runtime is disposed exactly once after success, readiness failure, bounded admin refusal, service-construction failure, command failure, or cleanup failure.
- Do not print until cleanup succeeds or its overriding failure is classified. Every failure is one stdout JSON line with exactly `ok` and `code`; stderr remains empty.
- Tests use fake engines, fake dialects, deterministic hooks, and disposable test databases only. They must not call live LLM, embedding, Slack, Google, OAuth, connector, or other provider APIs.
- PostgreSQL acceptance uses `127.0.0.1:55432` with both database and role names ending in `_test`, zero skips, pre-existing-identity refusal, controller-owned cleanup only, and exact/run-prefix postflight catalog `0:0:0:0`.
- C.5 product Task 6, Slack recovery, CDC/streaming, Deliverable D retrieval, Deliverable E Neo4j GraphRAG, application-wide lazy DB lifecycle, Alembic `engine_from_config`, and unrelated refactors remain out of scope.
- Apply TDD to every behavior change: observe the named RED result before production edits, make the smallest implementation, rerun GREEN, run scoped Ruff, and commit the slice.
- If implementation requires changing any approved taxonomy, public contract, compatibility behavior, cleanup precedence, or privacy boundary, stop and request a new human decision.

---

## File and Responsibility Map

| Path | Responsibility |
|---|---|
| `backend/app/db/initialization.py` | New leaf URL → Engine/sessionmaker runtime factory, typed errors, availability classifier, partial cleanup, idempotent disposal |
| `backend/app/db/session.py` | Existing process-global compatibility adapter exporting `engine`, `SessionLocal`, and `get_db` |
| `backend/app/admin/auto_review_keys.py` | Bounded CLI phases, rotate key-ring prevalidation, command classification, runtime ownership, single emission |
| `backend/tests/test_database_initialization.py` | New initializer, privacy, lifecycle, exact-option, URL-precedence, import, and consumer compatibility tests |
| `backend/tests/test_auto_review_provenance.py` | Existing key-admin in-process/subprocess tests, deterministic fake dialect, error matrix, disposal and output assertions |
| `docs/superpowers/specs/2026-08-29-database-storage-initialization-boundary-design.md` | Approved design status and final observed implementation evidence |
| `plan.md` | Current C.5 execution truth after verified acceptance |
| `docs/portfolio-log.md` | Architecture story and fresh verification evidence |
| `docs/superpowers/runbooks/session-handoff.md` | Exact commits, runtime contract, commands, cleanup, and next C.5 product Task 6 boundary |
| `.superpowers/sdd/2026-08-28-auto-review-trust-promotion/progress.md` | Local ignored C.5 task ledger |
| `.superpowers/sdd/2026-08-28-auto-review-trust-promotion/task-5-report.md` | Local ignored Task 5 evidence report; historical failed reviews remain unchanged |

Dependency order is strict: Task 1 defines the nominal public runtime; Task 2 completes initializer availability/privacy/lifecycle with a real temporary dialect; Task 3 writes all in-process and true-`python -m` CLI RED cases before moving CLI ownership; Task 4 changes the application compatibility adapter and then proves the import matrix; Task 5 runs the fail-closed PostgreSQL regression/cleanup controller; Task 6 synchronizes evidence and obtains final review over unchanged content before committing it.

## Spec Coverage Map

| Approved spec section | Implemented/proved by |
|---|---|
| §1–2 decision and selected DB-owned boundary | Tasks 1–4; no CLI-local `create_engine` and no app-wide lazy refactor |
| §3.1 typed initializer, first-match, privacy, partial/runtime cleanup, exact options | Tasks 1–2 |
| §3.2 application compatibility adapter and URL precedence | Task 4 |
| §3.3 rotate prevalidation, CLI taxonomy, single output, cleanup precedence | Task 3 |
| §4 end-to-end control flow and origin-scoped `ModuleNotFoundError` | Tasks 2–3 |
| §5 deterministic unit/subprocess/import/PostgreSQL tests | Tasks 1–5 |
| §6 file scope and exclusions | Global Constraints and every task file list |
| §7 acceptance, independent review, zero-skip PostgreSQL, cleanup `0:0:0:0` | Tasks 5–6 and Final Implementation Review Checklist |

---

## Execution Entry Gate

This planning artifact and the approved design status must be committed before Task 1 begins. Run this fail-closed check at the start of the implementation session:

```powershell
git ls-files --error-unmatch docs/superpowers/plans/2026-08-29-database-storage-initialization-boundary.md
if ($LASTEXITCODE -ne 0) { throw 'Implementation plan is not tracked' }
git ls-files --error-unmatch docs/superpowers/specs/2026-08-29-database-storage-initialization-boundary-design.md
if ($LASTEXITCODE -ne 0) { throw 'Approved design is not tracked' }
$entryStatus = git status --short
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect implementation entry status' }
if ($entryStatus) { throw "Implementation entry worktree is not clean: $entryStatus" }
$entryHead = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve implementation entry HEAD' }
Write-Host "Implementation entry HEAD: $entryHead"
```

Do not begin product/test edits if either document is untracked or the worktree contains unexplained changes. This check is intentionally before every TDD task, not deferred to final documentation.

---

### Task 1: Add the Typed Database Runtime and Configuration Boundary

**Files:**
- Create: `backend/app/db/initialization.py`
- Create: `backend/tests/test_database_initialization.py`

**Interfaces:**
- Consumes: SQLAlchemy `make_url`, `create_engine`, `Engine`, `Session`, and `sessionmaker`.
- Produces: `DatabaseRuntime.engine: Engine`, `DatabaseRuntime.session_factory: sessionmaker[Session]`, `DatabaseRuntime.dispose() -> None`, `initialize_database_runtime(database_url: str) -> DatabaseRuntime`, `DatabaseConfigurationError`, and `DatabaseInitializationError`.
- Leaves for Task 2: complete availability translation, partial-construction cleanup precedence, and sanitized disposal failure translation.

- [ ] **Step 1: Write the nominal runtime and configuration RED tests**

Create `backend/tests/test_database_initialization.py` with these imports and named tests:

```python
from __future__ import annotations

import traceback
from typing import Any

import pytest
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy import event

from backend.app.db import initialization


def test_initialize_database_runtime_preserves_exact_options_without_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    connect_events: list[str] = []
    engine = sqlalchemy_create_engine('sqlite:///:memory:')
    event.listen(engine, 'connect', lambda *_args: connect_events.append('connect'))

    def create_engine_probe(database_url: str, **kwargs: object):
        observed['database_url'] = database_url
        observed['kwargs'] = kwargs
        return engine

    monkeypatch.setattr(initialization, 'create_engine', create_engine_probe)
    runtime = initialization.initialize_database_runtime('sqlite:///:memory:')

    assert runtime.engine is engine
    assert runtime.session_factory.kw['bind'] is engine
    assert runtime.session_factory.kw['autoflush'] is False
    assert runtime.session_factory.kw['autocommit'] is False
    assert runtime.session_factory.kw['expire_on_commit'] is True
    assert observed == {
        'database_url': 'sqlite:///:memory:',
        'kwargs': {'pool_pre_ping': True},
    }
    assert connect_events == []
    assert 'Engine(' not in repr(runtime)
    engine.dispose()


@pytest.mark.parametrize(
    'database_url',
    (
        'not-a-sqlalchemy-url-sensitive',
        'paraworks_missing_dialect://user:secret@host/database',
    ),
)
def test_configuration_failures_are_typed_and_sanitized(database_url: str) -> None:
    with pytest.raises(initialization.DatabaseConfigurationError) as captured:
        initialization.initialize_database_runtime(database_url)

    error = captured.value
    rendered = ''.join(traceback.format_exception(error))
    assert error.code == 'database_configuration_invalid'
    assert error.args == ()
    assert error.__dict__ == {}
    assert error.__cause__ is None
    assert error.__context__ is None
    assert database_url not in repr(error)
    assert database_url not in rendered
    assert 'secret' not in rendered


def test_typed_error_codes_are_class_level_only() -> None:
    configuration = initialization.DatabaseConfigurationError()
    storage = initialization.DatabaseInitializationError()

    assert configuration.code == 'database_configuration_invalid'
    assert storage.code == 'database_initialization_failed'
    assert configuration.args == storage.args == ()
    assert configuration.__dict__ == storage.__dict__ == {}
```

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```powershell
uv run --no-cache --locked pytest backend/tests/test_database_initialization.py -q
```

Expected: collection fails because `backend.app.db.initialization` does not exist yet (currently `ImportError: cannot import name 'initialization' from 'backend.app.db'`). Do not create production code before recording this RED result.

- [ ] **Step 3: Implement the minimal nominal/configuration runtime**

Create `backend/app/db/initialization.py` with the exact public shape below. Configuration translation is raised only after the active `except` scope has ended.

```python
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, NoSuchModuleError
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    code = 'database_configuration_invalid'


class DatabaseInitializationError(RuntimeError):
    code = 'database_initialization_failed'


@dataclass(slots=True, repr=False)
class DatabaseRuntime:
    engine: Engine
    session_factory: sessionmaker[Session]
    _dispose_attempted: bool = field(default=False, init=False, repr=False)

    def dispose(self) -> None:
        if self._dispose_attempted:
            return
        self._dispose_attempted = True
        self.engine.dispose()


def initialize_database_runtime(database_url: str) -> DatabaseRuntime:
    configuration_failure = False
    try:
        make_url(database_url)
    except ArgumentError:
        configuration_failure = True
    if configuration_failure:
        raise DatabaseConfigurationError()

    configuration_failure = False
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
    except (NoSuchModuleError, ArgumentError):
        configuration_failure = True
    if configuration_failure:
        raise DatabaseConfigurationError()

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=True,
    )
    return DatabaseRuntime(engine=engine, session_factory=session_factory)
```

Do not import `Settings`, `os`, `db.session`, models, CLI modules, or application state.

- [ ] **Step 4: Run the focused tests GREEN**

Run:

```powershell
uv run --no-cache --locked pytest backend/tests/test_database_initialization.py -q
```

Expected: all Task 1 tests pass with zero skips and no connection event.

- [ ] **Step 5: Run the scoped static gate**

Run:

```powershell
uv run --no-cache --locked ruff check backend/app/db/initialization.py backend/tests/test_database_initialization.py
uv run --no-cache --locked python -m compileall -q backend/app/db/initialization.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 6: Commit the typed runtime slice**

```powershell
git add backend/app/db/initialization.py backend/tests/test_database_initialization.py
git commit -m "feat: add typed database runtime initialization"
```

---

### Task 2: Close Availability, Privacy, and Exactly-Once Cleanup Semantics

**Files:**
- Modify: `backend/app/db/initialization.py`
- Modify: `backend/tests/test_database_initialization.py`

**Interfaces:**
- Consumes: Task 1 public runtime contracts.
- Produces: first-match availability classification, sanitized typed translation, partial engine cleanup, and latched `DatabaseRuntime.dispose()` behavior used by Tasks 3–5.
- Error precedence: import/load failure → invalidated DBAPI → Operational/Interface/Timeout/Disconnection → raw nonavailability/programmer error.

- [ ] **Step 1: Add the complete availability hierarchy RED matrix**

Append helpers and parameterized tests. Add `from pathlib import Path` and `from sqlalchemy.dialects import registry` to the test imports. Construct DBAPI subclasses explicitly so inheritance order is tested rather than mocked away.

```python
from sqlalchemy.exc import (
    DBAPIError,
    DataError,
    DisconnectionError,
    IntegrityError,
    InterfaceError,
    InvalidRequestError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
    StatementError,
    TimeoutError as SQLAlchemyTimeoutError,
)


def _dbapi_failure(
    error_type: type[DBAPIError],
    *,
    connection_invalidated: bool,
) -> DBAPIError:
    return error_type(
        'SELECT sensitive_statement',
        {'secret': 'sensitive-param'},
        RuntimeError('sensitive-dbapi-original'),
        connection_invalidated=connection_invalidated,
    )


@pytest.mark.parametrize(
    'failure',
    (
        ModuleNotFoundError('sensitive-driver-module'),
        ImportError('sensitive-native-module'),
        OSError('sensitive-native-loader'),
        OperationalError(
            'statement', {}, RuntimeError('db'), connection_invalidated=False
        ),
        InterfaceError(
            'statement', {}, RuntimeError('db'), connection_invalidated=False
        ),
        SQLAlchemyTimeoutError('sensitive-pool-timeout'),
        DisconnectionError('sensitive-disconnect'),
        _dbapi_failure(IntegrityError, connection_invalidated=True),
        _dbapi_failure(ProgrammingError, connection_invalidated=True),
        _dbapi_failure(DataError, connection_invalidated=True),
        _dbapi_failure(DBAPIError, connection_invalidated=True),
    ),
)
def test_engine_availability_failures_become_sanitized_typed_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    error = captured.value
    rendered = ''.join(traceback.format_exception(error))
    assert error.args == ()
    assert error.__cause__ is None
    assert error.__context__ is None
    assert 'sensitive' not in rendered


@pytest.mark.parametrize(
    'failure',
    (
        _dbapi_failure(IntegrityError, connection_invalidated=False),
        _dbapi_failure(ProgrammingError, connection_invalidated=False),
        _dbapi_failure(DataError, connection_invalidated=False),
        _dbapi_failure(DBAPIError, connection_invalidated=False),
        StatementError('sensitive-statement', 'SELECT 1', {}, RuntimeError('raw')),
        InvalidRequestError('sensitive-invalid-request'),
        SQLAlchemyError('sensitive-generic-sqlalchemy-error'),
        TypeError('sensitive-programmer-error'),
    ),
)
def test_nonavailability_failures_remain_original_operation_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(type(failure)) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    assert captured.value is failure
```

Also add `OperationalError` and `InterfaceError` with both `connection_invalidated=True` and `False`, and all Integrity/Programming/Data/base DBAPI true/false combinations. The expected table is exact: any invalidated DBAPI is typed storage; Operational/Interface are typed storage even when false; every other false subtype remains original.

Before production classification exists, add a deterministic temporary dialect that exercises SQLAlchemy's real plugin and DBAPI-loading path rather than relying only on a monkeypatched `create_engine` exception:

```python
def _write_initializer_dialect_probe(tmp_path: Path) -> Path:
    (tmp_path / 'paraworks_initializer_probe_dialect.py').write_text(
        """
import os

from sqlalchemy.engine.default import DefaultDialect


class ProbeDialect(DefaultDialect):
    name = 'paraworks_initializer_probe'
    driver = 'probe'

    @classmethod
    def import_dbapi(cls):
        failure = os.environ['PARAWORKS_TEST_DBAPI_FAILURE']
        if failure == 'module':
            raise ModuleNotFoundError('dbapi-module-sensitive-sentinel')
        if failure == 'import':
            raise ImportError('dbapi-native-sensitive-sentinel')
        if failure == 'oserror':
            raise OSError('dbapi-loader-sensitive-sentinel')
        raise RuntimeError('unknown-probe-failure')
""".strip()
        + '\n',
        encoding='utf-8',
    )
    return tmp_path


@pytest.mark.parametrize('failure', ('module', 'import', 'oserror'))
def test_real_dialect_dbapi_load_failures_are_typed_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    probe_path = _write_initializer_dialect_probe(tmp_path)
    monkeypatch.syspath_prepend(str(probe_path))
    monkeypatch.setenv('PARAWORKS_TEST_DBAPI_FAILURE', failure)
    registry.register(
        'paraworks_initializer_probe',
        'paraworks_initializer_probe_dialect',
        'ProbeDialect',
    )
    database_url = (
        'paraworks_initializer_probe://probe_role_test:probe_password@'
        '127.0.0.1:55432/probe_database_test'
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        initialization.initialize_database_runtime(database_url)

    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert database_url not in rendered
    assert 'probe_password' not in rendered
    assert 'sensitive-sentinel' not in rendered


def test_real_dialect_programmer_failure_remains_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_path = _write_initializer_dialect_probe(tmp_path)
    monkeypatch.syspath_prepend(str(probe_path))
    monkeypatch.setenv('PARAWORKS_TEST_DBAPI_FAILURE', 'runtime')
    registry.register(
        'paraworks_initializer_probe',
        'paraworks_initializer_probe_dialect',
        'ProbeDialect',
    )

    with pytest.raises(RuntimeError) as captured:
        initialization.initialize_database_runtime(
            'paraworks_initializer_probe://probe_role_test@localhost/probe_database_test'
        )

    assert str(captured.value) == 'unknown-probe-failure'
```

This test must be part of the same RED run as the monkeypatched hierarchy matrix. It fails against Task 1 because DBAPI-owned `ModuleNotFoundError`/`ImportError`/`OSError` are not translated yet, then turns GREEN only after Task 2 implements the shared classifier.

- [ ] **Step 2: Add runtime and partial-cleanup RED tests**

Use a small fake engine with a sequential dispose counter:

```python
class _DisposeProbe:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def dispose(self) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure


def test_runtime_dispose_latches_before_availability_failure() -> None:
    engine = _DisposeProbe(
        OperationalError(
            'dispose-sensitive-statement',
            {'secret': 'dispose-sensitive-param'},
            RuntimeError('dispose-sensitive-original'),
            connection_invalidated=False,
        )
    )
    runtime = initialization.DatabaseRuntime(
        engine=engine,  # type: ignore[arg-type]
        session_factory=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        runtime.dispose()
    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'dispose-sensitive' not in rendered
    runtime.dispose()
    assert engine.calls == 1


def test_runtime_dispose_success_is_idempotent() -> None:
    engine = _DisposeProbe()
    runtime = initialization.DatabaseRuntime(
        engine=engine,  # type: ignore[arg-type]
        session_factory=object(),  # type: ignore[arg-type]
    )

    runtime.dispose()
    runtime.dispose()

    assert engine.calls == 1


def test_runtime_dispose_preserves_programmer_failure_and_does_not_retry() -> None:
    failure = TypeError('dispose-programmer-sentinel')
    engine = _DisposeProbe(failure)
    runtime = initialization.DatabaseRuntime(
        engine=engine,  # type: ignore[arg-type]
        session_factory=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError) as captured:
        runtime.dispose()
    assert captured.value is failure
    runtime.dispose()
    assert engine.calls == 1
```

Add the three sessionmaker-construction cases and the caller-active-context carveout explicitly:

```python
def _raise(error: Exception) -> None:
    raise error


def _chain_members(error: BaseException) -> tuple[BaseException, ...]:
    members: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in members:
        members.append(current)
        current = current.__cause__ or current.__context__
    return tuple(members)


def test_partial_engine_cleanup_success_preserves_sessionmaker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessionmaker_failure = TypeError('sessionmaker-programmer-sentinel')
    engine = _DisposeProbe()
    monkeypatch.setattr(initialization, 'create_engine', lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        initialization,
        'sessionmaker',
        lambda *_args, **_kwargs: _raise(sessionmaker_failure),
    )

    with pytest.raises(TypeError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    assert captured.value is sessionmaker_failure
    assert engine.calls == 1


def test_partial_cleanup_availability_failure_overrides_without_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessionmaker_failure = TypeError('sessionmaker-sensitive-sentinel')
    cleanup_failure = OperationalError(
        'dispose-sensitive-statement',
        {'secret': 'dispose-sensitive-param'},
        RuntimeError('dispose-sensitive-original'),
        connection_invalidated=False,
    )
    engine = _DisposeProbe(cleanup_failure)
    monkeypatch.setattr(initialization, 'create_engine', lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        initialization,
        'sessionmaker',
        lambda *_args, **_kwargs: _raise(sessionmaker_failure),
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'sensitive' not in rendered
    assert engine.calls == 1


def test_partial_cleanup_programmer_failure_overrides_without_factory_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessionmaker_failure = TypeError('sessionmaker-sensitive-sentinel')
    cleanup_failure = TypeError('dispose-programmer-sentinel')
    engine = _DisposeProbe(cleanup_failure)
    monkeypatch.setattr(initialization, 'create_engine', lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        initialization,
        'sessionmaker',
        lambda *_args, **_kwargs: _raise(sessionmaker_failure),
    )

    with pytest.raises(TypeError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value is cleanup_failure
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'sessionmaker-sensitive-sentinel' not in rendered
    assert engine.calls == 1


def test_typed_storage_error_retains_only_caller_active_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_failure = OperationalError(
        'storage-sensitive-statement',
        {'secret': 'storage-sensitive-param'},
        RuntimeError('storage-sensitive-original'),
        connection_invalidated=False,
    )
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: _raise(storage_failure),
    )
    caller_failure = ValueError('caller-active-sentinel')

    try:
        raise caller_failure
    except ValueError:
        with pytest.raises(initialization.DatabaseInitializationError) as captured:
            initialization.initialize_database_runtime('sqlite:///:memory:')

    chain = _chain_members(captured.value)
    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is caller_failure
    assert storage_failure not in chain
    assert 'storage-sensitive' not in rendered
    assert 'caller-active-sentinel' in rendered
```

- [ ] **Step 3: Run Task 2 tests and observe RED**

Run:

```powershell
uv run --no-cache --locked pytest backend/tests/test_database_initialization.py -q
```

Expected: failures show raw availability exceptions, missing cleanup translation, or wrong partial-cleanup precedence. Record the failure output before editing production code.

- [ ] **Step 4: Implement one shared private availability predicate and outside-except raises**

Add these imports and predicate:

```python
from sqlalchemy.exc import (
    DBAPIError,
    DisconnectionError,
    InterfaceError,
    OperationalError,
    TimeoutError as SQLAlchemyTimeoutError,
)


def _is_database_availability_error(error: Exception) -> bool:
    if isinstance(error, (ModuleNotFoundError, ImportError, OSError)):
        return True
    if isinstance(error, DBAPIError) and bool(error.connection_invalidated):
        return True
    return isinstance(
        error,
        (
            OperationalError,
            InterfaceError,
            SQLAlchemyTimeoutError,
            DisconnectionError,
        ),
    )
```

Replace direct engine construction with a captured failure and raise typed errors only after the `except`:

```python
    engine_failure: Exception | None = None
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
    except (NoSuchModuleError, ArgumentError):
        configuration_failure = True
    except Exception as error:
        engine_failure = error

    if configuration_failure:
        raise DatabaseConfigurationError()
    if engine_failure is not None:
        if _is_database_availability_error(engine_failure):
            raise DatabaseInitializationError()
        raise engine_failure
```

Capture sessionmaker construction failure, leave its `except`, attempt engine disposal once, leave the cleanup `except`, then apply this exact precedence:

```python
    session_factory: sessionmaker[Session] | None = None
    session_factory_failure: Exception | None = None
    try:
        session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=True,
        )
    except Exception as error:
        session_factory_failure = error

    if session_factory_failure is not None:
        cleanup_failure: Exception | None = None
        try:
            engine.dispose()
        except Exception as error:
            cleanup_failure = error
        if cleanup_failure is not None:
            if _is_database_availability_error(cleanup_failure):
                raise DatabaseInitializationError()
            raise cleanup_failure
        raise session_factory_failure

    return DatabaseRuntime(engine=engine, session_factory=session_factory)
```

Update `DatabaseRuntime.dispose()` to latch first, capture outside the active `except`, translate only availability errors, and re-raise every other error unchanged:

```python
    def dispose(self) -> None:
        if self._dispose_attempted:
            return
        self._dispose_attempted = True
        cleanup_failure: Exception | None = None
        try:
            self.engine.dispose()
        except Exception as error:
            cleanup_failure = error
        if cleanup_failure is None:
            return
        if _is_database_availability_error(cleanup_failure):
            raise DatabaseInitializationError()
        raise cleanup_failure
```

- [ ] **Step 5: Run the complete initializer gate GREEN**

Run:

```powershell
uv run --no-cache --locked pytest backend/tests/test_database_initialization.py -q
uv run --no-cache --locked ruff check backend/app/db/initialization.py backend/tests/test_database_initialization.py
uv run --no-cache --locked python -m compileall -q backend/app/db/initialization.py
git diff --check
```

Expected: all commands exit 0, every named matrix case passes, and no test skips.

- [ ] **Step 6: Commit lifecycle hardening**

```powershell
git add backend/app/db/initialization.py backend/tests/test_database_initialization.py
git commit -m "fix: bound database runtime lifecycle failures"
```

---

### Task 3: Give the Key-Admin CLI Explicit Runtime Ownership, Phase Taxonomy, and Subprocess Acceptance

**Files:**
- Modify: `backend/app/admin/auto_review_keys.py:444-449`
- Modify: `backend/app/admin/auto_review_keys.py:479-501`
- Modify: `backend/app/admin/auto_review_keys.py:576-646`
- Modify: `backend/tests/test_auto_review_provenance.py:3269-3559`

**Interfaces:**
- Consumes: Task 2 runtime contracts and preserves the existing programmatic `_default_service()` `SessionLocal` fallback, which Task 4 verifies through the compatibility adapter.
- Produces: `_CliOutcome`, `_LoadedDatabaseContract`, `_load_database_contract()`, `_FixedKeyRingSource`, `_dispatch_cli_command()`, phase-specific classifiers, and one-emission `main()`.
- Preserves: public service/helper signatures, canonical `python -m` module alias, parser syntax, success payload fields/order, readiness exit codes, and programmatic `_default_service()` behavior.


Execution order inside this task is strict: Steps 1–5 create and observe all in-process and true-`python -m` RED cases; Steps 6–9 change production code; Steps 10–11 verify and commit the complete CLI slice. Do not execute a production step while any RED case is still missing.

- [ ] **Step 1: Add executable in-process configuration and runtime-ownership RED tests**

In `backend/tests/test_auto_review_provenance.py`, import `SimpleNamespace` from `types` and the two Task 2 database errors, then add a fake owned runtime plus transition helpers that work against both the current and target private seams:

```python
class _OwnedRuntimeProbe:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        dispose_failure: Exception | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.dispose_failure = dispose_failure
        self.dispose_calls = 0

    def dispose(self) -> None:
        self.dispose_calls += 1
        if self.dispose_failure is not None:
            raise self.dispose_failure


def _install_owned_runtime(
    monkeypatch: pytest.MonkeyPatch,
    key_admin: object,
    runtime: _OwnedRuntimeProbe,
    initialize_calls: list[str],
) -> None:
    def initialize(database_url: str) -> _OwnedRuntimeProbe:
        initialize_calls.append(database_url)
        return runtime

    if hasattr(key_admin, '_load_database_contract'):
        contract = SimpleNamespace(
            initialize=initialize,
            configuration_error=DatabaseConfigurationError,
            initialization_error=DatabaseInitializationError,
        )
        monkeypatch.setattr(key_admin, '_load_database_contract', lambda: contract)
        return

    def initialize_legacy() -> sessionmaker:
        initialize('legacy-db-session-path')
        return runtime.session_factory

    monkeypatch.setattr(key_admin, '_initialize_cli_storage', initialize_legacy)


def _install_database_boundary_counter(
    monkeypatch: pytest.MonkeyPatch,
    key_admin: object,
    calls: list[str],
) -> None:
    def boundary_called() -> object:
        calls.append('called')
        raise AssertionError('database boundary must not run')

    if hasattr(key_admin, '_load_database_contract'):
        monkeypatch.setattr(key_admin, '_load_database_contract', boundary_called)
    else:
        monkeypatch.setattr(key_admin, '_initialize_cli_storage', boundary_called)
```

Do not use `raising=False`: the helpers deliberately patch the old seam before production changes and the new seam afterward. Add these named tests so the Step 5 selection is exact:

- `test_key_admin_runtime_owned_and_disposed_once_for_each_command`: `status`, `bootstrap`, `rebuild`, and `rotate` each initialize one CLI-owned runtime and dispose it once;
- `test_key_admin_runtime_disposes_before_readiness_output`: readiness false still returns its aggregate payload/exit 3 after one successful dispose;
- `test_key_admin_runtime_disposes_on_admin_service_and_command_failure`: allowlisted admin refusal, service-construction failure, and command-dispatch failure all dispose once before one JSON line;
- `test_key_admin_cleanup_availability_overrides_command_outcome`: cleanup `DatabaseInitializationError` overrides success or prior command error with `storage_unavailable`/3;
- `test_key_admin_cleanup_programmer_failure_overrides_command_outcome`: cleanup programmer error overrides success or prior command error with `operation_failed`/3;
- calling `dispose()` again does not touch the underlying engine, covered in Task 2;
- no test patches `db.session.SessionLocal` to drive the CLI anymore.

Add `test_key_admin_resolved_url_failure_keeps_configuration_origin`. Patch `Settings` with an object whose `resolved_database_url()` raises first a plain `ValueError`, then the real `DatabaseInitializationError`, and install `_install_database_boundary_counter`. Both cases must produce only `configuration_refused`/2 and leave `calls == []`. This proves a resolver exception cannot borrow initializer semantics merely because it has the same class, and that URL selection finishes before initializer-module loading/invocation. Against the old code, the output happens to be configuration/2 but the legacy storage counter is one, so the RED is semantic rather than a missing-symbol error.

For `test_key_admin_rotate_configuration_precedes_storage`, parameterize missing, blank version, secret shorter than 32 bytes, and equal current/next secret. Install `_install_database_boundary_counter`, invoke `main(['rotate', '--expected-version', 'v1', '--next-version', 'v2', '--reason', 'test-key-validation'])`, and assert:

- missing environment names → `key_ring_unavailable`/2;
- blank/short/equal values → `configuration_refused`/2;
- initializer call count is zero;
- no secret, reason, argument, traceback, or local path is emitted;
- `test_key_admin_nonrotate_never_loads_environment_key_ring` proves `status`/`bootstrap`/`rebuild` never call `_EnvironmentKeyRingSource.load()`.

- [ ] **Step 2: Add the exact command exception first-match RED matrix**

Inject each failure from service construction/dispatch and assert the expected bounded outcome:

| Failure | `connection_invalidated` | Outcome |
|---|---:|---|
| base `DBAPIError` | true | `storage_unavailable`/3 |
| `IntegrityError` | true | `storage_unavailable`/3 |
| `ProgrammingError` | true | `storage_unavailable`/3 |
| `DataError` | true | `storage_unavailable`/3 |
| `OperationalError` | true or false | `storage_unavailable`/3 |
| `InterfaceError` | true or false | `storage_unavailable`/3 |
| SQLAlchemy `TimeoutError` | n/a | `storage_unavailable`/3 |
| `DisconnectionError` | n/a | `storage_unavailable`/3 |
| base/Integrity/Programming/Data `DBAPIError` | false | `operation_failed`/3 |
| `InvalidRequestError`, nonavailability `StatementError`, other `SQLAlchemyError` | n/a | `operation_failed`/3 |
| unknown `ValueError`/`TypeError` during service construction or dispatch | n/a | `operation_failed`/3 |

Implement this table as `test_key_admin_command_failure_first_match_matrix`. Install an owned runtime through `_install_owned_runtime`, inject the failure from service construction or dispatch, and assert the runtime still records one disposal. Use the existing `_assert_bounded_cli_error` assertions for subprocess tests and equivalent exact `capsys` assertions for in-process tests. This fails against the old code on disposal and classification rather than on a missing target symbol.

- [ ] **Step 3: Add the temporary true-`python -m` dialect probe before CLI production edits**

Extend `_run_key_admin_module_cli` with an optional `pythonpath_entries: Sequence[Path] = ()` and import `Sequence` from `collections.abc`. Replace the helper body with this exact isolation boundary: allowlist only operating-system variables needed to launch Python, use a working directory with no `.env`, add the repository root through `PYTHONPATH`, set every test baseline explicitly, apply only the test's delta, then apply removals last. Do not copy arbitrary parent environment or run from the repository root.

```python
def _run_key_admin_module_cli(
    *args: str,
    env_updates: dict[str, str] | None = None,
    env_removals: tuple[str, ...] = (),
    pythonpath_entries: Sequence[Path] = (),
) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[2]
    isolated_cwd = Path(__file__).resolve().parent
    assert not (isolated_cwd / '.env').exists()
    allowed_parent_names = (
        'COMSPEC',
        'PATH',
        'PATHEXT',
        'SYSTEMROOT',
        'TEMP',
        'TMP',
        'WINDIR',
    )
    env = {
        name: os.environ[name]
        for name in allowed_parent_names
        if name in os.environ
    }
    env.update(
        {
            'AUTO_REVIEW_MODE': 'disabled',
            'AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS': '60',
            'AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS': '5',
            'AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS': '120',
            'AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS': '30',
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': 'sqlite:///:memory:',
            'DATABASE_URL': 'sqlite:///:memory:',
        }
    )
    env.update(env_updates or {})
    for name in env_removals:
        env.pop(name, None)
    pythonpath = [*(str(path) for path in pythonpath_entries), str(repository_root)]
    env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, '-m', 'backend.app.admin.auto_review_keys', *args],
        cwd=isolated_cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
```

The baseline must precede `env_updates`, so the malformed-settings test can deliberately override it. Change `test_key_admin_module_cli_bounds_projection_admin_errors` to pass only its explicit environment delta instead of an `os.environ.copy()` snapshot. The child receives no repository `.env`, unrelated recognized Settings variable, secret/provider variable, or inherited `PYTHONPATH`; even a hostile parent value such as `LANGGRAPH_CHECKPOINT_RETENTION_DAYS=not-an-int` cannot preempt the intended dialect/import phase.

Add this helper; it writes only inside pytest's temporary directory:

```python
def _write_database_cli_probe(tmp_path: Path) -> Path:
    (tmp_path / 'sitecustomize.py').write_text(
        """
import os
import sys

from sqlalchemy.dialects import registry

registry.register(
    'paraworks_probe',
    'paraworks_probe_dialect',
    'ProbeDialect',
)

if os.getenv('PARAWORKS_TEST_BLOCK_DB_INITIALIZER') == '1':
    class _InitializerBlocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'backend.app.db.initialization':
                raise ModuleNotFoundError('non-storage-import-sensitive-sentinel')
            return None

    sys.meta_path.insert(0, _InitializerBlocker())
""".strip()
        + '\n',
        encoding='utf-8',
    )
    (tmp_path / 'paraworks_probe_dialect.py').write_text(
        """
import os
from pathlib import Path

from sqlalchemy.engine.default import DefaultDialect


class ProbeDialect(DefaultDialect):
    name = 'paraworks_probe'
    driver = 'probe'

    @classmethod
    def import_dbapi(cls):
        marker = os.getenv('PARAWORKS_TEST_DBAPI_MARKER')
        if marker:
            Path(marker).write_text('called', encoding='utf-8')
        failure = os.environ['PARAWORKS_TEST_DBAPI_FAILURE']
        if failure == 'module':
            raise ModuleNotFoundError('dbapi-module-sensitive-sentinel')
        if failure == 'import':
            raise ImportError('dbapi-native-sensitive-sentinel')
        if failure == 'oserror':
            raise OSError('dbapi-loader-sensitive-sentinel')
        raise RuntimeError('unknown probe failure')
""".strip()
        + '\n',
        encoding='utf-8',
    )
    return tmp_path
```

The fake dialect's only production-like behavior is SQLAlchemy's real dialect registry and `create_engine` calling `import_dbapi()`. It performs no connection and imports no ParaWorks module.

- [ ] **Step 4: Replace environment-dependent CLI tests with deterministic subprocess RED cases**

Change `test_key_admin_module_cli_bounds_storage_initialization_failure` to expect malformed URL → `configuration_refused`/2 and add an unknown-dialect case with the same result.

Replace the missing-`psycopg2` assumption with:

```python
@pytest.mark.parametrize('failure', ('module', 'import', 'oserror'))
def test_key_admin_module_cli_classifies_owned_dbapi_load_failure_as_storage(
    tmp_path: Path,
    failure: str,
) -> None:
    probe_path = _write_database_cli_probe(tmp_path)
    sensitive_url = (
        'paraworks_probe://probe_role_test:probe_password@'
        '127.0.0.1:55432/probe_database_test'
    )
    completed = _run_key_admin_module_cli(
        'status',
        env_updates={
            'AUTO_REVIEW_MODE': 'disabled',
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': sensitive_url,
            'DATABASE_URL': sensitive_url,
            'PARAWORKS_TEST_DBAPI_FAILURE': failure,
        },
        pythonpath_entries=(probe_path,),
    )

    _assert_bounded_cli_error(
        completed,
        code='storage_unavailable',
        exit_code=3,
        forbidden=(
            sensitive_url,
            'paraworks_probe',
            'probe_password',
            'sensitive-sentinel',
        ),
    )
```

Add `test_key_admin_module_cli_classifies_initializer_import_failure_as_configuration`: run a separate process with `PARAWORKS_TEST_BLOCK_DB_INITIALIZER=1` and safe SQLite URL. It must return `configuration_refused`/2, not storage/3, and must hide `non-storage-import-sensitive-sentinel`.

Add `test_key_admin_module_cli_validates_rotate_before_storage` using an invalid rotate key, the probe URL, and `PARAWORKS_TEST_DBAPI_MARKER`. Assert configuration/2 and that the marker file does not exist, proving key validation occurs before DB initialization.

- [ ] **Step 5: Run the complete CLI RED gate before production edits**

Run both the in-process lifecycle/matrix selection and the real module subprocess selection:

```powershell
uv run --no-cache --locked pytest backend/tests/test_auto_review_provenance.py -q -k "key_admin_runtime or key_admin_cleanup or key_admin_rotate_configuration or key_admin_nonrotate or key_admin_command_failure or key_admin_resolved_url or key_admin_module_cli_bounds_forbidden_and_unknown_arguments or key_admin_module_cli_bounds_malformed_settings_before_initialization or key_admin_module_cli_bounds_storage_initialization_failure or key_admin_module_cli_classifies_owned_dbapi_load_failure_as_storage or key_admin_module_cli_classifies_initializer_import_failure_as_configuration or key_admin_module_cli_validates_rotate_before_storage"
```

Expected: the selection fails on semantic assertions, not missing private symbols. The old CLI records zero owned-runtime disposal, initializes storage before rotate validation, classifies nonavailability SQLAlchemy/programmer failures too broadly, misclassifies malformed URL, and cannot honor the initializer-module origin blocker. Preserve the exact RED output before changing production code.

- [ ] **Step 6: Add private CLI transport types and lazy DB contract loading**

Replace `_CliStorageInitializationError` with these private types. Add the required `Callable` and `Protocol` imports and specific SQLAlchemy exceptions; remove the broad main-level `SQLAlchemyError -> storage` rule.

```python
from collections.abc import Callable
from typing import Protocol

from sqlalchemy.exc import (
    DBAPIError,
    DisconnectionError,
    InterfaceError,
    OperationalError,
    TimeoutError as SQLAlchemyTimeoutError,
)
```

```python
@dataclass(frozen=True, slots=True)
class _CliOutcome:
    payload: dict[str, object]
    exit_code: int
    sort_keys: bool


class _CliDatabaseRuntime(Protocol):
    session_factory: sessionmaker[Session]

    def dispose(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _LoadedDatabaseContract:
    initialize: Callable[[str], _CliDatabaseRuntime]
    configuration_error: type[Exception]
    initialization_error: type[Exception]


def _load_database_contract() -> _LoadedDatabaseContract:
    from backend.app.db.initialization import (
        DatabaseConfigurationError,
        DatabaseInitializationError,
        initialize_database_runtime,
    )

    return _LoadedDatabaseContract(
        initialize=initialize_database_runtime,
        configuration_error=DatabaseConfigurationError,
        initialization_error=DatabaseInitializationError,
    )
```

The `pass` in the Protocol method is the abstract structural body, not deferred work. Keep this import inside the bounded configuration phase so a module import failure cannot escape before `main()`.

Add result helpers:

```python
def _error_outcome(*, code: str, exit_code: int) -> _CliOutcome:
    return _CliOutcome(
        payload={'ok': False, 'code': code},
        exit_code=exit_code,
        sort_keys=False,
    )


def _allowlisted_admin_outcome(error: Exception) -> _CliOutcome | None:
    if not isinstance(error, (AutoReviewKeyAdminError, AutoReviewKeyBootstrapError)):
        return None
    code = getattr(error, 'code', None)
    if code not in _CLI_ADMIN_ERROR_CODES:
        return None
    return _error_outcome(code=code, exit_code=2)


def _is_command_storage_unavailable(error: Exception) -> bool:
    if isinstance(error, DBAPIError) and bool(error.connection_invalidated):
        return True
    return isinstance(
        error,
        (
            OperationalError,
            InterfaceError,
            SQLAlchemyTimeoutError,
            DisconnectionError,
        ),
    )
```

- [ ] **Step 7: Prevalidate rotate key material once before storage**

Add the fixed source:

```python
@dataclass(frozen=True, slots=True)
class _FixedKeyRingSource:
    key_ring: FingerprintKeyRing

    def load(self) -> FingerprintKeyRing:
        return self.key_ring


def _prepare_cli_key_ring(args: argparse.Namespace) -> FingerprintKeyRingSource | None:
    if args.command != 'rotate':
        return None
    return _FixedKeyRingSource(_EnvironmentKeyRingSource().load())
```

This reads environment material once before `_load_database_contract()` or runtime initialization. The existing public `rotate_fingerprint_key` helper remains unchanged and may still create its default environment source for non-CLI programmatic use.

- [ ] **Step 8: Separate dispatch from emission without changing payloads**

Move the current four command branches into `_dispatch_cli_command(args, service)` and return, rather than print, the exact aggregate:

```python
def _dispatch_cli_command(
    args: argparse.Namespace,
    service: AutoReviewKeyAdminService,
) -> _CliOutcome:
    if args.command == 'status':
        result = service.status()
        payload = {
            'runtime_generation': result.runtime_generation,
            'runtime_version': result.runtime_version,
            'runtime_ready': result.runtime_ready,
            'projection_ready': result.projection_ready,
            'projection_rebuild_required': result.projection_rebuild_required,
            'nonterminal_extraction_count': result.nonterminal_extraction_count,
            'nonterminal_validation_count': result.nonterminal_validation_count,
        }
        ready = result.runtime_ready and result.projection_ready
    elif args.command == 'bootstrap':
        result = service.bootstrap()
        payload = {
            'schema_available': result.schema_available,
            'initialized': result.initialized,
            'ready': result.ready,
        }
        ready = result.ready
    elif args.command == 'rebuild':
        result = service.rebuild_trusted_fingerprint_projection()
        payload = _rebuild_payload(result)
        ready = result.ready
    else:
        result = service.rotate_fingerprint_key(
            expected_version=args.expected_version,
            next_version=args.next_version,
            reason=args.reason,
            principal='system:local-auto-review-key-admin',
        )
        payload = _rebuild_payload(result)
        ready = result.ready
    return _CliOutcome(
        payload=payload,
        exit_code=0 if ready else 3,
        sort_keys=True,
    )
```

- [ ] **Step 9: Implement phase ownership, cleanup precedence, and one emission**

Implement `_run_cli` with this exact order:

```python
def _run_cli(argv: list[str] | None) -> _CliOutcome:
    try:
        args = build_cli_parser().parse_args(argv)
        settings = Settings()
        key_ring_source = _prepare_cli_key_ring(args)
        database_url = settings.resolved_database_url()
        contract = _load_database_contract()
    except (AutoReviewKeyAdminError, AutoReviewKeyBootstrapError) as error:
        return _allowlisted_admin_outcome(error) or _error_outcome(
            code='configuration_refused',
            exit_code=2,
        )
    except Exception:
        return _error_outcome(code='configuration_refused', exit_code=2)

    try:
        runtime = contract.initialize(database_url)
    except contract.configuration_error:
        return _error_outcome(code='configuration_refused', exit_code=2)
    except contract.initialization_error:
        return _error_outcome(code='storage_unavailable', exit_code=3)
    except Exception:
        return _error_outcome(code='operation_failed', exit_code=3)

    command_outcome: _CliOutcome | None = None
    command_failure: Exception | None = None
    try:
        service = AutoReviewKeyAdminService(
            session_factory=runtime.session_factory,
            settings=settings,
            key_ring_source=key_ring_source,
        )
        command_outcome = _dispatch_cli_command(args, service)
    except Exception as error:
        command_failure = error

    cleanup_failure: Exception | None = None
    try:
        runtime.dispose()
    except Exception as error:
        cleanup_failure = error

    if cleanup_failure is not None:
        if isinstance(cleanup_failure, contract.initialization_error):
            return _error_outcome(code='storage_unavailable', exit_code=3)
        return _error_outcome(code='operation_failed', exit_code=3)

    if command_failure is not None:
        admin_outcome = _allowlisted_admin_outcome(command_failure)
        if admin_outcome is not None:
            return admin_outcome
        if _is_command_storage_unavailable(command_failure):
            return _error_outcome(code='storage_unavailable', exit_code=3)
        return _error_outcome(code='operation_failed', exit_code=3)

    if command_outcome is None:
        return _error_outcome(code='operation_failed', exit_code=3)
    return command_outcome
```

Emit once and only after `_run_cli` returns:

```python
def main(argv: list[str] | None = None) -> int:
    try:
        outcome = _run_cli(argv)
    except Exception:
        outcome = _error_outcome(code='operation_failed', exit_code=3)
    print(
        json.dumps(
            outcome.payload,
            separators=(',', ':'),
            sort_keys=outcome.sort_keys,
        )
    )
    return outcome.exit_code
```

Delete `_CliStorageInitializationError`, `_initialize_cli_storage()`, and the now-unused `_emit_cli_error()`. Do not change the `if __name__ == '__main__'` alias/exit block or `_default_service()` lazy `SessionLocal` fallback.

- [ ] **Step 10: Run the complete CLI boundary GREEN**

Run the in-process lifecycle/matrix gate, deterministic true-`python -m` gate, initializer regressions, and static checks:

```powershell
uv run --no-cache --locked pytest backend/tests/test_auto_review_provenance.py -q -k "key_admin_runtime or key_admin_cleanup or key_admin_rotate_configuration or key_admin_nonrotate or key_admin_command_failure or key_admin_resolved_url or key_admin_module_cli_bounds_forbidden_and_unknown_arguments or key_admin_module_cli_bounds_malformed_settings_before_initialization or key_admin_module_cli_bounds_storage_initialization_failure or key_admin_module_cli_classifies_owned_dbapi_load_failure_as_storage or key_admin_module_cli_classifies_initializer_import_failure_as_configuration or key_admin_module_cli_validates_rotate_before_storage"
uv run --no-cache --locked pytest backend/tests/test_database_initialization.py -q
uv run --no-cache --locked ruff check backend/app/db/initialization.py backend/app/admin/auto_review_keys.py backend/tests/test_database_initialization.py backend/tests/test_auto_review_provenance.py
uv run --no-cache --locked python -m compileall -q backend/app/db/initialization.py backend/app/admin/auto_review_keys.py
uv lock --check
git diff --check
```

Expected: every command exits 0 with zero skips; each owned runtime is disposed exactly once; malformed URL/dialect and non-storage import failure are configuration/2; owned DBAPI load failures are storage/3; output is one bounded JSON line; no test depends on an optional package being absent.

- [ ] **Step 11: Commit the complete CLI ownership slice**

```powershell
git add backend/app/admin/auto_review_keys.py backend/tests/test_auto_review_provenance.py
git commit -m "fix: own key admin database runtime"
```

---

### Task 4: Convert `db.session` into the Process-Global Compatibility Adapter

**Files:**
- Modify: `backend/app/db/session.py:1-18`
- Modify: `backend/tests/test_database_initialization.py`
- Modify: `backend/tests/test_auto_review_provenance.py:3450-3559`
- Test without modification: `backend/tests/test_db_init.py`

**Interfaces:**
- Consumes: completed Task 2 `initialize_database_runtime` and `DatabaseRuntime`; runs after the Task 3 CLI boundary so compatibility verification cannot hide a CLI regression.
- Produces: unchanged public `engine`, `SessionLocal`, and `get_db` objects for FastAPI, Celery, scripts, agent bootstrap/retention, and tests.
- Ownership: one private process-global runtime is retained for process lifetime and is not disposed by this delivery.

- [ ] **Step 1: Add adapter identity, close, URL, and isolated import RED tests**

Add direct public-contract tests:

```python
from collections.abc import Generator

from backend.app.db import session as database_session


def test_session_adapter_preserves_public_engine_and_factory_contract() -> None:
    assert database_session.engine.pool._pre_ping is True
    assert database_session.SessionLocal.kw['bind'] is database_session.engine
    assert database_session.SessionLocal.kw['autoflush'] is False
    assert database_session.SessionLocal.kw['autocommit'] is False
    assert database_session.SessionLocal.kw['expire_on_commit'] is True


def test_get_db_still_closes_the_request_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionProbe:
        closed = False

        def close(self) -> None:
            self.closed = True

    session = SessionProbe()
    monkeypatch.setattr(database_session, 'SessionLocal', lambda: session)
    dependency: Generator[object, None, None] = database_session.get_db()
    assert next(dependency) is session
    dependency.close()
    assert session.closed is True
```

Because `backend/tests/conftest.py` imports `db.session` during collection, prove URL forwarding and import order in fresh processes. Add `os`, `subprocess`, and `sys` imports and this helper. It uses the same operating-system allowlist and Settings baseline as Task 3, adds the repository only through `PYTHONPATH`, and runs from `backend/tests` where `.env` is absent:

```python
def _run_isolated_python(script: str) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[2]
    isolated_cwd = Path(__file__).resolve().parent
    assert not (isolated_cwd / '.env').exists()
    allowed_parent_names = (
        'COMSPEC',
        'PATH',
        'PATHEXT',
        'SYSTEMROOT',
        'TEMP',
        'TMP',
        'WINDIR',
    )
    env = {
        name: os.environ[name]
        for name in allowed_parent_names
        if name in os.environ
    }
    env.update(
        {
            'AUTO_REVIEW_MODE': 'disabled',
            'AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS': '60',
            'AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS': '5',
            'AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS': '120',
            'AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS': '30',
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': 'sqlite:///:memory:',
            'DATABASE_URL': 'sqlite:///:memory:',
            'PYTHONPATH': str(repository_root),
        }
    )
    return subprocess.run(
        [sys.executable, '-c', script],
        cwd=isolated_cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
```

The helper must not copy arbitrary parent variables or run from the repository root.

The URL-forwarding process must:

1. import `backend.app.core.config` and `backend.app.db.initialization`;
2. construct a real `config.Settings(_env_file=None, **case_inputs)` for the case, with `paraworks_demo_mode`, `paraworks_demo_database_url`, `paraworks_database_url`, and `database_url` all supplied explicitly;
3. replace `config.get_settings` with a zero-argument callable returning that real `Settings` instance (do not substitute a fake object or fake `resolved_database_url()` implementation);
4. replace `initialization.initialize_database_runtime` with a probe returning an object with `engine` and `session_factory`;
5. import `backend.app.db.session` for the first time;
6. independently assert `settings.resolved_database_url()` equals the case's expected URL and assert the initializer probe received that exact URL once.

Parameterize the three exact precedence cases:

| Mode | Inputs | Expected URL |
|---|---|---|
| demo | `paraworks_demo_mode=True`, demo URL, ParaWorks URL, and fallback URL all explicitly set | demo URL |
| non-demo override | `paraworks_demo_mode=False`, demo URL, ParaWorks URL, and fallback URL all explicitly set | ParaWorks URL |
| fallback | `paraworks_demo_mode=False`, both override URLs explicitly `None`, and fallback URL explicitly set | `database_url` |

Add an explicit direct/application-first matrix:

```python
CONSUMER_MODULES = (
    'backend.app.db.init_db',
    'backend.app.tasks.sync',
    'backend.app.tasks.rag_indexing',
    'backend.app.agent_runtime.bootstrap',
    'backend.app.agent_runtime.retention',
)


@pytest.mark.parametrize('module_name', CONSUMER_MODULES)
@pytest.mark.parametrize('application_first', (False, True))
def test_database_consumers_import_direct_and_application_first(
    module_name: str,
    application_first: bool,
) -> None:
    statements = ['import importlib']
    if application_first:
        statements.append("importlib.import_module('backend.app.main')")
    statements.append(f'importlib.import_module({module_name!r})')
    completed = _run_isolated_python(';'.join(statements))

    assert completed.returncode == 0
    assert completed.stdout == ''
    assert completed.stderr == ''


@pytest.mark.parametrize('application_first', (False, True))
def test_fastapi_get_db_override_keeps_identity_across_import_order(
    application_first: bool,
) -> None:
    if application_first:
        ordered_imports = (
            'from backend.app.main import app\n'
            'from backend.app.db.session import get_db\n'
        )
    else:
        ordered_imports = (
            'from backend.app.db.session import get_db\n'
            'from backend.app.main import app\n'
        )
    script = ordered_imports + (
        'from fastapi.routing import APIRoute\n'
        'def walk_dependencies(dependant):\n'
        '    yield dependant\n'
        '    for child in dependant.dependencies:\n'
        '        yield from walk_dependencies(child)\n'
        'route = next(\n'
        '    route for route in app.routes\n'
        '    if isinstance(route, APIRoute)\n'
        "    and route.path == '/api/v1/documents'\n"
        "    and 'GET' in route.methods\n"
        ')\n'
        'matches = [\n'
        '    dependant for dependant in walk_dependencies(route.dependant)\n'
        '    if dependant.call is get_db\n'
        ']\n'
        'assert len(matches) == 1\n'
        'def override_db():\n'
        '    yield object()\n'
        'app.dependency_overrides[get_db] = override_db\n'
        'assert app.dependency_overrides[matches[0].call] is override_db\n'
        'app.dependency_overrides.clear()\n'
    )
    completed = _run_isolated_python(script)

    assert completed.returncode == 0
    assert completed.stdout == ''
    assert completed.stderr == ''
```

Also run `db.initialization -> db.session`, `db.session -> db.initialization`, `db.session -> backend.app.main`, and `backend.app.main -> db.session` as explicit isolated scripts. Every process uses the safe SQLite baseline and must not connect, print, or create a cycle.

Add an executable AST/source guard over `backend.app.db.initialization`. Reject `import os`, `from os`, any import of `backend.app.core.config` or `backend.app.db.session`, and attribute/name use of `environ`, `getenv`, `get_settings`, or `Settings`. This proves the leaf initializer cannot read Settings or environment state rather than checking only two module imports.

Retain the existing direct/application-first imports for `backend.app.knowledge.claim_fingerprints`, `backend.app.knowledge.trusted_fingerprint_projection`, `backend.app.knowledge.trusted_provenance`, and `backend.app.review.auto_review_resolution`. Add the exact test `test_database_import_key_admin_canonical_identity_direct_and_module`, whose subprocess assertions prove that direct import and `python -m backend.app.admin.auto_review_keys status` retain the early canonical `sys.modules.setdefault` identity and produce no duplicate module/class identity. The `database_import` selector in every scoped gate must collect this named test.

Update `test_key_admin_status_exit_code_tracks_fresh_readiness_without_key_output` to patch the now-existing `_load_database_contract` with a separate owned runtime around the existing test factory for each call. Assert the healthy call and the post-projection-deletion readiness call each use a distinct runtime disposed exactly once; do not patch `db.session.SessionLocal`.

- [ ] **Step 2: Run adapter tests and observe RED**

Run:

```powershell
uv run --no-cache --locked pytest backend/tests/test_database_initialization.py backend/tests/test_db_init.py -q
uv run --no-cache --locked pytest backend/tests/test_auto_review_provenance.py -q -k "key_admin_status_exit_code or task5_modules_import or database_import"
```

Expected: the URL-forwarding/initializer-probe test fails because current `db.session` still constructs its own engine/sessionmaker and bypasses the new runtime path. Record that observed failure before changing the adapter.

- [ ] **Step 3: Replace only the construction lines in `db.session`**

Keep `get_db()` byte-for-byte equivalent and change the module to:

```python
from collections.abc import Generator

from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.initialization import initialize_database_runtime

settings = get_settings()
_runtime = initialize_database_runtime(settings.resolved_database_url())
engine = _runtime.engine
SessionLocal = _runtime.session_factory


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Do not add an application shutdown disposer, lazy singleton, new dependency injection container, or migration changes.

- [ ] **Step 4: Run compatibility tests GREEN**

Run:

```powershell
uv run --no-cache --locked pytest backend/tests/test_database_initialization.py backend/tests/test_db_init.py backend/tests/test_health.py backend/tests/test_rag_indexing_tasks.py -q
uv run --no-cache --locked pytest backend/tests/test_agent_runtime_bootstrap.py backend/tests/test_agent_runtime_retention.py backend/tests/test_agent_runtime_lifespan.py -q
uv run --no-cache --locked pytest backend/tests/test_auto_review_provenance.py -q -k "key_admin_status_exit_code or task5_modules_import or database_import"
uv run --no-cache --locked ruff check backend/app/db/initialization.py backend/app/db/session.py backend/tests/test_database_initialization.py backend/tests/test_auto_review_provenance.py
uv run --no-cache --locked python -m compileall -q backend/app/db/initialization.py backend/app/db/session.py
git diff --check
```

Expected: every command exits 0. Existing FastAPI/Celery/bootstrap/retention consumers continue importing the same symbols, and the new import matrix emits nothing.

- [ ] **Step 5: Commit the compatibility adapter**

```powershell
git add backend/app/db/session.py backend/tests/test_database_initialization.py backend/tests/test_auto_review_provenance.py
git commit -m "refactor: share database runtime initialization"
```

---


### Task 5: Run the Fail-Closed PostgreSQL Regression and Cleanup Controller

**Files:**
- Test without modification: `backend/tests/test_auto_review_migration.py`
- Test without modification: `backend/tests/test_auto_review_key_bootstrap.py`
- Test without modification: `backend/tests/test_keyed_mutation_guard.py`
- Test without modification: `backend/tests/test_database_initialization.py`
- Test without modification: `backend/tests/test_auto_review_provenance.py`
- Test without modification: `backend/tests/test_review_knowledge_promotion.py`
- Test without modification: `backend/tests/test_review_transitions.py`
- Test without modification: `backend/tests/test_review_transition_postgres.py`
- Test without modification: `backend/tests/test_review_resolution_actors.py`
- Test without modification: `backend/tests/test_auth_api.py`
- Test without modification: `backend/tests/test_review_rbac.py`
- Test without modification: `backend/tests/test_audit_logs.py`
- Test without modification: representative application-consumer modules named in the controller below

**Interfaces:**
- Consumes: committed Tasks 1–4 and an existing or newly started `paraworks-postgres` service that exactly matches the validated image, Compose service label, loopback port, server identity, and health contract.
- Produces: fresh zero-skip initializer, CLI, C.5 product Task 5, product Task 4, consumer, static, and lock evidence plus exact/run-prefix catalog cleanup `0:0:0:0`.
- Owns only the database and role successfully created by this controller invocation. It never stops/removes the container, deletes a volume, or adopts/deletes a pre-existing database or role.

**Release gate:** This task adds no behavior. If a gate exposes a product defect, preserve the failure, return to the owning Task 1–4 TDD step, commit the minimal correction, and rerun this entire controller from the beginning.

- [ ] **Step 1: Confirm the immutable controller targets and refusal policy**

The exact test-only identities are:

```text
database: paraworks_c5t5_dbinit_20260829_database_test
role:     paraworks_c5t5_dbinit_20260829_role_test
password: paraworks-c5t5-dbinit-test-only
container: paraworks-postgres
image:     pgvector/pgvector:pg17
host port: 127.0.0.1:55432
```

The controller must abort before creation when any exact or literal run-prefix database/role count is nonzero. Pre-existing identities are unowned, even if they look stale; do not terminate their sessions or drop them. A later human may decide how to handle such a blocker.

A pre-existing container is treated as shared infrastructure, not controller-owned. Reuse it only when its exact name, image, Compose service label, loopback port, PostgreSQL superuser/database identity, and health all match. Never run `docker compose down`, `docker rm`, or volume deletion.

- [ ] **Step 2: Run creation, every gate, and cleanup as one PowerShell block**

Run the following block from the repository worktree. Do not split it into separate shell calls. Every native command is checked, created-resource ownership is latched only after successful creation, environment values are restored, cleanup attempts continue after an individual cleanup error, and cleanup failure takes reporting precedence.

```powershell
& {
$taskPreviousErrorActionPreference = $ErrorActionPreference
$taskHasNativeErrorPreference =
    Test-Path Variable:PSNativeCommandUseErrorActionPreference
$taskPreviousNativeErrorPreference = $null
if ($taskHasNativeErrorPreference) {
    $taskPreviousNativeErrorPreference =
        $PSNativeCommandUseErrorActionPreference
}

$taskContainer = 'paraworks-postgres'
$taskDatabase = 'paraworks_c5t5_dbinit_20260829_database_test'
$taskRole = 'paraworks_c5t5_dbinit_20260829_role_test'
$taskPassword = 'paraworks-c5t5-dbinit-test-only'
$taskPrefix = 'paraworks_c5t5_dbinit_20260829'
$databaseCreated = $false
$roleCreated = $false
$ownershipPreflightPassed = $false
$verificationFailure = $null
$cleanupFailures = [System.Collections.Generic.List[string]]::new()

$taskEnvironmentNames = @(
    'PARAWORKS_POSTGRES_PORT',
    'PARAWORKS_TEST_POSTGRES_URL',
    'PARAWORKS_DATABASE_URL',
    'DATABASE_URL',
    'PARAWORKS_DEMO_MODE',
    'AUTO_REVIEW_MODE',
    'AUTO_REVIEW_ENFORCE_PERCENTAGE',
    'AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS',
    'AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS',
    'AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS',
    'AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS',
    'AGENT_RUNTIME_FINGERPRINT_KEY_VERSION',
    'AGENT_RUNTIME_FINGERPRINT_SECRET'
)
$previousTaskEnvironment = @{}
foreach ($taskName in $taskEnvironmentNames) {
    $previousTaskEnvironment[$taskName] =
        [Environment]::GetEnvironmentVariable($taskName, 'Process')
}

function Assert-TaskNativeSuccess {
    param([Parameter(Mandatory)][string]$Label)
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with native exit code $LASTEXITCODE"
    }
}

function Get-TaskDockerValue {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $taskOutput = & docker @Arguments
    Assert-TaskNativeSuccess -Label $Label
    return (($taskOutput | Out-String).Trim())
}

function Get-TaskPostgresScalar {
    param([Parameter(Mandatory)][string]$Sql)
    $taskOutput = & docker exec $taskContainer psql -X -U paraworks `
        -d postgres -v ON_ERROR_STOP=1 -tA -c $Sql
    Assert-TaskNativeSuccess -Label 'PostgreSQL scalar query'
    return (($taskOutput | Out-String).Trim())
}

function Invoke-TaskNative {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    & $Command
    Assert-TaskNativeSuccess -Label $Label
}

function Invoke-TaskPytest {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string[]]$TestArguments
    )
    $taskOutput = & uv run --no-cache --locked pytest @TestArguments 2>&1
    $taskExit = $LASTEXITCODE
    Write-Host ($taskOutput -join [Environment]::NewLine)
    if ($taskExit -ne 0) {
        throw "$Label failed with native exit code $taskExit"
    }
    if (($taskOutput -join [Environment]::NewLine) -match '(?i)\bskipped\b') {
        throw "$Label reported a skipped test"
    }
}

function Invoke-TaskCleanupNative {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    try {
        & $Command
        $taskCleanupExit = $LASTEXITCODE
        if ($taskCleanupExit -ne 0) {
            $cleanupFailures.Add(
                "$Label exited with native code $taskCleanupExit"
            )
        }
    }
    catch {
        $cleanupFailures.Add("$Label threw: $($_.Exception.Message)")
    }
}

try {
    $ErrorActionPreference = 'Stop'
    if ($taskHasNativeErrorPreference) {
        $PSNativeCommandUseErrorActionPreference = $false
    }

    if (
        $taskDatabase -ne 'paraworks_c5t5_dbinit_20260829_database_test' -or
        $taskRole -ne 'paraworks_c5t5_dbinit_20260829_role_test' -or
        -not $taskDatabase.EndsWith('_test') -or
        -not $taskRole.EndsWith('_test')
    ) {
        throw 'Refusing unresolved or non-test PostgreSQL identities'
    }

    $env:PARAWORKS_POSTGRES_PORT = '55432'
    $containerMatches = @(
        & docker ps -a --filter "name=^/$taskContainer$" --format '{{.Names}}'
    )
    Assert-TaskNativeSuccess -Label 'Docker container lookup'
    $containerMatches = @(
        $containerMatches | Where-Object { $_ -eq $taskContainer }
    )

    if ($containerMatches.Count -eq 0) {
        Invoke-TaskNative -Label 'Start dedicated PostgreSQL service' -Command {
            docker compose up -d --wait postgres
        }
        $containerMatches = @(
            & docker ps -a --filter "name=^/$taskContainer$" `
                --format '{{.Names}}'
        )
        Assert-TaskNativeSuccess -Label 'Docker container post-start lookup'
        $containerMatches = @(
            $containerMatches | Where-Object { $_ -eq $taskContainer }
        )
    }
    if ($containerMatches.Count -ne 1) {
        throw "Expected exactly one $taskContainer container"
    }

    $containerImage = Get-TaskDockerValue `
        -Label 'Docker image inspection' `
        -Arguments @('inspect', '--format', '{{.Config.Image}}', $taskContainer)
    $containerService = Get-TaskDockerValue `
        -Label 'Docker service-label inspection' `
        -Arguments @(
            'inspect',
            '--format',
            '{{index .Config.Labels "com.docker.compose.service"}}',
            $taskContainer
        )
    $containerPort = Get-TaskDockerValue `
        -Label 'Docker port inspection' `
        -Arguments @('port', $taskContainer, '5432/tcp')
    if (
        $containerImage -ne 'pgvector/pgvector:pg17' -or
        $containerService -ne 'postgres' -or
        $containerPort -ne '127.0.0.1:55432'
    ) {
        throw (
            'Refusing a same-named container that does not match ' +
            'the fixture contract'
        )
    }

    $containerState = Get-TaskDockerValue `
        -Label 'Docker state inspection' `
        -Arguments @(
            'inspect', '--format', '{{.State.Status}}', $taskContainer
        )
    if ($containerState -ne 'running') {
        Invoke-TaskNative -Label 'Start validated PostgreSQL container' -Command {
            docker start $taskContainer
        }
    }

    $containerHealth = ''
    foreach ($taskAttempt in 1..30) {
        $containerHealth = Get-TaskDockerValue `
            -Label 'Docker health inspection' `
            -Arguments @(
                'inspect',
                '--format',
                '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}',
                $taskContainer
            )
        if ($containerHealth -in @('healthy', 'unhealthy')) {
            break
        }
        Start-Sleep -Seconds 1
    }
    if ($containerHealth -ne 'healthy') {
        throw "Validated PostgreSQL container is not healthy: $containerHealth"
    }

    $serverIdentity = Get-TaskPostgresScalar -Sql (
        "SELECT current_user || ':' || current_database();"
    )
    if ($serverIdentity -ne 'paraworks:postgres') {
        throw "Refusing unexpected PostgreSQL server identity: $serverIdentity"
    }

    $catalogSql = @"
SELECT
    (SELECT count(*) FROM pg_database WHERE datname='$taskDatabase') || ':' ||
    (SELECT count(*) FROM pg_roles WHERE rolname='$taskRole') || ':' ||
    (
        SELECT count(*) FROM pg_database
        WHERE left(datname, length('$taskPrefix'))='$taskPrefix'
          AND right(datname, 5)='_test'
    ) || ':' ||
    (
        SELECT count(*) FROM pg_roles
        WHERE left(rolname, length('$taskPrefix'))='$taskPrefix'
          AND right(rolname, 5)='_test'
    );
"@
    $preflightCatalog = Get-TaskPostgresScalar -Sql $catalogSql
    if ($preflightCatalog -ne '0:0:0:0') {
        throw "Refusing pre-existing unowned identities: $preflightCatalog"
    }
    $ownershipPreflightPassed = $true

    Invoke-TaskNative -Label 'Create owned test role' -Command {
        docker exec $taskContainer psql -X -U paraworks -d postgres `
            -v ON_ERROR_STOP=1 -c (
                "CREATE ROLE $taskRole LOGIN PASSWORD '$taskPassword';"
            )
    }
    $roleCreated = $true

    Invoke-TaskNative -Label 'Create owned test database' -Command {
        docker exec $taskContainer psql -X -U paraworks -d postgres `
            -v ON_ERROR_STOP=1 -c (
                "CREATE DATABASE $taskDatabase OWNER $taskRole;"
            )
    }
    $databaseCreated = $true

    Invoke-TaskNative -Label 'Create pgvector extension' -Command {
        docker exec $taskContainer psql -X -U paraworks -d $taskDatabase `
            -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION IF NOT EXISTS vector;'
    }

    $taskDatabaseUrl =
        "postgresql+psycopg://$($taskRole):$taskPassword@127.0.0.1:55432/$taskDatabase"
    $env:PARAWORKS_TEST_POSTGRES_URL = $taskDatabaseUrl
    $env:PARAWORKS_DATABASE_URL = $taskDatabaseUrl
    $env:DATABASE_URL = $taskDatabaseUrl
    $env:PARAWORKS_DEMO_MODE = 'false'
    $env:AUTO_REVIEW_MODE = 'disabled'
    Remove-Item Env:AUTO_REVIEW_ENFORCE_PERCENTAGE -ErrorAction SilentlyContinue
    $env:AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS = '60'
    $env:AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS = '5'
    $env:AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS = '120'
    $env:AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS = '30'
    $env:AGENT_RUNTIME_FINGERPRINT_KEY_VERSION = 'v1'
    $env:AGENT_RUNTIME_FINGERPRINT_SECRET =
        'tttttttttttttttttttttttttttttttttttttttttttttttt'

    Invoke-TaskPytest -Label 'Migration/bootstrap/lock gate' -TestArguments @(
        'backend/tests/test_auto_review_migration.py',
        'backend/tests/test_auto_review_key_bootstrap.py',
        'backend/tests/test_keyed_mutation_guard.py',
        '-q'
    )
    Invoke-TaskPytest -Label 'Database initializer gate' -TestArguments @(
        'backend/tests/test_database_initialization.py',
        '-q'
    )
    Invoke-TaskPytest -Label 'CLI/import gate' -TestArguments @(
        'backend/tests/test_auto_review_provenance.py',
        '-q',
        '-k',
        'key_admin_module_cli or key_admin_status_exit_code or task5_modules_import or database_import'
    )
    Invoke-TaskPytest -Label 'C.5 product Task 5 union' -TestArguments @(
        'backend/tests/test_auto_review_provenance.py',
        'backend/tests/test_review_knowledge_promotion.py',
        'backend/tests/test_review_transitions.py',
        'backend/tests/test_review_transition_postgres.py',
        '-q'
    )
    Invoke-TaskPytest -Label 'C.5 product Task 4 union' -TestArguments @(
        'backend/tests/test_review_resolution_actors.py',
        'backend/tests/test_auth_api.py',
        'backend/tests/test_review_rbac.py',
        'backend/tests/test_audit_logs.py',
        '-q'
    )
    Invoke-TaskPytest `
        -Label 'Representative application consumer gate' `
        -TestArguments @(
            'backend/tests/test_db_init.py',
            'backend/tests/test_health.py',
            'backend/tests/test_agent_runtime_lifespan.py',
            'backend/tests/test_agent_runtime_bootstrap.py',
            'backend/tests/test_agent_runtime_retention.py',
            'backend/tests/test_rag_indexing_tasks.py',
            'backend/tests/test_review_v2_api.py',
            '-q'
        )
    Invoke-TaskNative -Label 'Ruff gate' -Command {
        uv run --no-cache --locked ruff check `
            backend/app/db/initialization.py `
            backend/app/db/session.py `
            backend/app/admin/auto_review_keys.py `
            backend/tests/test_database_initialization.py `
            backend/tests/test_auto_review_provenance.py
    }
    Invoke-TaskNative -Label 'Compile gate' -Command {
        uv run --no-cache --locked python -m compileall -q `
            backend/app/db/initialization.py `
            backend/app/db/session.py `
            backend/app/admin/auto_review_keys.py
    }
    Invoke-TaskNative -Label 'Lock gate' -Command {
        uv lock --check
    }
    Invoke-TaskNative -Label 'Diff gate' -Command {
        git diff --check
    }
}
catch {
    $verificationFailure = $_
}
finally {
    $ErrorActionPreference = 'Stop'
    if ($taskHasNativeErrorPreference) {
        $PSNativeCommandUseErrorActionPreference = $false
    }

    if ($databaseCreated) {
        Invoke-TaskCleanupNative `
            -Label 'Terminate exact test database sessions' `
            -Command {
                docker exec $taskContainer psql -X -U paraworks `
                    -d postgres -v ON_ERROR_STOP=1 -c (
                        "SELECT pg_terminate_backend(pid) " +
                        "FROM pg_stat_activity WHERE datname='$taskDatabase' " +
                        'AND pid <> pg_backend_pid();'
                    )
            }
        Invoke-TaskCleanupNative `
            -Label 'Drop exact owned test database' `
            -Command {
                docker exec $taskContainer psql -X -U paraworks `
                    -d postgres -v ON_ERROR_STOP=1 `
                    -c "DROP DATABASE IF EXISTS $taskDatabase;"
            }
    }

    if ($roleCreated) {
        Invoke-TaskCleanupNative `
            -Label 'Drop exact owned test role' `
            -Command {
                docker exec $taskContainer psql -X -U paraworks `
                    -d postgres -v ON_ERROR_STOP=1 `
                    -c "DROP ROLE IF EXISTS $taskRole;"
            }
    }

    if ($ownershipPreflightPassed) {
        try {
            $postflightOutput = & docker exec $taskContainer psql -X `
                -U paraworks -d postgres -v ON_ERROR_STOP=1 `
                -tA -c $catalogSql
            $postflightExit = $LASTEXITCODE
            $postflightCatalog = (($postflightOutput | Out-String).Trim())
            if ($postflightExit -ne 0) {
                $cleanupFailures.Add(
                    "Postflight query exited with native code $postflightExit"
                )
            }
            elseif ($postflightCatalog -ne '0:0:0:0') {
                $cleanupFailures.Add(
                    "Postflight catalog was $postflightCatalog"
                )
            }
        }
        catch {
            $cleanupFailures.Add(
                "Postflight query threw: $($_.Exception.Message)"
            )
        }
    }

    foreach ($taskName in $taskEnvironmentNames) {
        try {
            [Environment]::SetEnvironmentVariable(
                $taskName,
                $previousTaskEnvironment[$taskName],
                'Process'
            )
        }
        catch {
            $cleanupFailures.Add(
                "Restore environment $taskName threw: $($_.Exception.Message)"
            )
        }
    }

    $ErrorActionPreference = $taskPreviousErrorActionPreference
    if ($taskHasNativeErrorPreference) {
        $PSNativeCommandUseErrorActionPreference =
            $taskPreviousNativeErrorPreference
    }
}

if ($cleanupFailures.Count -ne 0 -and $null -ne $verificationFailure) {
    $verificationText = (($verificationFailure | Out-String).Trim())
    throw (
        "Verification failed: $verificationText | Cleanup failed: " +
        ($cleanupFailures -join '; ')
    )
}
if ($cleanupFailures.Count -ne 0) {
    throw "PostgreSQL cleanup failed: $($cleanupFailures -join '; ')"
}
if ($null -ne $verificationFailure) {
    throw $verificationFailure
}
Write-Host 'C.5 Task 5 verification and cleanup PASS (0:0:0:0)'
}
```

Expected: the block exits successfully, every pytest gate reports no skipped tests, the final line reports PASS, and the postflight catalog is exactly `0:0:0:0`. The container may remain running because it is shared infrastructure; the controller owns and removes only its database and role.

- [ ] **Step 3: Preserve actual output and handle failures without weakening the gate**

Record the actual pass counts, zero-skip evidence, container contract, and `0:0:0:0` cleanup result for this plan's documentation Task 6. Do not copy historical counts as current evidence.

If the controller fails:

1. report the original verification failure and every cleanup failure;
2. do not manually drop a pre-existing identity that the preflight refused;
3. if a created identity remains because cleanup failed, stop and request explicit direction before another destructive action;
4. for a product defect, return to the owning TDD task, add/retain the RED regression, make the minimum correction, commit it, and rerun the whole Task 5 controller.

No Task 5 commit is expected because this is a verification-only checkpoint.

---

### Task 6: Synchronize Evidence, Obtain Final Review, and Commit the Reviewed Documentation

**Files:**
- Modify after observed evidence: `docs/superpowers/specs/2026-08-29-database-storage-initialization-boundary-design.md`
- Modify after observed evidence: `plan.md`
- Modify after observed evidence: `docs/portfolio-log.md`
- Modify after observed evidence: `docs/superpowers/runbooks/session-handoff.md`
- Modify locally after observed evidence: `.superpowers/sdd/2026-08-28-auto-review-trust-promotion/progress.md`
- Modify locally after observed evidence: `.superpowers/sdd/2026-08-28-auto-review-trust-promotion/task-5-report.md`
- Create locally: `.superpowers/sdd/2026-08-28-auto-review-trust-promotion/task-5-review-final-6.md`

**Interfaces:**
- Consumes: unchanged final implementation HEAD from Tasks 1–4 and the complete Task 5 controller evidence.
- Produces: synchronized product truth, an exact staged documentation diff hash, independent final Spec/Quality PASS against the final implementation and documentation content, and one public evidence commit.
- Adds no runtime behavior and does not authorize C.5 product Task 6, Deliverable D/E, Slack recovery, push, merge, or PR creation.

- [ ] **Step 1: Confirm the planning document is already committed and implementation state is stable**

Before editing evidence, run:

```powershell
git ls-files --error-unmatch docs/superpowers/plans/2026-08-29-database-storage-initialization-boundary.md
if ($LASTEXITCODE -ne 0) { throw 'Implementation plan is not tracked' }
$task6Status = git status --short
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect Task 6 status' }
if ($task6Status) { throw "Task 6 entry state is not clean: $task6Status" }
$task6VerifiedHead = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve Task 5-verified HEAD' }
Write-Host "Task 5-verified HEAD: $task6VerifiedHead"
```

Expected: the implementation plan is tracked, no uncommitted product/test changes remain, and HEAD is the exact Task 5-verified implementation commit. If product/test files change after this point, invalidate Task 5 evidence and rerun the entire owning TDD gate plus Task 5.

- [ ] **Step 2: Update product truth only with observed evidence**

After the Task 5 controller passes:

- change the approved spec status from planning-approved to implemented/verified;
- update `plan.md` to state C.5 product Tasks 1–5 actual status and keep product Task 6, D, E, and Slack in the approved order;
- prepend `docs/portfolio-log.md` with the typed-boundary rationale, actual commits, actual pass/skip counts, privacy/lifecycle behavior, and no-live-provider evidence;
- prepend `docs/superpowers/runbooks/session-handoff.md` with public contracts, exact commands/results, disposable identity/cleanup proof, exclusions, and next C.5 product Task 6;
- update ignored `progress.md` to C.5 Task 5 COMPLETE / 5 of 16 and `task-5-report.md` with the final correction/evidence;
- create `task-5-review-final-6.md` for the final independent verdict; never overwrite `task-5-review-final-5.md`.

Do not claim frontend, Slack, product Task 6, Deliverable D, or Deliverable E work occurred.

- [ ] **Step 3: Stage only public evidence and compute the exact review hash**

```powershell
git diff --check
if ($LASTEXITCODE -ne 0) { throw 'Public evidence diff check failed' }
git add docs/superpowers/specs/2026-08-29-database-storage-initialization-boundary-design.md plan.md docs/portfolio-log.md docs/superpowers/runbooks/session-handoff.md
if ($LASTEXITCODE -ne 0) { throw 'Unable to stage public evidence' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Staged evidence diff check failed' }
$stagedEvidenceFiles = @(git diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Unable to list staged evidence' }
$expectedEvidenceFiles = @(
    'docs/portfolio-log.md',
    'docs/superpowers/runbooks/session-handoff.md',
    'docs/superpowers/specs/2026-08-29-database-storage-initialization-boundary-design.md',
    'plan.md'
)
$stagedDifference = Compare-Object `
    -ReferenceObject ($expectedEvidenceFiles | Sort-Object) `
    -DifferenceObject ($stagedEvidenceFiles | Sort-Object)
if ($stagedDifference) {
    throw "Unexpected staged evidence files: $($stagedDifference | Out-String)"
}

$taskTempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$stagedDiffPath = [IO.Path]::GetFullPath(
    [IO.Path]::Combine(
        $taskTempRoot,
        "paraworks-c5t5-staged-$([guid]::NewGuid().ToString('N'))_test.patch"
    )
)
$stagedDiffDirectory = [IO.Path]::GetDirectoryName($stagedDiffPath)
if (
    $stagedDiffDirectory.TrimEnd([IO.Path]::DirectorySeparatorChar) -ne
    $taskTempRoot.TrimEnd([IO.Path]::DirectorySeparatorChar)
) {
    throw 'Refusing staged-diff path outside the system temp directory'
}
try {
    git diff --cached --binary --output=$stagedDiffPath
    if ($LASTEXITCODE -ne 0) { throw 'Unable to materialize staged diff' }
    $stagedDiffHash = git hash-object -- $stagedDiffPath
    if ($LASTEXITCODE -ne 0) { throw 'Unable to hash staged diff' }
    Write-Host "Staged evidence diff hash: $stagedDiffHash"
}
finally {
    if ([IO.File]::Exists($stagedDiffPath)) {
        [IO.File]::Delete($stagedDiffPath)
    }
}
```

Expected: exactly the four public evidence files are staged. Record the staged-diff hash in the local final review report. The implementation plan was committed before execution and therefore is not part of this evidence diff.

- [ ] **Step 4: Obtain independent Spec and Quality PASS on final unchanged content**

Give the reviewer:

- the approved spec and this implementation plan;
- C.5 Task 5 base `38be771ee3f300c76ac735e575ab2f79d87b44f2`;
- the Task 5-verified implementation HEAD;
- the exact staged-diff hash;
- Task 5 command output and `0:0:0:0` cleanup evidence.

Require separate verdicts:

- Spec PASS/FAIL against every initializer, CLI, compatibility, lifecycle, privacy, and exclusion clause;
- Quality PASS/FAIL for exception hierarchy, import cycles, session ownership, duplicate runtimes, test determinism, sensitive output, and controller safety;
- Critical/Important/Minor findings with file/line references;
- confirmation that the reviewed staged diff hash matches Step 3;
- no code or documentation edits by the reviewer.

Write the verdict to the exact local `task-5-review-final-6.md` path and include exactly one machine-readable line `reviewed_staged_diff_hash: <observed git hash-object value>`. Any Critical or Important finding blocks completion. A product/test correction returns to its owning TDD task, invalidates and reruns Task 5, restages evidence, recomputes the hash, and repeats the full review. A documentation-only correction must be restaged, rehashed, and fully re-reviewed. The final commit must contain the exact reviewed staged content.

- [ ] **Step 5: Commit the exact reviewed public evidence and prove clean tracked state**

Only after independent Spec PASS and Quality PASS:

```powershell
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Reviewed staged diff changed or is invalid' }
$reviewReportPath = '.superpowers/sdd/2026-08-28-auto-review-trust-promotion/task-5-review-final-6.md'
$reviewedHashLines = @(
    Select-String `
        -LiteralPath $reviewReportPath `
        -Pattern '^reviewed_staged_diff_hash: ([0-9a-fA-F]{40,64})$'
)
if ($reviewedHashLines.Count -ne 1) {
    throw 'Final review report must contain exactly one reviewed staged hash'
}
$reviewedStagedDiffHash =
    $reviewedHashLines[0].Matches[0].Groups[1].Value.ToLowerInvariant()

$task6TempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$task6DiffPath = [IO.Path]::GetFullPath(
    [IO.Path]::Combine(
        $task6TempRoot,
        "paraworks-c5t5-final-$([guid]::NewGuid().ToString('N'))_test.patch"
    )
)
$task6DiffDirectory = [IO.Path]::GetDirectoryName($task6DiffPath)
if (
    $task6DiffDirectory.TrimEnd([IO.Path]::DirectorySeparatorChar) -ne
    $task6TempRoot.TrimEnd([IO.Path]::DirectorySeparatorChar)
) {
    throw 'Refusing final staged-diff path outside the system temp directory'
}
try {
    git diff --cached --binary --output=$task6DiffPath
    if ($LASTEXITCODE -ne 0) { throw 'Unable to materialize final staged diff' }
    $currentStagedDiffHash = git hash-object -- $task6DiffPath
    if ($LASTEXITCODE -ne 0) { throw 'Unable to hash final staged diff' }
}
finally {
    if ([IO.File]::Exists($task6DiffPath)) {
        [IO.File]::Delete($task6DiffPath)
    }
}
if ($currentStagedDiffHash.ToLowerInvariant() -ne $reviewedStagedDiffHash) {
    throw 'Staged evidence changed after independent review'
}
git commit -m "docs: accept task 5 database initialization boundary"
if ($LASTEXITCODE -ne 0) { throw 'Unable to commit reviewed public evidence' }
$task6FinalStatus = git status --short
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect final tracked state' }
if ($task6FinalStatus) { throw "Final tracked state is not clean: $task6FinalStatus" }
$task6FinalHead = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve final evidence HEAD' }
Write-Host "Final evidence HEAD: $task6FinalHead"
```

Expected: the commit succeeds, tracked status is clean, and the ignored SDD progress/report/review files remain local evidence rather than being force-added. Do not push, merge, open a PR, deploy, or begin C.5 product Task 6 without separate user authorization.

---

## Final Implementation Review Checklist

- [ ] The initializer is a leaf and reads only its `database_url` argument.
- [ ] Exact engine/sessionmaker options, bind identity, lazy connection, and app-global lifetime are preserved.
- [ ] Typed errors have fixed class-level codes, empty args/dict, and retain no initializer input or caught original in rendered chains.
- [ ] Initializer and command DBAPI subclass matrices apply `connection_invalidated=True` before concrete subtype rules.
- [ ] Nonavailability SQLAlchemy and programmer errors are `operation_failed`, never storage.
- [ ] Partial cleanup and runtime cleanup attempt disposal once, latch before attempt, never retry, and apply the approved overriding precedence.
- [ ] Rotate key material is validated once before storage; non-rotate commands never read it.
- [ ] CLI-owned runtime is distinct from the process-global `db.session` runtime and is disposed before any output.
- [ ] Every CLI failure is one exact two-field JSON line with empty stderr and no URL, driver, secret, local path, argument, or traceback.
- [ ] Fresh `python -m` tests reproduce ModuleNotFoundError, ImportError, and OSError through a temporary registered dialect, not optional package absence.
- [ ] Direct/application-first imports cover FastAPI, `init_db`, Celery tasks, bootstrap/retention, `db.session`, initializer, and key-admin identity without cycles or output.
- [ ] New focused, Task 5, Task 4, representative consumer, Ruff, compile, lock, and diff gates have fresh evidence.
- [ ] PostgreSQL database and role both end in `_test`, all required PostgreSQL tests have zero skips, and exact/run-prefix cleanup is `0:0:0:0`.
- [ ] Independent Spec and Quality reviews both PASS with no unresolved Critical or Important finding.
- [ ] Slack, CDC/streaming, C.5 product Task 6, Deliverable D, Deliverable E, migrations, dependencies, and unrelated refactors remain untouched.
- [ ] No push, merge, PR, deployed rollout, or next-task implementation occurs without separate user authorization.
