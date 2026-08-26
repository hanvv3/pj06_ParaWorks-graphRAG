# LangGraph Runtime and Checkpoint Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Deliverable B's source-agnostic LangGraph runtime, persistence, and checkpoint lifecycle primitives on a clean non-Slack backend baseline, without exposing or cutting over a public V2 workflow.

**Architecture:** Keep durable graph state limited to JSON-safe ids, hashes, counters, and bounded status codes. PostgreSQL remains the application source of truth through workflow-owned tables, while LangGraph persistence is managed by a lifespan-scoped saver selected by an explicit mode matrix; graph execution can update application status only after a synchronous saver tuple is confirmed. The existing Slack, Mail/Document, RAG, Review Queue, and legacy orchestration routes remain behaviorally unchanged in this deliverable.

**Tech Stack:** Python 3.12, LangChain 1.3.17, LangGraph 1.2.11, `langgraph-checkpoint-postgres` 3.1.2, psycopg 3.3.4, psycopg-pool 3.3.1, FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL, SQLite smoke mode, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-08-26-langchain-langgraph-runtime-foundation-design.md`

## Global Constraints

- Keep `langchain>=1.3.17,<1.4.0`, `langgraph>=1.2.11,<1.3.0`, `langchain-openai>=1.6.0,<1.7.0`, `langchain-google-genai>=4.3.5,<4.4.0`, and `langgraph-checkpoint-postgres>=3.1.2,<3.2.0` unchanged.
- Do not add, remove, or change any public V2 endpoint in Deliverable B.
- Do not implement `interrupt()`-driven Review Queue behavior, business handling of a review-resolution `Command`, ReviewItem transitions, or knowledge promotion behavior; those belong to Deliverable C.
- Do not change `/ask`, `/search`, assistant retrieval, or add Neo4j; those belong to Deliverable D and the later GraphRAG program.
- Do not change Slack application code or Slack tests in this plan. The ten known Slack-related backend failures are explicitly deferred to the final Slack restoration phase because the former live Slack data source is unavailable.
- Do not skip, delete, weaken, or mark Slack tests `xfail`. The complete suite must continue to expose those failures until the Slack phase resolves them.
- Use Gmail, Drive, Calendar, approved knowledge, deterministic fixtures, and fake models for non-Slack verification. No live Slack, Google, LLM, embedding, or OAuth call is permitted in automated tests.
- Keep `LANGGRAPH_REVIEW_V2_ENABLED=false` and `LANGGRAPH_RAG_V2_ENABLED=false` by default.
- Production PostgreSQL must never fall back silently to `InMemorySaver`; unavailable checkpoint infrastructure fails only the future Review V2 path closed.
- `PostgresSaver.setup()` is allowed only in the explicit bootstrap command, never during import, application startup, readiness, or ordinary tests.
- Initial and resumed durable invocations use `durability="sync"`; top-level `checkpoint_ns` stays at the LangGraph root namespace `""`.
- Checkpoint state and audit metadata must not contain questions, prompts, source URLs, snippets, message bodies, model output, provider exception text, credentials, ORM objects, sessions, registries, or clients.
- Existing full-suite baseline on the accepted dependency lock is `518 passed, 11 failed, 1 skipped`; after Task 1 it must become the same suite with only the ten explicitly deferred Slack failures and zero new failures.

---

### Task 1: Restore the Non-Slack Backend Baseline Contract

**Files:**
- Modify: `backend/tests/test_agent_preflight.py:18-22`
- Test: `backend/tests/test_agent_preflight.py`
- Test: `backend/tests/test_mail_document_agent_review_bridge.py`

**Interfaces:**
- Consumes: `PermissionContext.allowed_permission_levels` and the existing source/chunk permission filters in `build_mail_document_evidence_packet`.
- Produces: an accurate admin preflight fixture that explicitly grants access to its seeded `restricted` Drive evidence.

The production permission filter is correct. Commit `b6c5d04` strengthened both `Source` and `DocumentChunk` filtering and updated similar tests to pass exact permission levels, but missed this preflight test. Do not infer permissions from `role`, broaden `PermissionContext` defaults, or weaken the query.

- [ ] **Step 1: Re-run the existing regression and preserve the RED evidence**

```powershell
uv run --locked pytest backend/tests/test_agent_preflight.py::test_mail_document_agent_preflight_reports_cost_without_running_llm -vv
```

Expected: FAIL at `evidence_message_count == 1` with actual value `0`.

- [ ] **Step 2: Correct only the stale permission fixture**

Replace the current context construction with:

```python
permission_context=PermissionContext(
    user_id='demo-admin',
    role='admin',
    allowed_permission_levels=('public', 'internal', 'restricted'),
),
```

Do not change the seeded evidence from `restricted` to `internal`; the test must still prove that preflight cost calculation includes evidence the caller is explicitly permitted to read.

- [ ] **Step 3: Verify the preflight file is GREEN**

```powershell
uv run --locked pytest backend/tests/test_agent_preflight.py -q
```

Expected: both preflight tests pass.

- [ ] **Step 4: Verify the permission guard remains fail-closed**

```powershell
uv run --locked pytest backend/tests/test_mail_document_agent_review_bridge.py::test_mail_document_evidence_packet_excludes_disallowed_permissions backend/tests/test_mail_document_agent_review_bridge.py::test_mail_document_evidence_packet_excludes_disallowed_source_permission_even_when_chunk_is_internal -q
```

Expected: `2 passed`; restricted evidence remains hidden from a public/internal-only caller.

- [ ] **Step 5: Lint and commit the bounded baseline repair**

```powershell
uv run --locked ruff check backend/tests/test_agent_preflight.py
```

```powershell
git add backend/tests/test_agent_preflight.py
git commit -m "test: align mail preflight permission fixture"
```

---

### Task 2: Add Checkpoint-Safe State and Keyed Fingerprint Contracts

**Files:**
- Create: `backend/app/agent_runtime/state.py`
- Create: `backend/app/agent_runtime/fingerprints.py`
- Modify: `backend/app/agent_runtime/__init__.py`
- Create: `backend/tests/test_agent_runtime_state.py`
- Create: `backend/tests/test_agent_runtime_fingerprints.py`

**Interfaces:**
- Consumes: LangGraph `StateGraph(ReviewGraphState, context_schema=ReviewRuntimeContext)` typed-state conventions and the approved spec's state/error allowlists.
- Produces: `ReviewGraphInput`, `ReviewGraphState`, `ReviewGraphOutput`, `ReviewRuntimeContext`, `merge_completed_nodes`, `merge_error_codes`, `validate_checkpoint_state`, `canonical_json_bytes`, and `keyed_fingerprint`.

- [ ] **Step 1: Write RED tests for replacement fields, bounded reducers, and state safety**

Create `backend/tests/test_agent_runtime_state.py` with these cases:

```python
import pytest

from backend.app.agent_runtime.state import (
    merge_completed_nodes,
    merge_error_codes,
    validate_checkpoint_state,
)


def test_completed_nodes_are_append_unique_and_bounded() -> None:
    current = [f'node-{index}' for index in range(64)]
    assert merge_completed_nodes(current, ['node-1', 'node-64']) == [
        *current[1:],
        'node-64',
    ]


def test_error_codes_reject_unknown_values_and_stay_bounded() -> None:
    with pytest.raises(ValueError, match='unsupported checkpoint error code'):
        merge_error_codes([], ['raw-provider-exception'])

    merged = merge_error_codes(
        ['invalid_input'] * 2,
        ['checkpoint_failed', 'checkpoint_failed'],
    )
    assert merged == ['invalid_input', 'checkpoint_failed']


@pytest.mark.parametrize(
    'unsafe_key,unsafe_value',
    [
        ('question', 'secret question'),
        ('source_url', 'https://restricted.example'),
        ('model_output', {'answer': 'secret'}),
        ('session', object()),
    ],
)
def test_checkpoint_state_rejects_sensitive_or_non_json_values(
    unsafe_key: str,
    unsafe_value: object,
) -> None:
    state = {
        'workflow_thread_id': 'thread-1',
        'graph_version': 'company-memory-review-v2.0',
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_item_ids': [],
        'review_status_counts': {},
        'phase': 'created',
        'completed_nodes': [],
        'error_codes': [],
        unsafe_key: unsafe_value,
    }

    with pytest.raises(ValueError):
        validate_checkpoint_state(state)
```

- [ ] **Step 2: Run the state tests and verify RED**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_state.py -q
```

Expected: collection fails because `backend.app.agent_runtime.state` does not exist.

- [ ] **Step 3: Implement the exact checkpoint state boundary**

Create `backend/app/agent_runtime/state.py` with:

```python
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Protocol

from langgraph.runtime import Runtime
from typing_extensions import TypedDict

MAX_COMPLETED_NODES = 64
MAX_ERROR_CODES = 16
ALLOWED_REVIEW_ERROR_CODES = frozenset({
    'invalid_input',
    'idempotency_key_reused',
    'evidence_changed',
    'permission_denied',
    'checkpoint_unavailable',
    'checkpoint_failed',
    'review_unresolved',
    'runtime_version_unavailable',
    'model_unavailable',
    'concurrent_resume',
    'invalid_state_transition',
})


class ReviewGraphInput(TypedDict):
    workflow_thread_id: str


class ReviewGraphState(TypedDict):
    workflow_thread_id: str
    graph_version: str
    input_hash: str
    evidence_version_hash: str
    review_item_ids: list[int]
    review_status_counts: dict[str, int]
    phase: str
    completed_nodes: Annotated[list[str], merge_completed_nodes]
    error_codes: Annotated[list[str], merge_error_codes]


class ReviewGraphOutput(TypedDict):
    workflow_thread_id: str
    status: str
    review_item_count: int
    review_status_counts: dict[str, int]
    error_codes: list[str]


class PermissionResolver(Protocol):
    def __call__(self, actor_subject_id: str) -> Sequence[str]:
        raise NotImplementedError


class ReviewRuntimeContext(TypedDict):
    session_factory: Callable[[], object]
    actor_subject_id: str
    permission_resolver: PermissionResolver
    agent_registry: object
    draft_service: object
    lease_service: object


ReviewRuntime = Runtime[ReviewRuntimeContext]
```

Define both reducers before the TypedDict declarations. `merge_completed_nodes` must append new values once and retain only the newest 64. `merge_error_codes` must reject values outside `ALLOWED_REVIEW_ERROR_CODES`, append once, and retain only the newest 16. `validate_checkpoint_state` must allow only the nine `ReviewGraphState` keys and recursively accept only exact `None`, `bool`, `int`, finite `float`, `str`, `list`, and string-keyed `dict` values; tuples, bytes, exceptions, ORM objects, and extra keys must raise `ValueError`.

- [ ] **Step 4: Write RED tests for canonical HMAC fingerprints**

Create `backend/tests/test_agent_runtime_fingerprints.py`:

```python
import unicodedata

from backend.app.agent_runtime.fingerprints import keyed_fingerprint


def test_keyed_fingerprint_is_stable_for_key_order_and_unicode_form() -> None:
    secret = b'test-only-fingerprint-secret'
    composed = '담당자 김하나'
    decomposed = unicodedata.normalize('NFD', composed)

    left = keyed_fingerprint(
        {'name': composed, 'agents': ['mail_document_agent', 'memory_extraction_agent']},
        secret=secret,
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )
    right = keyed_fingerprint(
        {'agents': ['mail_document_agent', 'memory_extraction_agent'], 'name': decomposed},
        secret=secret,
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )

    assert left == right
    assert len(left) == 64
    assert composed not in left


def test_keyed_fingerprint_changes_for_list_order_policy_or_secret() -> None:
    payload = {'source_refs': ['source-a:v1', 'source-b:v2']}
    baseline = keyed_fingerprint(
        payload,
        secret=b'secret-a',
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )

    assert baseline != keyed_fingerprint(
        {'source_refs': list(reversed(payload['source_refs']))},
        secret=b'secret-a',
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )
    assert baseline != keyed_fingerprint(
        payload,
        secret=b'secret-a',
        schema_version='review-input:v1',
        policy_version='ranked:v2',
    )
    assert baseline != keyed_fingerprint(
        payload,
        secret=b'secret-b',
        schema_version='review-input:v1',
        policy_version='ranked:v1',
    )
```

- [ ] **Step 5: Run the fingerprint tests and verify RED**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_fingerprints.py -q
```

Expected: collection fails because `backend.app.agent_runtime.fingerprints` does not exist.

- [ ] **Step 6: Implement versioned canonical HMAC-SHA-256**

Create `backend/app/agent_runtime/fingerprints.py` with these public signatures:

```python
def _normalize_json_value(value: object) -> object:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('fingerprint payload contains a non-finite float')
        return value
    if type(value) is str:
        return unicodedata.normalize('NFC', value)
    if type(value) is list:
        return [_normalize_json_value(item) for item in value]
    if type(value) is dict:
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('fingerprint payload keys must be strings')
            normalized_key = unicodedata.normalize('NFC', key)
            if normalized_key in normalized:
                raise ValueError('fingerprint payload has duplicate normalized keys')
            normalized[normalized_key] = _normalize_json_value(item)
        return normalized
    raise ValueError('fingerprint payload must contain JSON-safe primitives')


def canonical_json_bytes(value: object) -> bytes:
    normalized = _normalize_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')


def keyed_fingerprint(
    value: object,
    *,
    secret: bytes,
    schema_version: str,
    policy_version: str,
) -> str:
    if not secret or not schema_version.strip() or not policy_version.strip():
        raise ValueError('fingerprint secret and versions are required')
    payload = b'\n'.join(
        [
            schema_version.encode('utf-8'),
            policy_version.encode('utf-8'),
            canonical_json_bytes(value),
        ]
    )
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()
```

Never return or log the canonical payload.

- [ ] **Step 7: Export, run, lint, and commit the runtime contracts**

Export the six public state/fingerprint contracts used by later tasks from `backend/app/agent_runtime/__init__.py` without removing legacy exports.

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_state.py backend/tests/test_agent_runtime_fingerprints.py -q
```

```powershell
uv run --locked ruff check backend/app/agent_runtime/state.py backend/app/agent_runtime/fingerprints.py backend/app/agent_runtime/__init__.py backend/tests/test_agent_runtime_state.py backend/tests/test_agent_runtime_fingerprints.py
```

```powershell
git add backend/app/agent_runtime/state.py backend/app/agent_runtime/fingerprints.py backend/app/agent_runtime/__init__.py backend/tests/test_agent_runtime_state.py backend/tests/test_agent_runtime_fingerprints.py
git commit -m "feat: add checkpoint safe runtime contracts"
```

---

### Task 3: Add Application-Owned Workflow and Idempotency Schema

**Files:**
- Create: `backend/app/models/agent_workflows.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/models/review.py`
- Modify: `backend/app/models/agent_runs.py`
- Modify: `backend/app/models/knowledge.py`
- Create: `backend/migrations/versions/2f6a8b9c0d1e_add_agent_workflow_runtime_foundation.py`
- Create: `backend/tests/test_agent_workflow_models.py`
- Create: `backend/tests/test_agent_runtime_migration.py`
- Modify: `backend/tests/test_db_init.py`
- Modify: `backend/tests/test_db_schema_operations.py`

**Interfaces:**
- Consumes: the HMAC values from Task 2 and the current SQLAlchemy/Alembic baseline pattern.
- Produces: `AgentWorkflowThread`, `AgentWorkflowRequest`, `AgentWorkflowEvidenceRef`, `AgentRuntimeSchemaVersion`, and nullable thread/provenance columns required by Deliverables C and D.

- [ ] **Step 1: Write RED model tests for thread ownership and uniqueness**

The new tests must create an in-memory SQLite schema and assert:

```python
assert inspect(engine).get_unique_constraints('agent_workflow_evidence_refs')
assert thread.status == 'created'
assert thread.state_version == 0
assert request.request_kind == 'review_source_versions'
```

Also insert duplicate non-null values and require `IntegrityError` for:

```text
(security_scope_id, workflow_name, owner_subject_id, client_request_id)
(workflow_thread_id, ordinal)
(workflow_thread_id, candidate_key)
(workflow_thread_id, effect_key)
DecisionRecord.source_review_item_id
HistoryEvent.source_review_item_id
TimelineEvent.source_review_item_id
Todo.source_review_item_id
```

Insert two rows with null client/idempotency/provenance columns and assert both are accepted, preserving compatibility for legacy data.

- [ ] **Step 2: Run the model tests and verify RED**

```powershell
uv run --locked pytest backend/tests/test_agent_workflow_models.py -q
```

Expected: collection fails because the workflow models do not exist.

- [ ] **Step 3: Create the workflow persistence models**

Define the following fields exactly in `backend/app/models/agent_workflows.py`:

```python
def utc_now() -> datetime:
    return datetime.now(UTC)


class AgentWorkflowThread(Base):
    __tablename__ = 'agent_workflow_threads'

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workflow_name: Mapped[str] = mapped_column(String(64), index=True)
    graph_version: Mapped[str] = mapped_column(String(64))
    checkpoint_thread_id: Mapped[str] = mapped_column(String(128), unique=True)
    checkpoint_store: Mapped[str] = mapped_column(String(32))
    owner_subject_id: Mapped[str] = mapped_column(String(128), index=True)
    security_scope_id: Mapped[str] = mapped_column(String(128), default='default')
    client_request_id: Mapped[str | None] = mapped_column(String(128))
    input_hash: Mapped[str] = mapped_column(String(64))
    evidence_version_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default='created', index=True)
    state_version: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    checkpoint_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_by_subject_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class AgentWorkflowRequest(Base):
    __tablename__ = 'agent_workflow_requests'

    workflow_thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    input_schema_version: Mapped[str] = mapped_column(String(32))
    request_kind: Mapped[str] = mapped_column(String(64), default='review_source_versions')
    agent_names: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    selection_policy_version: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(32))


class AgentWorkflowEvidenceRef(Base):
    __tablename__ = 'agent_workflow_evidence_refs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_thread_id: Mapped[str] = mapped_column(String(64), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    canonical_source_type: Mapped[str] = mapped_column(String(32))
    canonical_table: Mapped[str] = mapped_column(String(64))
    canonical_row_id: Mapped[int] = mapped_column(Integer)
    document_version_id: Mapped[int | None] = mapped_column(Integer)
    external_revision: Mapped[str | None] = mapped_column(String(255))
    content_signature: Mapped[str] = mapped_column(String(128))
    permission_level_snapshot: Mapped[str] = mapped_column(String(32))
    content_fingerprint: Mapped[str] = mapped_column(String(64))


class AgentRuntimeSchemaVersion(Base):
    __tablename__ = 'agent_runtime_schema_versions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(64), unique=True)
    package_name: Mapped[str] = mapped_column(String(128))
    package_version: Mapped[str] = mapped_column(String(32))
    schema_revision: Mapped[int] = mapped_column(Integer)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
```

Use table constraints/indexes rather than role inference:

```python
Index(
    'uq_agent_workflow_thread_client_request',
    'security_scope_id',
    'workflow_name',
    'owner_subject_id',
    'client_request_id',
    unique=True,
    postgresql_where=text('client_request_id IS NOT NULL'),
    sqlite_where=text('client_request_id IS NOT NULL'),
)
UniqueConstraint('workflow_thread_id', 'ordinal', name='uq_agent_workflow_evidence_ref_ordinal')
```

Do not duplicate source bodies, URLs, snippets, or prompts in these tables.

- [ ] **Step 4: Add nullable idempotency/provenance columns without behavior changes**

Add these mapped columns and partial unique indexes:

```text
ReviewItem.workflow_thread_id: String(64), nullable
ReviewItem.candidate_key: String(128), nullable
ReviewItem.predecessor_review_item_id: Integer, nullable
AgentRun.workflow_thread_id: String(64), nullable
AgentRun.effect_key: String(128), nullable
DecisionRecord.source_review_item_id: Integer, nullable
HistoryEvent.source_review_item_id: Integer, nullable
TimelineEvent.source_review_item_id: Integer, nullable
Todo.source_review_item_id: Integer, nullable
```

Index names must be:

```text
uq_review_items_workflow_candidate
uq_agent_runs_workflow_effect
uq_decision_records_source_review_item
uq_history_events_source_review_item
uq_timeline_events_source_review_item
uq_todos_source_review_item
```

Each index is unique only when its nullable key columns are non-null. Do not modify current insert, review, or promotion services in Deliverable B.

- [ ] **Step 5: Export the models and verify fresh SQLite metadata**

Export all four workflow models from `backend/app/models/__init__.py`. Extend `test_init_db_creates_expected_tables_on_fresh_engine` so the expected set includes:

```python
{
    'agent_workflow_threads',
    'agent_workflow_requests',
    'agent_workflow_evidence_refs',
    'agent_runtime_schema_versions',
}
```

Run:

```powershell
uv run --locked pytest backend/tests/test_agent_workflow_models.py backend/tests/test_db_init.py::test_init_db_creates_expected_tables_on_fresh_engine -q
```

Expected: GREEN.

- [ ] **Step 6: Write RED migration tests for fresh and existing schemas**

Create `backend/tests/test_agent_runtime_migration.py`. Use a temporary SQLite file and `Alembic Config('alembic.ini')`; set `PARAWORKS_DATABASE_URL` to that file and clear `get_settings()` around each Alembic command.

The fresh test runs `command.upgrade(config, 'head')` and asserts all new tables, columns, and index names exist.

The existing-schema test must:

1. create minimal legacy `review_items`, `agent_runs`, `decision_records`, `history_events`, `timeline_events`, and `todos` tables without the new columns;
2. run `command.stamp(config, 'b4b6d9f4d3e1')`;
3. run `command.upgrade(config, 'head')`;
4. assert legacy rows remain and all new nullable columns/tables/indexes exist;
5. run `command.upgrade(config, 'head')` a second time and assert it is idempotent.

Run:

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_migration.py -q
```

Expected: RED because the new revision is absent.

- [ ] **Step 7: Implement an idempotent Alembic revision**

Create revision `2f6a8b9c0d1e`, with `down_revision = 'b4b6d9f4d3e1'`. Because `0001_create_current_schema` imports current `Base.metadata`, every operation must inspect first so a fresh full upgrade does not recreate current-model tables, columns, or indexes.

Use these helpers with concrete table/column/index arguments:

```python
def _create_table_if_missing(table_name: str, *columns, **kwargs) -> None:
    if table_name not in inspect(op.get_bind()).get_table_names():
        op.create_table(table_name, *columns, **kwargs)


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    column_names = {
        item['name'] for item in inspect(op.get_bind()).get_columns(table_name)
    }
    if column.name not in column_names:
        op.add_column(table_name, column)


def _create_index_if_missing(
    index_name: str,
    table_name: str,
    columns: list[str],
    *,
    unique: bool,
    where: str | None = None,
) -> None:
    index_names = {
        item['name'] for item in inspect(op.get_bind()).get_indexes(table_name)
    }
    if index_name in index_names:
        return
    dialect_options = {}
    if where is not None:
        dialect_options = {
            'postgresql_where': sa.text(where),
            'sqlite_where': sa.text(where),
        }
    op.create_index(
        index_name,
        table_name,
        columns,
        unique=unique,
        **dialect_options,
    )
```

Pass both `postgresql_where=sa.text(where)` and `sqlite_where=sa.text(where)` for partial indexes. Downgrade must remove only this revision's indexes, columns, and four tables in reverse dependency order; it must not drop legacy rows or unrelated schema.

- [ ] **Step 8: Run, lint, and commit the schema foundation**

```powershell
uv run --locked pytest backend/tests/test_agent_workflow_models.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_init.py backend/tests/test_db_schema_operations.py -q
```

```powershell
uv run --locked ruff check backend/app/models backend/migrations/versions/2f6a8b9c0d1e_add_agent_workflow_runtime_foundation.py backend/tests/test_agent_workflow_models.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_init.py backend/tests/test_db_schema_operations.py
```

```powershell
git add backend/app/models backend/migrations/versions/2f6a8b9c0d1e_add_agent_workflow_runtime_foundation.py backend/tests/test_agent_workflow_models.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_init.py backend/tests/test_db_schema_operations.py
git commit -m "feat: add agent workflow persistence schema"
```

---

### Task 4: Add Checkpoint Configuration, Safe DSN, and Strict Serializer

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/agent_runtime/fingerprints.py`
- Create: `backend/app/agent_runtime/checkpointing.py`
- Create: `backend/tests/test_agent_runtime_checkpointing.py`

**Interfaces:**
- Consumes: `Settings.resolved_database_url()`, `JsonPlusSerializer`, `InMemorySaver`, `PostgresSaver`, and psycopg `ConnectionPool`.
- Produces: `fingerprint_secret_bytes`, `CheckpointMode`, `CheckpointReadiness`, `resolve_checkpoint_mode`, `sqlalchemy_url_to_psycopg_dsn`, `build_strict_checkpoint_serializer`, and `build_postgres_pool`.

- [ ] **Step 1: Write RED mode-matrix, DSN, and serializer tests**

Cover exactly these cases:

```python
assert resolve_checkpoint_mode(Settings(_env_file=None, langgraph_review_v2_enabled=False)) == 'disabled'
assert resolve_checkpoint_mode(Settings(_env_file=None, paraworks_demo_mode=True, langgraph_review_v2_enabled=True)) == 'memory'
assert resolve_checkpoint_mode(Settings(_env_file=None, database_url='sqlite://', langgraph_review_v2_enabled=True)) == 'memory'
assert resolve_checkpoint_mode(Settings(_env_file=None, database_url='postgresql+psycopg://user:p%40ss@db/runtime', langgraph_review_v2_enabled=True)) == 'postgres'
```

Assert DSN conversion returns `postgresql://user:p%40ss@db/runtime`, a SQLite URL raises a sanitized `ValueError` that contains neither the password nor input URL, and the pool factory receives:

```python
{
    'kwargs': {'autocommit': True, 'row_factory': dict_row},
    'min_size': 1,
    'max_size': 4,
    'open': False,
}
```

Serialize and deserialize a valid `ReviewGraphState` with the strict serializer. Assert `pickle_fallback is False`, and assert serializing `object()` raises rather than encoding a custom Python object.

Assert `fingerprint_secret_bytes` rejects the local default secret when
`paraworks_env='production'`, rejects an empty key version, and returns the
configured secret bytes plus key version for a non-default production secret.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_checkpointing.py -q
```

Expected: collection fails because `checkpointing.py` does not exist and settings fields are absent.

- [ ] **Step 3: Add explicit runtime settings**

Add to `Settings`:

```python
from pydantic import Field

langgraph_review_v2_enabled: bool = False
langgraph_rag_v2_enabled: bool = False
langgraph_strict_msgpack: bool = False
langgraph_checkpoint_retention_days: int = Field(default=30, ge=1, le=3650)
agent_runtime_fingerprint_secret: str = 'local-development-agent-runtime-fingerprint-secret'
agent_runtime_fingerprint_key_version: str = 'v1'
```

Do not add Slack-dependent or public-backend selection settings in this task.

Add this boundary to `fingerprints.py`:

```python
LOCAL_FINGERPRINT_SECRET = 'local-development-agent-runtime-fingerprint-secret'


def fingerprint_secret_bytes(settings: Settings) -> tuple[bytes, str]:
    secret = settings.agent_runtime_fingerprint_secret
    key_version = settings.agent_runtime_fingerprint_key_version.strip()
    if not secret or not key_version:
        raise ValueError('agent runtime fingerprint configuration is incomplete')
    if settings.paraworks_env == 'production' and secret == LOCAL_FINGERPRINT_SECRET:
        raise ValueError('production requires a dedicated fingerprint secret')
    return secret.encode('utf-8'), key_version
```

- [ ] **Step 4: Implement the safe factories**

In `checkpointing.py`, implement:

```python
CheckpointMode = Literal['disabled', 'memory', 'postgres']


class CheckpointUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointReadiness:
    enabled: bool
    mode: CheckpointMode
    ready: bool
    durable: bool
    checkpoint_store: str
    error_code: str | None = None


def resolve_checkpoint_mode(settings: Settings) -> CheckpointMode:
    if not settings.langgraph_review_v2_enabled:
        return 'disabled'
    try:
        backend = make_url(settings.resolved_database_url()).get_backend_name()
    except (ArgumentError, ValueError):
        raise ValueError('unsupported checkpoint database URL') from None
    if settings.paraworks_demo_mode or backend == 'sqlite':
        return 'memory'
    if backend == 'postgresql':
        return 'postgres'
    raise ValueError('unsupported checkpoint database URL')


def sqlalchemy_url_to_psycopg_dsn(database_url: str) -> str:
    try:
        url = make_url(database_url)
        if url.get_backend_name() != 'postgresql':
            raise ValueError
        return url.set(drivername='postgresql').render_as_string(
            hide_password=False,
        )
    except (ArgumentError, ValueError):
        raise ValueError('unsupported checkpoint database URL') from None


def build_strict_checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=None,
    )


def build_postgres_pool(dsn: str) -> ConnectionPool:
    return ConnectionPool(
        conninfo=dsn,
        kwargs={'autocommit': True, 'row_factory': dict_row},
        min_size=1,
        max_size=4,
        open=False,
    )


def build_postgres_saver(
    pool: ConnectionPool,
    serializer: SerializerProtocol,
) -> PostgresSaver:
    return PostgresSaver(pool, serde=serializer)


def checkpoint_tables_ready(pool: ConnectionPool) -> bool:
    with pool.connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    to_regclass('checkpoint_migrations') IS NOT NULL AS migrations,
                    to_regclass('checkpoints') IS NOT NULL AS checkpoints,
                    to_regclass('checkpoint_blobs') IS NOT NULL AS blobs,
                    to_regclass('checkpoint_writes') IS NOT NULL AS writes,
                    EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'checkpoint_writes'
                          AND column_name = 'task_path'
                    ) AS task_path
                """
            )
            row = cursor.fetchone()
    return bool(row) and all(bool(row[key]) for key in row)
```

Use `sqlalchemy.engine.make_url`; accept only PostgreSQL backends, replace the driver with `postgresql`, and render with `hide_password=False`. Catch parser/backend errors and raise `ValueError('unsupported checkpoint database URL')` without echoing credentials.

- [ ] **Step 5: Run, lint, and commit the configuration boundary**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_checkpointing.py -q
```

```powershell
uv run --locked ruff check backend/app/core/config.py backend/app/agent_runtime/checkpointing.py backend/tests/test_agent_runtime_checkpointing.py
```

```powershell
git add backend/app/core/config.py backend/app/agent_runtime/checkpointing.py backend/tests/test_agent_runtime_checkpointing.py
git commit -m "feat: add langgraph checkpoint configuration"
```

---

### Task 5: Implement Lifespan-Scoped Saver Lifecycle and Readiness

**Files:**
- Modify: `backend/app/agent_runtime/checkpointing.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_agent_runtime_lifespan.py`
- Modify: `backend/tests/test_agent_runtime_checkpointing.py`

**Interfaces:**
- Consumes: Task 4's mode, DSN, serializer, and pool factories.
- Produces: `CheckpointRuntime.start()`, `CheckpointRuntime.close()`, `CheckpointRuntime.readiness`, `CheckpointRuntime.saver`, and `build_checkpoint_runtime(settings)` exposed on `app.state.agent_checkpoint_runtime`.

- [ ] **Step 1: Write RED lifecycle tests with fake pools and savers**

The test doubles must record `open`, `wait`, `check`, `connection`, `close`, and `setup` calls. Prove:

```text
flag off -> no saver, no pool, ready=false, store=none, no error
demo/SQLite -> one InMemorySaver per CheckpointRuntime, ready=true, durable=false
PostgreSQL + strict flag + ready tables -> pool opens once, PostgresSaver is created, ready=true, durable=true
PostgreSQL + missing table -> application runtime remains constructed, ready=false, error=checkpoint_unavailable, no memory fallback
PostgreSQL + strict flag false -> no pool connection, ready=false, error=strict_serializer_required
readiness/startup -> saver.setup call count stays zero
close -> pool closes once and repeated close is safe
```

The table probe checks all five installed-saver tables: `checkpoint_migrations`, `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, and `checkpoint_writes`'s `task_path` column.

- [ ] **Step 2: Run lifecycle tests and verify RED**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_lifespan.py -q
```

Expected: RED because lifecycle behavior is not implemented.

- [ ] **Step 3: Implement `CheckpointRuntime` without startup migration**

Add this public shape:

```python
class CheckpointRuntime:
    def __init__(
        self,
        settings: Settings,
        *,
        pool_factory: Callable[[str], ConnectionPool] = build_postgres_pool,
        saver_factory: Callable[[object, SerializerProtocol], BaseCheckpointSaver] = build_postgres_saver,
        table_probe: Callable[[object], bool] = checkpoint_tables_ready,
    ) -> None:
        self._settings = settings
        self._mode = resolve_checkpoint_mode(settings)
        self._pool_factory = pool_factory
        self._saver_factory = saver_factory
        self._table_probe = table_probe
        self._pool: object | None = None
        self._saver: BaseCheckpointSaver | None = None
        self._started = False
        self._readiness = CheckpointReadiness(
            enabled=self._mode != 'disabled',
            mode=self._mode,
            ready=False,
            durable=False,
            checkpoint_store='none',
        )

    @property
    def saver(self) -> BaseCheckpointSaver | None:
        return self._saver

    @property
    def readiness(self) -> CheckpointReadiness:
        return self._readiness

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        serializer = build_strict_checkpoint_serializer()
        if self._mode == 'disabled':
            return
        if self._mode == 'memory':
            self._saver = InMemorySaver(serde=serializer)
            self._readiness = CheckpointReadiness(
                enabled=True,
                mode='memory',
                ready=True,
                durable=False,
                checkpoint_store='memory',
            )
            return
        if not self._settings.langgraph_strict_msgpack:
            self._readiness = replace(
                self._readiness,
                error_code='strict_serializer_required',
            )
            return
        try:
            dsn = sqlalchemy_url_to_psycopg_dsn(
                self._settings.resolved_database_url()
            )
            self._pool = self._pool_factory(dsn)
            self._pool.open(wait=True)
            self._pool.check()
            if not self._table_probe(self._pool):
                raise CheckpointUnavailableError
            self._saver = self._saver_factory(self._pool, serializer)
            self._readiness = CheckpointReadiness(
                enabled=True,
                mode='postgres',
                ready=True,
                durable=True,
                checkpoint_store='postgres',
            )
        except Exception:
            if self._pool is not None:
                self._pool.close()
            self._pool = None
            self._saver = None
            self._readiness = CheckpointReadiness(
                enabled=True,
                mode='postgres',
                ready=False,
                durable=False,
                checkpoint_store='postgres',
                error_code='checkpoint_unavailable',
            )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
        self._pool = None
        self._saver = None


def build_checkpoint_runtime(settings: Settings) -> CheckpointRuntime:
    return CheckpointRuntime(settings)
```

`start()` is idempotent. Memory mode constructs `InMemorySaver(serde=build_strict_checkpoint_serializer())`. PostgreSQL mode first enforces `langgraph_strict_msgpack`, opens and waits for the pool, probes readiness, then constructs `PostgresSaver(pool, serde=serializer)`. Any PostgreSQL connection/schema error is converted to `checkpoint_unavailable` without including the original exception text in readiness or logs. Close a partially opened pool before returning unready. Never call `setup()` and never construct an in-memory fallback in PostgreSQL mode.

- [ ] **Step 4: Wire lifecycle into FastAPI without changing routes**

Change `create_app` to accept an injectable factory while preserving no-argument callers:

```python
CheckpointRuntimeFactory = Callable[[Settings], CheckpointRuntime]


def create_app(
    *,
    checkpoint_runtime_factory: CheckpointRuntimeFactory = build_checkpoint_runtime,
) -> FastAPI:
    settings = get_settings()
    checkpoint_runtime = checkpoint_runtime_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        checkpoint_runtime.start()
        app.state.agent_checkpoint_runtime = checkpoint_runtime
        try:
            yield
        finally:
            checkpoint_runtime.close()

    app = FastAPI(title='ParaWorks Harness', lifespan=lifespan)
```

Do not add readiness data to `/health` or orchestration routes in Deliverable B. Test that importing `backend.app.main` and calling `create_app()` without entering `TestClient` does not open a pool, while `TestClient` enters and exits the injected fake runtime exactly once.

- [ ] **Step 5: Run regression, lint, and commit lifecycle ownership**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_health.py -q
```

```powershell
uv run --locked ruff check backend/app/agent_runtime/checkpointing.py backend/app/main.py backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_lifespan.py
```

```powershell
git add backend/app/agent_runtime/checkpointing.py backend/app/main.py backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_lifespan.py
git commit -m "feat: manage checkpoint savers in app lifespan"
```

---

### Task 6: Add an Explicit Checkpointer Bootstrap Command

**Files:**
- Create: `backend/app/agent_runtime/bootstrap.py`
- Create: `scripts/bootstrap_langgraph_checkpointer.py`
- Create: `backend/tests/test_agent_runtime_bootstrap.py`
- Modify: `backend/app/agent_runtime/__init__.py`

**Interfaces:**
- Consumes: `AgentRuntimeSchemaVersion`, Task 4's PostgreSQL factories, `PostgresSaver.MIGRATIONS`, and installed package metadata.
- Produces: `bootstrap_langgraph_checkpointer(settings, backup_confirmed, pool_factory, saver_factory, session_factory, now) -> BootstrapResult` and an operator command that is the only application-owned caller of `PostgresSaver.setup()`.

- [ ] **Step 1: Write RED bootstrap tests**

Use fake saver/pool/session factories and assert:

```text
backup_confirmed=false -> raises before opening a pool
non-PostgreSQL URL -> raises sanitized configuration error
strict msgpack=false -> raises before setup
valid invocation -> opens pool, calls setup exactly once, verifies latest migration revision, records schema metadata, closes pool
recorded component -> langgraph_checkpoint
recorded package -> langgraph-checkpoint-postgres
recorded package version -> importlib.metadata.version result
recorded schema revision -> len(PostgresSaver.MIGRATIONS) - 1
returned/printed values -> no DSN, username, password, source content, or provider error text
```

Also retain the Task 5 spy assertion that ordinary startup/readiness calls `setup()` zero times.

- [ ] **Step 2: Run the bootstrap tests and verify RED**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_bootstrap.py -q
```

Expected: collection fails because the bootstrap module does not exist.

- [ ] **Step 3: Implement the bootstrap service**

Use these result and function signatures:

```python
class CheckpointBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class BootstrapResult:
    component: str
    package_name: str
    package_version: str
    schema_revision: int
    applied_at: datetime


def bootstrap_langgraph_checkpointer(
    settings: Settings,
    *,
    backup_confirmed: bool,
    pool_factory: Callable[[str], object] = build_postgres_pool,
    saver_factory: Callable[[object, SerializerProtocol], PostgresSaver] = build_postgres_saver,
    session_factory: Callable[[], Session] = SessionLocal,
    now: Callable[[], datetime] = utc_now,
) -> BootstrapResult:
    if not backup_confirmed:
        raise CheckpointBootstrapError('database backup confirmation is required')
    if not settings.langgraph_strict_msgpack:
        raise CheckpointBootstrapError('strict msgpack is required')
    try:
        dsn = sqlalchemy_url_to_psycopg_dsn(settings.resolved_database_url())
        pool = pool_factory(dsn)
        pool.open(wait=True)
        saver = saver_factory(pool, build_strict_checkpoint_serializer())
        saver.setup()
        expected_revision = len(PostgresSaver.MIGRATIONS) - 1
        with pool.connection() as connection:
            row = connection.execute(
                'SELECT MAX(v) AS v FROM checkpoint_migrations'
            ).fetchone()
        if row is None or row['v'] != expected_revision:
            raise CheckpointBootstrapError('checkpoint schema revision mismatch')
        applied_at = now()
        package_version = version('langgraph-checkpoint-postgres')
        with session_factory() as session:
            record = session.scalar(
                select(AgentRuntimeSchemaVersion).where(
                    AgentRuntimeSchemaVersion.component == 'langgraph_checkpoint'
                )
            )
            if record is None:
                record = AgentRuntimeSchemaVersion(component='langgraph_checkpoint')
                session.add(record)
            record.package_name = 'langgraph-checkpoint-postgres'
            record.package_version = package_version
            record.schema_revision = expected_revision
            record.applied_at = applied_at
            session.commit()
        return BootstrapResult(
            component='langgraph_checkpoint',
            package_name='langgraph-checkpoint-postgres',
            package_version=package_version,
            schema_revision=expected_revision,
            applied_at=applied_at,
        )
    except CheckpointBootstrapError:
        raise
    except Exception:
        raise CheckpointBootstrapError('checkpoint bootstrap failed') from None
    finally:
        if 'pool' in locals():
            pool.close()
```

Define a module-local `utc_now()` that returns `datetime.now(UTC)`; do not
import the model module's private clock helper. The default `SessionLocal` is
valid because the operator command obtains the same cached `Settings` used by
`backend.app.db.session`; tests that pass a different URL must inject their own
session factory.

Require PostgreSQL, `langgraph_strict_msgpack=True`, and `backup_confirmed=True`. Open the pool, call `PostgresSaver.setup()` once, query `checkpoint_migrations` to require the latest revision, then insert-or-update the single `AgentRuntimeSchemaVersion(component='langgraph_checkpoint')` row. Commit only that metadata row and always close the pool. Convert database exceptions to a sanitized `CheckpointBootstrapError` without embedding the original exception or DSN.

- [ ] **Step 4: Implement the thin operator command**

`scripts/bootstrap_langgraph_checkpointer.py` must accept only:

```text
--confirm-backup
```

Without the flag it exits non-zero before connecting. With the flag it calls the service and prints only component, package version, schema revision, and applied timestamp. It must not accept a raw DSN argument; configuration continues through `Settings.resolved_database_url()`.

- [ ] **Step 5: Run, lint, and commit the bootstrap boundary**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_bootstrap.py backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_lifespan.py -q
```

```powershell
uv run --locked ruff check backend/app/agent_runtime/bootstrap.py backend/app/agent_runtime/__init__.py scripts/bootstrap_langgraph_checkpointer.py backend/tests/test_agent_runtime_bootstrap.py
```

```powershell
git add backend/app/agent_runtime/bootstrap.py backend/app/agent_runtime/__init__.py scripts/bootstrap_langgraph_checkpointer.py backend/tests/test_agent_runtime_bootstrap.py
git commit -m "feat: add explicit langgraph checkpoint bootstrap"
```

---

### Task 7: Add Safe Checkpoint Retention Pruning

**Files:**
- Create: `backend/app/agent_runtime/retention.py`
- Create: `scripts/prune_langgraph_checkpoints.py`
- Create: `backend/tests/test_agent_runtime_retention.py`
- Modify: `backend/app/agent_runtime/__init__.py`

**Interfaces:**
- Consumes: `AgentWorkflowThread`, `langgraph_checkpoint_retention_days`, and a ready PostgreSQL checkpoint pool.
- Produces: `prune_expired_checkpoints(settings, session_factory, pool_factory, now, limit) -> CheckpointPruneResult` and a non-interactive operator command.

- [ ] **Step 1: Write RED retention tests with fake application and checkpoint stores**

Seed workflow threads covering every status and age boundary. Assert the prune
selector includes only `completed`, `needs_more_evidence`, `failed`, and
`cancelled` threads whose terminal timestamp is older than
`langgraph_checkpoint_retention_days`. It must exclude `created`, `drafting`,
`checkpoint_pending`, `awaiting_human_review`, `resuming`, and
`checkpoint_failed` even when old.

The fake checkpoint connection must record statements and prove deletion order
is `checkpoint_writes`, `checkpoint_blobs`, then `checkpoints`, all constrained
by the exact selected `checkpoint_thread_id` list. Assert it never deletes from
`checkpoint_migrations`, `agent_workflow_threads`, `audit_logs`, Review Queue,
or knowledge tables. A limit of 100 must select at most 100 oldest terminal
threads.

- [ ] **Step 2: Run the retention tests and verify RED**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_retention.py -q
```

Expected: collection fails because the retention module does not exist.

- [ ] **Step 3: Implement the bounded prune service**

Create:

```python
TERMINAL_PRUNABLE_STATUSES = (
    'completed',
    'needs_more_evidence',
    'failed',
    'cancelled',
)


@dataclass(frozen=True)
class CheckpointPruneResult:
    selected_thread_count: int
    deleted_write_count: int
    deleted_blob_count: int
    deleted_checkpoint_count: int
    cutoff: datetime


def prune_expired_checkpoints(
    settings: Settings,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    pool_factory: Callable[[str], ConnectionPool] = build_postgres_pool,
    now: Callable[[], datetime] = utc_now,
    limit: int = 100,
) -> CheckpointPruneResult:
    if limit < 1 or limit > 1000:
        raise ValueError('checkpoint prune limit must be between 1 and 1000')
    cutoff = now() - timedelta(days=settings.langgraph_checkpoint_retention_days)
    with session_factory() as session:
        terminal_at = func.coalesce(
            AgentWorkflowThread.completed_at,
            AgentWorkflowThread.cancelled_at,
            AgentWorkflowThread.updated_at,
        )
        checkpoint_thread_ids = list(
            session.scalars(
                select(AgentWorkflowThread.checkpoint_thread_id)
                .where(AgentWorkflowThread.status.in_(TERMINAL_PRUNABLE_STATUSES))
                .where(AgentWorkflowThread.checkpoint_store == 'postgres')
                .where(terminal_at < cutoff)
                .order_by(terminal_at, AgentWorkflowThread.thread_id)
                .limit(limit)
            )
        )
    if not checkpoint_thread_ids:
        return CheckpointPruneResult(0, 0, 0, 0, cutoff)
    dsn = sqlalchemy_url_to_psycopg_dsn(settings.resolved_database_url())
    pool = pool_factory(dsn)
    try:
        pool.open(wait=True)
        with pool.connection() as connection:
            deleted_writes = connection.execute(
                'DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)',
                (checkpoint_thread_ids,),
            ).rowcount
            deleted_blobs = connection.execute(
                'DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)',
                (checkpoint_thread_ids,),
            ).rowcount
            deleted_checkpoints = connection.execute(
                'DELETE FROM checkpoints WHERE thread_id = ANY(%s)',
                (checkpoint_thread_ids,),
            ).rowcount
            connection.commit()
        return CheckpointPruneResult(
            selected_thread_count=len(checkpoint_thread_ids),
            deleted_write_count=deleted_writes,
            deleted_blob_count=deleted_blobs,
            deleted_checkpoint_count=deleted_checkpoints,
            cutoff=cutoff,
        )
    finally:
        pool.close()
```

Convert connection failures to a sanitized `CheckpointUnavailableError` from
Task 4. Do not mark threads deleted and do not change application-owned status;
the retained thread record remains the audit/ownership source of truth.
Define a module-local `utc_now()` returning `datetime.now(UTC)` rather than
importing another module's private clock helper.

- [ ] **Step 4: Add the operator command and verify no secret output**

`scripts/prune_langgraph_checkpoints.py` accepts `--limit` with default `100`,
calls the service, and prints only cutoff and row counts. It must reject
SQLite/non-PostgreSQL configuration through the same sanitized DSN boundary and
must not print a DSN, username, password, thread id, source id, or raw exception.

- [ ] **Step 5: Run, lint, and commit retention ownership**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_retention.py -q
```

```powershell
uv run --locked ruff check backend/app/agent_runtime/retention.py backend/app/agent_runtime/__init__.py scripts/prune_langgraph_checkpoints.py backend/tests/test_agent_runtime_retention.py
```

```powershell
git add backend/app/agent_runtime/retention.py backend/app/agent_runtime/__init__.py scripts/prune_langgraph_checkpoints.py backend/tests/test_agent_runtime_retention.py
git commit -m "feat: add bounded checkpoint retention pruning"
```

---

### Task 8: Add Graph Version and Saver-Confirmation Primitives

**Files:**
- Create: `backend/app/agent_runtime/graph_versions.py`
- Create: `backend/app/agent_runtime/checkpoint_execution.py`
- Modify: `backend/app/agent_runtime/__init__.py`
- Create: `backend/tests/test_agent_runtime_graph_versions.py`
- Create: `backend/tests/test_agent_runtime_checkpoint_execution.py`

**Interfaces:**
- Consumes: a compiled LangGraph, Task 5's saver, an immutable application `graph_version`, and a server-issued `checkpoint_thread_id`.
- Produces: `GraphVersionRegistry`, `checkpoint_config`, `require_resumable_checkpoint`, `CheckpointConfirmation`, `invoke_and_confirm_checkpoint`, and typed errors for unavailable versions or unconfirmed saver writes.

- [ ] **Step 1: Write RED registry tests**

Cover:

```python
registry.register('company-memory-review', 'company-memory-review-v2.0', builder_v2)
assert registry.resolve('company-memory-review', 'company-memory-review-v2.0') is builder_v2
```

Assert duplicate registration raises, unknown versions raise `RuntimeVersionUnavailable`, and resolving a thread always uses its immutable stored `graph_version`; the feature flag must not redirect a waiting V2 thread into the legacy builder.

- [ ] **Step 2: Implement the immutable registry**

Create:

```python
GraphBuilder = Callable[[BaseCheckpointSaver], object]


class RuntimeVersionUnavailable(RuntimeError):
    def __init__(self, graph_version: str) -> None:
        super().__init__(f'unsupported graph version: {graph_version}')


@dataclass(frozen=True)
class GraphVersionKey:
    workflow_name: str
    graph_version: str


class GraphVersionRegistry:
    def __init__(self) -> None:
        self._builders: dict[GraphVersionKey, GraphBuilder] = {}

    def register(
        self,
        workflow_name: str,
        graph_version: str,
        builder: GraphBuilder,
    ) -> None:
        if not workflow_name.strip() or not graph_version.strip():
            raise ValueError('workflow_name and graph_version are required')
        key = GraphVersionKey(workflow_name, graph_version)
        if key in self._builders:
            raise ValueError('graph version is already registered')
        self._builders = {**self._builders, key: builder}

    def resolve(self, workflow_name: str, graph_version: str) -> GraphBuilder:
        key = GraphVersionKey(workflow_name, graph_version)
        try:
            return self._builders[key]
        except KeyError:
            raise RuntimeVersionUnavailable(graph_version) from None
```

Validate both names as non-empty, copy the builder map on registration, reject replacement, and expose no mutation method. Do not register a Review V2 business graph in Deliverable B.

- [ ] **Step 3: Write RED checkpoint-confirmation tests with a real in-memory LangGraph**

Build a test-only graph with `StateGraph`, `InMemorySaver(serde=build_strict_checkpoint_serializer())`, and one `interrupt({'kind': 'review_resolution'})` node. Assert:

```text
checkpoint_config -> configurable.thread_id only; no checkpoint_ns
invoke uses durability="sync"
returned __interrupt__ agrees with graph.get_state(config) pending interrupts
saver.get_tuple(config) exists before confirmation returns
saved tuple configurable.checkpoint_ns is absent or root ""
saved checkpoint contains ids/hashes/status only and no question/url/snippet/model_output
missing tuple -> CheckpointConfirmationError
interrupt mismatch -> CheckpointConfirmationError
new InMemorySaver after process restart -> checkpoint_unavailable before resume
```

- [ ] **Step 4: Implement synchronous invocation and confirmation**

Create:

```python
class CheckpointConfirmationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointConfirmation:
    result: dict[str, object]
    checkpoint_id: str
    checkpoint_thread_id: str
    checkpoint_ns: str
    interrupted: bool


def checkpoint_config(checkpoint_thread_id: str) -> dict[str, dict[str, str]]:
    if not checkpoint_thread_id.strip():
        raise ValueError('checkpoint_thread_id is required')
    return {'configurable': {'thread_id': checkpoint_thread_id}}


def require_resumable_checkpoint(
    saver: BaseCheckpointSaver,
    checkpoint_thread_id: str,
) -> dict[str, dict[str, str]]:
    config = checkpoint_config(checkpoint_thread_id)
    if saver.get_tuple(config) is None:
        raise CheckpointUnavailableError('checkpoint_unavailable')
    return config


def invoke_and_confirm_checkpoint(
    *,
    graph: object,
    saver: BaseCheckpointSaver,
    command_or_input: object,
    checkpoint_thread_id: str,
    runtime_context: object,
    expect_interrupt: bool,
) -> CheckpointConfirmation:
    config = checkpoint_config(checkpoint_thread_id)
    result = graph.invoke(
        command_or_input,
        config,
        context=runtime_context,
        durability='sync',
    )
    if not isinstance(result, dict):
        raise CheckpointConfirmationError('graph result must be a mapping')
    saved = saver.get_tuple(config)
    if saved is None:
        raise CheckpointConfirmationError('checkpoint was not persisted')
    saved_config = saved.config.get('configurable', {})
    saved_thread_id = saved_config.get('thread_id')
    checkpoint_id = saved_config.get('checkpoint_id')
    checkpoint_ns = saved_config.get('checkpoint_ns', '')
    if saved_thread_id != checkpoint_thread_id or not checkpoint_id:
        raise CheckpointConfirmationError('checkpoint identity mismatch')
    if checkpoint_ns != '':
        raise CheckpointConfirmationError('top-level checkpoint namespace must be root')
    snapshot = graph.get_state(config)
    returned_interrupt = bool(result.get('__interrupt__'))
    pending_interrupt = any(
        bool(getattr(task, 'interrupts', ())) for task in snapshot.tasks
    )
    if returned_interrupt != pending_interrupt or returned_interrupt != expect_interrupt:
        raise CheckpointConfirmationError('checkpoint interrupt state mismatch')
    return CheckpointConfirmation(
        result=result,
        checkpoint_id=checkpoint_id,
        checkpoint_thread_id=checkpoint_thread_id,
        checkpoint_ns=checkpoint_ns,
        interrupted=returned_interrupt,
    )
```

Call `graph.invoke(command_or_input, config, context=runtime_context, durability='sync')`, then `saver.get_tuple(config)`, then `graph.get_state(config)`. Require the saved tuple's thread id to match, normalize absent `checkpoint_ns` to `''`, reject non-root namespace, and require returned/pending interrupt agreement. This function does not update `AgentWorkflowThread`; Deliverable C may change application status only after it returns successfully.

For the SQLite restart case, pause with saver A, construct a new independent
`InMemorySaver` B, and assert `require_resumable_checkpoint` raises the
sanitized `checkpoint_unavailable` error. Do not imply restart durability when
`CheckpointReadiness.durable` is false.

- [ ] **Step 5: Verify strict tuple contents and reducer replay**

Invoke the test graph twice with replayed completed/error updates and assert reducers do not duplicate entries. Recursively inspect `CheckpointTuple.checkpoint` and `pending_writes`; fail if a key from this set appears anywhere:

```python
{
    'question',
    'objective',
    'source_url',
    'source_snippet',
    'message_body',
    'model_output',
    'provider_error',
    'api_key',
    'oauth_token',
}
```

Also fail if any value is not JSON-safe according to `validate_checkpoint_state`'s recursive primitive policy.

- [ ] **Step 6: Run, lint, and commit the graph primitives**

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py -q
```

```powershell
uv run --locked ruff check backend/app/agent_runtime/graph_versions.py backend/app/agent_runtime/checkpoint_execution.py backend/app/agent_runtime/__init__.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py
```

```powershell
git add backend/app/agent_runtime/graph_versions.py backend/app/agent_runtime/checkpoint_execution.py backend/app/agent_runtime/__init__.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py
git commit -m "feat: add versioned checkpoint execution primitives"
```

---

### Task 9: Prove PostgreSQL Restart Persistence and Close Deliverable B

**Files:**
- Create: `backend/tests/test_agent_runtime_postgres_checkpoint.py`
- Modify: `plan.md`
- Modify: `docs/portfolio-log.md`
- Modify: `docs/superpowers/runbooks/session-handoff.md`

**Interfaces:**
- Consumes: Tasks 2-8 and a dedicated PostgreSQL test database supplied through `PARAWORKS_TEST_POSTGRES_URL`.
- Produces: restart/resume evidence, a non-Slack regression gate, and a documented rollback point authorizing a separate Deliverable C plan.

- [ ] **Step 1: Write the PostgreSQL restart integration test**

The test must skip only when `PARAWORKS_TEST_POSTGRES_URL` is absent. When present, it must require a PostgreSQL URL, run Alembic to head, explicitly bootstrap with a test backup confirmation, and then:

1. create pool/saver A;
2. compile the same test-only interrupt graph with `company-memory-review-v2.0`;
3. invoke with `checkpoint_thread_id='postgres-restart:<uuid>'` and confirm an actual pause;
4. close pool A to simulate process loss;
5. create independent pool/saver B and rebuild the same graph version;
6. resume with `Command(resume={'event': 'review_resolution_checked', 'state_version': 1})` on the identical thread id;
7. assert terminal state, saved tuple progression, and root `checkpoint_ns == ''`;
8. inspect checkpoint records to prove no sensitive keys/content were stored;
9. in `finally`, delete only this test's exact thread id from
   `checkpoint_writes`, `checkpoint_blobs`, and `checkpoints` in that order;
10. close pool B.

Do not use a live source connector, model, or embedding provider.

- [ ] **Step 2: Run the dedicated PostgreSQL gate**

Point the variable at a disposable, access-controlled test database rather than the development or production database:

```powershell
$env:PARAWORKS_TEST_POSTGRES_URL='postgresql+psycopg://paraworks:paraworks@localhost:5432/paraworks_runtime_test'
uv run --locked pytest backend/tests/test_agent_runtime_postgres_checkpoint.py -q
```

Expected: GREEN, with no skip. Deliverable B is not complete if this test is merely skipped.

- [ ] **Step 3: Run the complete Deliverable B focused suite**

```powershell
uv run --locked pytest backend/tests/test_agent_preflight.py backend/tests/test_agent_runtime_state.py backend/tests/test_agent_runtime_fingerprints.py backend/tests/test_agent_workflow_models.py backend/tests/test_agent_runtime_migration.py backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_agent_runtime_bootstrap.py backend/tests/test_agent_runtime_retention.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py backend/tests/test_agent_runtime_postgres_checkpoint.py backend/tests/test_db_init.py backend/tests/test_db_schema_operations.py backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

Expected: all selected tests pass and the PostgreSQL test is not skipped.

- [ ] **Step 4: Run the non-Slack backend regression gate**

Run the complete backend suite while explicitly deselecting only the ten user-deferred Slack cases:

```powershell
uv run --locked pytest backend/tests -q --deselect=backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_runs_real_agent_services --deselect=backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_skips_agents_that_exceed_cost_budget --deselect=backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_uses_cache_when_evidence_is_unchanged --deselect=backend/tests/test_oauth_pkce.py::test_slack_oauth_pkce_generation --deselect=backend/tests/test_oauth_pkce.py::test_slack_callback_with_custom_redirect_uri_and_pkce --deselect=backend/tests/test_oauth_pkce.py::test_api_endpoints_support_redirect_uri --deselect=backend/tests/test_orchestration_api.py::test_company_memory_orchestration_api_runs_agent_services --deselect=backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_company_memory_emits_review_checkpoint_without_paid_calls --deselect=backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_cache_hit_does_not_duplicate_agent_runs_or_review_items --deselect=backend/tests/test_slack_oauth.py::test_slack_sync_endpoint_uses_installed_connection_token_without_exposing_it
```

Expected: all selected tests pass. If any additional test fails, stop and investigate; do not add it to the Slack allowlist without evidence and user approval.

- [ ] **Step 5: Run the full backend suite and compare the deferred baseline**

```powershell
uv run --locked pytest backend/tests -q
```

Expected: exactly the same ten Slack-related test ids listed in Step 4 fail, with every non-Slack test passing. A different failure id, a changed Slack failure symptom caused by this branch, or more than ten failures blocks Deliverable B.

- [ ] **Step 6: Run lock, lint, and diff integrity checks**

```powershell
uv lock --check
```

```powershell
uv run --locked ruff check backend/app/agent_runtime backend/app/models backend/app/core/config.py backend/app/main.py backend/migrations/versions/2f6a8b9c0d1e_add_agent_workflow_runtime_foundation.py scripts/bootstrap_langgraph_checkpointer.py scripts/prune_langgraph_checkpoints.py backend/tests/test_agent_runtime_*.py backend/tests/test_agent_workflow_models.py backend/tests/test_agent_preflight.py
```

```powershell
git diff --check
```

Expected: lock and Ruff pass; diff check has no output.

- [ ] **Step 7: Update source-of-truth documentation only after the gates pass**

Update `plan.md` Milestone 4 to mark Deliverable B complete, name the locked graph version and checkpoint modes, and state that Deliverable C still has no implementation authorization until its separate plan is reviewed.

Add a portfolio entry recording:

```text
- JSON-safe, HMAC-keyed runtime contracts and application-owned workflow schema
- explicit bootstrap/readiness separation with no startup setup call
- SQLite process-local versus PostgreSQL durable mode behavior
- real PostgreSQL pause, pool restart, and same-thread resume evidence
- zero new non-Slack backend failures; ten Slack failures remain visible and deferred
```

Update the handoff with the exact PostgreSQL command/result, migration revision, graph version contract, rollback point, and the statement that no public V2 route, Review Queue transition, RAG cutover, or Slack change was made.

- [ ] **Step 8: Commit documentation and confirm the branch**

```powershell
git add plan.md docs/portfolio-log.md docs/superpowers/runbooks/session-handoff.md
git commit -m "docs: record runtime checkpoint foundation"
```

```powershell
git status --short
```

Expected: no output.

Deliverable B is complete only when the dedicated PostgreSQL restart test ran without skip, the non-Slack gate is fully green, and the full backend suite differs from green only by the exact ten Slack failures already deferred by the user. The next step is a separate Deliverable C implementation plan, not direct Review HITL or GraphRAG coding.

## Deferred Slack Restoration Phase

This section records sequencing, not authorization to implement Slack changes.

When Deliverables B-D are green on Gmail, Drive, Calendar, approved knowledge, and deterministic fixtures, prepare a separate Slack recovery design and ask the user to choose among:

1. **Deterministic local Slack archive/fixture (recommended first):** reconstruct representative channel, thread, permission, decision, todo, and timeline events through the existing `SourceEvent` contract. This restores repeatable regression/demo coverage without depending on a live workspace.
2. **New seeded Slack workspace:** create a disposable workspace/app/channel dataset and validate OAuth, incremental sync, threads, and live connector behavior. This gives the strongest connector proof but requires new external ownership, credentials, and ongoing maintenance.
3. **Alternate chat connector:** implement another provider only through `SourceEvent` and `ConnectorManifest`. This is the largest scope and should be chosen only if Slack is no longer a product requirement.

Do not mix these choices into Deliverable B, C, or D. OAuth PKCE and fake-client contract failures are Slack-related code regressions rather than missing-data failures, but the user's priority decision still defers them to this final phase.
