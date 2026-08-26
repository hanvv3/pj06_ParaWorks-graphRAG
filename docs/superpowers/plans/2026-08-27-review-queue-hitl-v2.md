# Review Queue HITL V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved non-Slack Review Queue HITL V2 flow so a user can sync Gmail, Drive, or Calendar evidence, inspect the deterministic cost preview, explicitly create review candidates, resolve them in Review, and explicitly resume the same durable LangGraph thread without duplicate candidates or trusted-knowledge effects.

**Architecture:** Keep PostgreSQL application rows authoritative and checkpoint only bounded identifiers, HMACs, counts, phases, and allowlisted error codes. Canonical source/version validation and scope-level batch ownership happen before graph execution; a short-lease draft service calls Registry-backed real LangChain adapters outside database transactions; a real LangGraph 1.2 graph pauses with `interrupt()` and resumes with `Command(resume=...)`; every Review mutation passes through one locked transition/promotion service. The V2 UI stays inside the existing Integrations and Review screens, and V1 remains a truthful metadata-only rollback path.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLAlchemy 2, PostgreSQL, SQLite smoke mode, LangChain 1.3.17, LangGraph 1.2.11, `langgraph-checkpoint-postgres` 3.1.2, pytest, Ruff, Next.js 16, React 19, TypeScript, Tailwind CSS, Playwright

**Spec:** `docs/superpowers/specs/2026-08-27-review-queue-hitl-v2-design.md`

## Global Constraints

- Keep `langchain>=1.3.17,<1.4.0`, `langgraph>=1.2.11,<1.3.0`, `langchain-openai>=1.6.0,<1.7.0`, `langchain-google-genai>=4.3.5,<4.4.0`, and `langgraph-checkpoint-postgres>=3.1.2,<3.2.0` unchanged.
- Use the real LangChain adapters already present in `mail_document_agent` and `memory_extraction_agent` for configured production model calls. Deterministic models are permitted only for local/demo mode, tests, and dry-run estimation. Do not replace orchestration or extraction with new hard-coded model-call functions.
- Use the real LangGraph 1.2 `StateGraph`, `interrupt()`, and same-checkpoint-thread `Command(resume=...)`. Do not emulate pause/resume with response metadata or application-only status flags.
- Do not add an Alembic migration, batch/outbox table, queue, broker, CDC producer, streaming consumer, or source-event projection. If implementation proves that a new schema/public field is necessary, stop at the human gate before changing it.
- Keep Slack source ingestion, Slack agents, Slack-specific fixtures/expectations, and the ten known deferred Slack failures untouched. Shared orchestration/integration test files may change only their V1-truthfulness or non-Slack V2 assertions. Slack remains the final recovery phase because the former live source is unavailable.
- Keep RAG execution, `/ask`, `/search`, assistant retrieval, Neo4j, GraphRAG projection, Knowledge Map, and vector indexing out of Deliverable C.
- Keep `LANGGRAPH_REVIEW_V2_ENABLED=false` by default. New V2 dry-run/run calls are unavailable while disabled; registered immutable V2 builders and existing-thread status/cancel behavior remain independent of the legacy graph.
- Never silently downgrade PostgreSQL checkpointing to `InMemorySaver`. SQLite/demo memory mode is explicitly non-durable and process-local.
- Keep the graph input to `workflow_thread_id` only and reuse Deliverable B's `ReviewGraphState`. No source content, URL, snippet, objective, prompt, model output, provider exception, credential, ORM object, session, client, or registry may enter a checkpoint.
- The V2 summary API must never expose source refs, source text, ReviewItem ids, checkpoint ids, raw exceptions, or credentials. Review evidence remains accessible only through the existing permission-aware Review API.
- Use exact `DemoUser.permission_levels`/`PermissionContext.allowed_permission_levels`; never infer accessible levels from a role name.
- Revalidate source version and permission at initial request, before candidate persistence, at every Review transition, and before resume.
- Candidate generation must preserve source links, snippets, identifiers, confidence, permission, and uncertainty. Missing evidence creates no ReviewItem.
- Use `AgentRegistry` manifest names exactly: `mail_document_agent`, `timeline_agent`, `history_agent`, `decision_record_agent`, `todo_agent`. Reject aliases such as `memory_extraction_agent`.
- Apply TDD to every behavior change: run the named focused test and observe its expected failure before production edits, implement the smallest passing slice, run the green test, lint touched Python, and commit only that slice.
- Tests must use fake/deterministic model and connector clients and must not call live Slack, Google, OAuth, OpenAI, Gemini, embedding, or other provider APIs.
- Never hold a database row/advisory lock while invoking a model. Use a short lease transaction, execute the adapter outside the transaction, then revalidate lease, state, version, permission, and cancellation before writing effects.
- Keep the exact source-version batch owner stable across all statuses, including zero-effect `failed` and `cancelled`. Only `checkpoint_failed` permits same-thread repair.
- Treat approval, rejection, and needs-more-evidence as explicit Review operations. They must never auto-resume the graph.
- Preserve user/teammate changes and do not broaden refactors beyond the files named in the active task.

## Human-Gated Public Contract Frozen by Approval of This Plan

Plan approval freezes the following additive response contract. Any implementation-time change to these fields, Review trust boundaries, permission rules, token-budget policy, duplicate-resolution behavior, or schema requires another human decision before coding continues.

### Constants

```python
COMPANY_MEMORY_REVIEW_WORKFLOW = 'company-memory-review'
COMPANY_MEMORY_REVIEW_GRAPH_VERSION = 'company-memory-review-v2.0'
COMPANY_MEMORY_SELECTION_POLICY_VERSION = 'company-memory-review-selection:v1'
COMPANY_MEMORY_INPUT_SCHEMA_VERSION = 'review-source-versions:v1'

DEFAULT_REVIEW_AGENT_NAMES = (
    'mail_document_agent',
    'timeline_agent',
    'history_agent',
    'decision_record_agent',
    'todo_agent',
)
```

### Request and source-reference shapes

```python
class ReviewWorkflowSourceRef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_type: Literal['gmail', 'gmail_attachment', 'drive', 'calendar']
    source_id: str = Field(min_length=1, max_length=255)
    version_or_signature: str = Field(min_length=1, max_length=128)


class ReviewWorkflowRunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_refs: list[ReviewWorkflowSourceRef] = Field(min_length=1, max_length=500)
    agent_names: list[str] = Field(min_length=1, max_length=5)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)
```

The backend sorts and deduplicates refs by `(source_type, source_id, version_or_signature)` and agents by the fixed default order. Unknown fields, raw content, arbitrary questions/objectives/prompts, duplicate agent names, group aliases, and unsupported agents are 400.

### Diagnostic, dry-run, and status shapes

```python
BudgetStatus = Literal['within_budget', 'cached', 'no_input', 'over_budget']
CheckpointMode = Literal['disabled', 'memory', 'postgres']
ReviewItemResolutionStatus = Literal[
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
]
ReviewWorkflowErrorCode = Literal[
    'invalid_input',
    'not_found',
    'idempotency_key_reused',
    'evidence_changed',
    'permission_denied',
    'checkpoint_unavailable',
    'checkpoint_failed',
    'review_unresolved',
    'runtime_version_unavailable',
    'model_unavailable',
    'budget_exceeded',
    'concurrent_resume',
    'invalid_state_transition',
]


class ReviewWorkflowDiagnosticResponse(BaseModel):
    enabled: bool
    available: bool
    checkpoint_mode: CheckpointMode
    durable: bool
    graph_version: str
    default_agent_names: list[str]
    error_code: ReviewWorkflowErrorCode | None


class ReviewWorkflowDryRunResponse(BaseModel):
    workflow_name: Literal['company-memory-review']
    graph_version: Literal['company-memory-review-v2.0']
    source_count: int
    agent_names: list[str]
    selection_policy_version: Literal['company-memory-review-selection:v1']
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    budget_limit_usd: float | None
    budget_status: BudgetStatus
    cache_hit: bool
    requires_explicit_run: Literal[True]


class ReviewWorkflowStatusResponse(BaseModel):
    thread_id: str
    status: Literal[
        'created',
        'drafting',
        'checkpoint_pending',
        'awaiting_human_review',
        'resuming',
        'completed',
        'needs_more_evidence',
        'checkpoint_failed',
        'failed',
        'cancelled',
    ]
    review_item_count: int
    review_status_counts: dict[ReviewItemResolutionStatus, int]
    durable: bool
    graph_version: str
    review_resolution_ready: bool
    checkpoint_resumable: bool
    resume_allowed: bool
    retry_allowed: bool
    created_at: datetime
    updated_at: datetime
    error_code: ReviewWorkflowErrorCode | None
    resume_error_code: ReviewWorkflowErrorCode | None
```

Dry-run budget aggregation is deterministic: `over_budget` wins; if every selected agent has no input, return `no_input`; if every runnable result is a cache hit, return `cached`; otherwise return `within_budget`. `estimated_*` and cost are sums, and `budget_limit_usd` is the configured whole-run cap. `cache_hit=true` only when at least one agent has a cache hit and every other selected agent is cached or no-input; an all-no-input preview returns `cache_hit=false`. Dry-run performs no provider call and no write.

The API returns bounded errors as `{"detail": {"code": "<allowlisted-code>"}}`. The V2 route maps invalid shape/agent to 400, hidden or unauthorized resource to 404 `not_found`, version/idempotency/state/budget conflicts to 409, and unavailable checkpoint/model infrastructure to 503. Internal readiness values such as `strict_serializer_required` are mapped to public `checkpoint_unavailable`. It never places raw exception text in `detail`.

`GET /api/v1/orchestration/v2/company-memory` is always authenticated 200. New dry-run/run is 404 while disabled, 503 when enabled but unavailable, 202 only after a candidate-bearing interrupt is synchronously confirmed, and 200 for no-candidate terminal execution or an existing terminal replay.

### Review approval additions

Single approval preserves current fields and adds:

```json
{
  "replayed": false,
  "promotion": {
    "target_type": "decision_record",
    "created_record_ids": [42],
    "created_timeline_event_ids": [81]
  }
}
```

`promotion` is nullable only for a pre-V2 approved legacy row that has no `source_review_item_id` provenance. Such a replay returns `replayed=true`, `promotion=null`, creates nothing, and performs no heuristic record lookup. V2-created approvals must always return their canonical provenance ids. Bulk responses preserve existing approved/rejected/failed/skipped collections and add a `replayed_items` collection using the same item-level promotion metadata. A V2 workflow summary never returns these ids.

## File and Interface Map

### Existing foundations to reuse

| Responsibility | Existing source |
|---|---|
| Workflow rows and creator-only client key | `backend/app/models/agent_workflows.py` |
| AgentRun/ReviewItem effect and candidate keys | `backend/app/models/agent_runs.py`, `backend/app/models/review.py` |
| Knowledge provenance partial unique indexes | `backend/app/models/knowledge.py` |
| Safe graph state and allowlisted error/status values | `backend/app/agent_runtime/state.py` |
| HMAC canonical serialization | `backend/app/agent_runtime/fingerprints.py` |
| Checkpoint runtime/readiness | `backend/app/agent_runtime/checkpointing.py` |
| Synchronous interrupt confirmation | `backend/app/agent_runtime/checkpoint_execution.py` |
| Immutable graph builder registry | `backend/app/agent_runtime/graph_versions.py` |
| Manifest registry/contracts | `backend/app/agent_runtime/registry.py`, `backend/app/agent_runtime/contracts.py` |
| Real LangChain model adapters | `backend/app/agents/mail_document_agent/llm.py`, `backend/app/agents/memory_extraction_agent/langchain_adapter.py` |
| Existing Review and promotion behavior to centralize | `backend/app/api/v1/review.py`, `backend/app/knowledge/promotion.py` |
| Connector ingestion and async SyncJob projection | `backend/app/ingestion/sync.py`, `backend/app/ingestion/service.py`, `backend/app/api/v1/integrations.py` |
| Two-screen UI | `frontend/src/app/integrations/page.tsx`, `frontend/src/app/review/page.tsx` |

### New focused modules

| New file | Sole responsibility |
|---|---|
| `backend/app/schemas/review_workflow.py` | Strict public V2 request/response models and constants |
| `backend/app/ingestion/source_versions.py` | Canonical `SourceVersionRef` serialization, waterline markers, async job ref recovery |
| `backend/app/agent_runtime/canonical_sources.py` | Permission/version-aware canonical resolution and evidence-ref revalidation |
| `backend/app/agent_runtime/review_v2_preflight.py` | Normalization, HMACs, creator idempotency, shared batch lock/create/reuse |
| `backend/app/agent_runtime/model_router.py` | Production LangChain/deterministic-demo model construction behind replaceable settings |
| `backend/app/agent_runtime/review_v2_agents.py` | Exact manifest/adaptor catalog and cost preflight |
| `backend/app/agent_runtime/review_v2_drafting.py` | Lease, provider-outside-transaction execution, effect/candidate insert-or-return |
| `backend/app/review/transitions.py` | Locked Review state machine and exactly-once promotion transaction |
| `backend/app/agent_runtime/review_v2_graph.py` | Actual LangGraph topology, interrupt, and DB-resolution routing |
| `backend/app/agent_runtime/review_v2_service.py` | Start/status/resume/cancel/repair/reconciliation and authorization |
| `backend/app/api/v1/orchestration_v2.py` | Validation/authentication and bounded HTTP mapping only |
| `frontend/src/lib/api/reviewWorkflow.ts` | Typed V2 route wrappers |
| `frontend/src/app/integrations/ReviewCandidateLaunchPanel.tsx` | Cost preview and explicit launch in the sync completion modal |
| `frontend/src/app/review/ReviewWorkflowContextPanel.tsx` | Filtered workflow progress and explicit completion/retry action |

There is deliberately no new migration file and no Slack, RAG, Neo4j, streaming, or Agent Runs navigation file in this map.

## Spec Coverage and Dependency Map

| Spec concern | Owning tasks |
|---|---|
| Legacy truthfulness and exact public output | 1, 7 |
| Canonical refs, permission/version checks, shared scope batch, creator retry key | 2 |
| Async ref recovery, V1/V2 waterline, legacy suppression | 3 |
| Review state machine, permission recheck, exactly-once promotion | 4 |
| Real Registry adapters, LangChain model path, cost/cache, short leases | 5 |
| Actual interrupt, same-thread resume, PostgreSQL-authoritative resolution | 6 |
| Lifecycle API matrix, live projections, repair/cancel/authorization | 7 |
| Typed frontend contract and transport | 8 |
| Integrations cost/launch UX | 9 |
| Review filtering/progress/explicit resume UX | 10 |
| PostgreSQL restart/concurrency, regressions, rollout evidence | 11 |

Tasks 1-4 establish contracts and business idempotency before graph execution. Task 5 establishes side-effect-safe drafting. Task 6 adds the graph. Task 7 exposes it. Tasks 8-10 add the two-screen UX. Task 11 is the release gate. Each task is a separate review checkpoint and commit.

---

### Task 1: Freeze V2 Schemas and Make Legacy Metadata Truthful

**Files:**
- Create: `backend/app/schemas/review_workflow.py`
- Modify: `backend/app/api/v1/orchestration.py`
- Modify: `backend/app/agent_runtime/company_memory.py`
- Create: `backend/tests/test_review_v2_schemas.py`
- Create: `backend/tests/test_legacy_company_memory_metadata.py`
- Test: `backend/tests/test_orchestration_api.py`

**Interfaces:**
- Produces the exact models and constants in the human-gated contract above.
- Corrects V1 cost-policy and run-output metadata: `hitl_checkpointing=false`, `checkpoint_store='none'`, `review_boundary='metadata_only'`, and a non-resumable metadata checkpoint description.
- Does not add V2 routes, write workflow rows, or alter the legacy graph.

- [ ] **Step 1: Write strict schema tests and a legacy truthfulness regression**

Add tests that prove:

```python
def test_review_workflow_request_rejects_raw_prompt_and_group_alias() -> None: ...
def test_review_workflow_request_accepts_all_four_canonical_source_types() -> None: ...
def test_review_workflow_response_schema_excludes_internal_ids_and_evidence() -> None: ...
def test_legacy_company_memory_status_is_metadata_only(client) -> None: ...
def test_legacy_run_describes_review_queue_metadata_without_pause_or_resume(db_session) -> None: ...
```

The strict request test passes an extra `prompt` and then `memory_extraction_agent`; both must fail. The response test serializes every public response model and asserts that none of `review_item_ids`, `checkpoint_thread_id`, `source_refs`, `source_snippets`, or `exception` appears.

- [ ] **Step 2: Run the RED tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_schemas.py backend/tests/test_legacy_company_memory_metadata.py backend/tests/test_orchestration_api.py::test_legacy_company_memory_status_is_metadata_only -q
```

Expected: FAIL because the production schema module does not exist and V1 currently advertises a resumable human-approval checkpoint.

- [ ] **Step 3: Add the strict Pydantic models and constants**

Implement the frozen contract with `ConfigDict(extra='forbid')`, bounded strings/lists, and validators that normalize whitespace but reject duplicate/unknown agent names. Provide one normalization helper:

```python
def normalize_agent_names(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(value.strip() for value in values)
    if len(requested) != len(set(requested)):
        raise ValueError('duplicate agent name')
    unknown = set(requested) - set(DEFAULT_REVIEW_AGENT_NAMES)
    if unknown:
        raise ValueError('unsupported agent name')
    return tuple(name for name in DEFAULT_REVIEW_AGENT_NAMES if name in requested)
```

Do not include internal checkpoint/evidence/item fields on any response model.

- [ ] **Step 4: Correct the legacy status metadata without changing behavior**

Change `_cost_policy_response()` in `backend/app/api/v1/orchestration.py` so all three legacy endpoints return:

```python
{
    **existing_fields,
    'hitl_checkpointing': False,
    'checkpoint_store': 'none',
    'review_boundary': 'metadata_only',
}
```

In `company_memory.py`, change the legacy run output to `review_boundary='metadata_only'`. Preserve the `hitl_checkpoint` object and its keys for response compatibility, but make its values truthful:

```python
{
    'checkpoint_type': 'review_queue_metadata',
    'status': 'metadata_only',
    'review_item_ids': review_item_ids,
    'resume_from_node': None,
    'resume_policy': 'not_resumable',
    'required_review_statuses': ['approved', 'rejected', 'needs_more_evidence'],
    'trusted_knowledge_requires_approval': True,
    'paid_llm_calls': False,
}
```

This is still useful Review Queue metadata, but it must not claim that execution paused or can resume. Do not rename the route, remove the object, alter the legacy linear execution, or route V1 through the new graph.

- [ ] **Step 5: Run the focused GREEN tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_schemas.py backend/tests/test_legacy_company_memory_metadata.py backend/tests/test_orchestration_api.py -q
```

Expected: all selected tests pass and existing V1 payload assertions remain compatible after the additive `review_boundary` field.

- [ ] **Step 6: Lint and commit**

```powershell
uv run --locked ruff check backend/app/schemas/review_workflow.py backend/app/api/v1/orchestration.py backend/app/agent_runtime/company_memory.py backend/tests/test_review_v2_schemas.py backend/tests/test_legacy_company_memory_metadata.py backend/tests/test_orchestration_api.py
git diff --check
git add backend/app/schemas/review_workflow.py backend/app/api/v1/orchestration.py backend/app/agent_runtime/company_memory.py backend/tests/test_review_v2_schemas.py backend/tests/test_legacy_company_memory_metadata.py backend/tests/test_orchestration_api.py
git commit -m "feat: define review workflow v2 contracts"
```

---

### Task 2: Resolve Canonical Evidence and Create or Reuse One Scoped Workflow Thread

**Files:**
- Create: `backend/app/ingestion/source_versions.py`
- Create: `backend/app/agent_runtime/canonical_sources.py`
- Create: `backend/app/agent_runtime/review_v2_preflight.py`
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/agent_runtime/state.py`
- Create: `backend/tests/test_review_v2_canonical_sources.py`
- Create: `backend/tests/test_review_v2_preflight.py`
- Test: `backend/tests/test_agent_runtime_state.py`
- Test: `backend/tests/test_agent_workflow_models.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SourceVersionRef:
    source_type: Literal['gmail', 'gmail_attachment', 'drive', 'calendar']
    source_id: str
    version_or_signature: str


@dataclass(frozen=True)
class ResolvedSourceVersion:
    source_type: str
    canonical_table: str
    canonical_row_id: int
    document_version_id: int | None
    external_revision: str | None
    content_signature: str
    permission_level: str
    content_fingerprint: str


@dataclass(frozen=True)
class PreparedReviewRequest:
    source_refs: tuple[ResolvedSourceVersion, ...]
    agent_names: tuple[str, ...]
    input_hash: str
    evidence_version_hash: str
    selection_policy_version: str


@dataclass(frozen=True)
class WorkflowPreflightResult:
    thread: AgentWorkflowThread
    created: bool
    shared_reuse: bool


def prepare_review_request(
    db: Session,
    *,
    request: ReviewWorkflowRunRequest,
    actor: DemoUser,
    registry: AgentRegistry,
    settings: Settings,
) -> PreparedReviewRequest: ...


def create_or_reuse_review_thread(
    db: Session,
    *,
    prepared: PreparedReviewRequest,
    request: ReviewWorkflowRunRequest,
    actor: DemoUser,
    settings: Settings,
) -> WorkflowPreflightResult: ...
```

Add `agent_runtime_security_scope_id: str = 'default'` to `Settings`. It is server-derived deployment/company scope and never accepted from the request.

- [ ] **Step 1: Write canonical mapping, permission, HMAC, idempotency, and concurrency tests**

Cover at minimum:

```python
def test_canonical_refs_sort_dedupe_and_require_prefixed_source_id(db_session) -> None: ...
def test_canonical_ref_rejects_source_type_mismatch(db_session) -> None: ...
def test_hidden_source_is_not_distinguishable_from_missing(db_session) -> None: ...
def test_changed_source_signature_raises_evidence_changed(db_session) -> None: ...
def test_parsed_drive_ref_rechecks_current_document_version_and_parser_revision(db_session) -> None: ...
def test_same_scope_exact_batch_reuses_thread_across_authorized_owners(db_session) -> None: ...
def test_cross_owner_shared_reuse_does_not_bind_caller_client_key(db_session) -> None: ...
def test_same_owner_client_key_mismatch_wins_before_shared_lookup(db_session) -> None: ...
def test_different_scope_creates_independent_thread(db_session) -> None: ...
def test_cancelled_and_failed_zero_effect_threads_remain_stable_batch_owners(db_session) -> None: ...
def test_group_alias_and_unregistered_manifest_are_rejected(db_session) -> None: ...
def test_postgres_preflight_uses_transaction_scoped_advisory_lock(fake_postgres_session) -> None: ...
```

Add a SQLite test that injects two launches through a `ThreadPoolExecutor`; both must observe one workflow thread under the process lock. The fake PostgreSQL session test must assert the exact `pg_advisory_xact_lock` statement and that lookup/create happens inside that transaction. Task 11 owns the non-skippable real PostgreSQL concurrency proof so Task 2 cannot appear green merely because a database URL is absent.

- [ ] **Step 2: Run the RED contract tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_canonical_sources.py backend/tests/test_review_v2_preflight.py -q
```

Expected: FAIL because canonical resolution and scoped preflight modules do not exist.

- [ ] **Step 3: Implement canonical source/version resolution**

In `canonical_sources.py`:

- Query only the exact prefixed `Source.source_id` and require the stored/requested canonical type mapping.
- Check `source.permission_level in actor.permission_levels` before revealing whether the row exists.
- Compare `version_or_signature` to the current `Source.raw_metadata['content_signature']`.
- For parsed documents, resolve current `DocumentVersion`/`DocumentParserRun` and include their version/revision/signature in `content_fingerprint`.
- Produce `AgentWorkflowEvidenceRef` fields without persisting URLs, snippets, or content.
- Return 400-class typed errors for malformed refs, a hidden/not-found typed error for missing or invisible refs, and `evidence_changed` for current-version mismatch.

Use `build_keyed_fingerprint()` over canonical JSON for evidence and input HMACs. The batch/input HMAC contains exactly `security_scope_id`, workflow name, immutable graph version, evidence-version HMAC, normalized ordered agent names, and selection-policy version; it excludes `owner_subject_id` and `client_request_id`. Never log the canonical ref values.

- [ ] **Step 4: Implement creator-first idempotency and scope-level ownership without schema changes**

Use the existing `AgentWorkflowThread.input_hash` as the exact batch HMAC and compare every stored request/evidence field before declaring a match. The PostgreSQL critical section is:

```python
db.execute(
    text('SELECT pg_advisory_xact_lock(:key)'),
    {'key': advisory_key_from_hmac(prepared.input_hash)},
)
```

SQLite/demo uses one module-level `threading.RLock`; document that it guarantees only single-process behavior. Inside either lock:

1. Look up the current owner's `client_request_id`; exact match replays, mismatch raises `idempotency_key_reused`.
2. Repeat that creator-key lookup after acquiring the batch lock.
3. Query existing thread by scope/workflow/graph/input/evidence hashes, join its request, then compare full normalized agent names, policy, and ordered evidence refs.
4. Return the existing thread for every status/effect count if the actor can still see every bound ref; otherwise return the hidden/not-found error.
5. Create `AgentWorkflowThread`, its single request row, and ordered evidence-ref rows only when no exact thread exists.
6. Store `client_request_id` only when this actor creates the thread. Never create a cross-owner alias/reservation.

No migration, new unique index, batch row, or audit source-id list is allowed.

- [ ] **Step 5: Narrow runtime context types and verify checkpoint safety**

Replace `object` placeholders in `ReviewRuntimeContext` with small Protocols imported only under `TYPE_CHECKING` or defined without ORM payload fields. Keep the exact state keys unchanged. Add `cancelled` only to lifecycle/service types, not to `ReviewGraphState.review_status_counts`; update allowlisted graph error codes only when a named C error is actually stored.

```powershell
uv run --locked pytest backend/tests/test_agent_runtime_state.py backend/tests/test_agent_workflow_models.py -q
```

Expected: all existing safe-state/schema tests pass; no state key or database column changes.

- [ ] **Step 6: Run GREEN canonical, process-lock, and advisory-lock contract tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_canonical_sources.py backend/tests/test_review_v2_preflight.py -q
```

Expected: all selected tests pass. SQLite proves single-process serialization and the fake PostgreSQL session proves transaction-scoped advisory-lock ordering. Task 11 supplies the real PostgreSQL race evidence.

- [ ] **Step 7: Lint and commit**

```powershell
uv run --locked ruff check backend/app/ingestion/source_versions.py backend/app/agent_runtime/canonical_sources.py backend/app/agent_runtime/review_v2_preflight.py backend/app/core/config.py backend/app/agent_runtime/state.py backend/tests/test_review_v2_canonical_sources.py backend/tests/test_review_v2_preflight.py backend/tests/test_agent_runtime_state.py backend/tests/test_agent_workflow_models.py
git diff --check
git add backend/app/ingestion/source_versions.py backend/app/agent_runtime/canonical_sources.py backend/app/agent_runtime/review_v2_preflight.py backend/app/core/config.py backend/app/agent_runtime/state.py backend/tests/test_review_v2_canonical_sources.py backend/tests/test_review_v2_preflight.py backend/tests/test_agent_runtime_state.py backend/tests/test_agent_workflow_models.py
git commit -m "feat: add canonical review workflow preflight"
```

---

### Task 3: Add Canonical Sync Refs and Enforce the V1/V2 Waterline

**Files:**
- Modify: `backend/app/ingestion/service.py`
- Modify: `backend/app/ingestion/sync.py`
- Modify: `backend/app/ingestion/source_versions.py`
- Modify: `backend/app/api/v1/integrations.py`
- Test: `backend/tests/test_connector_ingestion_contract.py`
- Create: `backend/tests/test_review_v2_waterline.py`
- Modify: `backend/tests/test_mail_document_agent_api.py`
- Modify: `backend/tests/test_integration_runtime_status.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class IngestionResult:
    created_review_items: int
    changed_source_ids: list[str]
    changed_source_refs: list[SourceVersionRef]


@dataclass(frozen=True)
class ConnectorSyncResult:
    job_id: str
    connector_type: str
    status: str
    fetched_events: int
    created_review_items: int
    skipped_events: int
    changed_source_ids: list[str]
    changed_source_refs: list[SourceVersionRef]
    parser_status_counts: dict[str, int]
```

`changed_source_ids` remains additive-compatible. Authenticated responses add serialized `changed_source_refs`; V2 audit metadata uses only `changed_source_count` and a keyed batch HMAC.

- [ ] **Step 1: Write sync/waterline/recovery regressions**

Cover:

```python
def test_sync_returns_canonical_refs_after_ingestion_commit(db_session) -> None: ...
def test_async_status_recovers_only_refs_marked_by_latest_job(db_session) -> None: ...
def test_async_status_omits_refs_hidden_from_current_actor(db_session) -> None: ...
def test_v2_mode_marks_explicit_waterline_and_suppresses_legacy_candidates(client) -> None: ...
def test_v2_mode_no_change_recovery_never_runs_legacy_candidates(client) -> None: ...
def test_legacy_success_marks_current_signature_legacy_inline(client) -> None: ...
def test_pre_waterline_or_legacy_marker_is_rejected_by_v2_preflight(db_session) -> None: ...
def test_rollback_v1_does_not_process_batch_with_existing_v2_thread(client) -> None: ...
def test_slack_sync_behavior_is_unchanged_when_v2_flag_changes(client) -> None: ...
```

The Slack regression may use a fake connector and must only assert unchanged branching; do not edit Slack implementation/test fixtures or call Slack.

- [ ] **Step 2: Run the RED tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_waterline.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_mail_document_agent_api.py backend/tests/test_integration_runtime_status.py -q
```

Expected: FAIL because sync results contain only ids, async status cannot reconstruct the completed job's refs, and the integration route always calls the legacy bridge.

- [ ] **Step 3: Produce refs only after canonical rows are committed**

After `persist_parsed_document()` and the ingestion commit, reload the changed `Source` rows and call the source-version helper. Never derive refs from connector events. Serialize only refs visible to the authenticated user at the API boundary.

For a V2-mode Gmail/Drive/Calendar sync, update each changed source's mutable metadata in a committed application transaction:

```python
source.raw_metadata = {
    **(source.raw_metadata or {}),
    'last_changed_sync_job_id': job.job_id,
    'review_batch_mode': 'v2_explicit',
    'review_batch_signature': current_content_signature(source),
}
```

For V1, write `review_batch_mode='legacy_inline'` and the same current signature only after the legacy candidate transaction succeeds. If it fails, do not advance the marker.

- [ ] **Step 4: Recover async refs from `Source.raw_metadata` and current permission**

Change `_sync_job_response` to receive `db` and `user`, query Sources whose `last_changed_sync_job_id` equals the completed job, sort/dedupe their current refs, and filter them using the exact actor permission levels. The initial queued response returns `changed_source_refs=[]`.

Do not store refs in `SyncJob.message`, `AuditLog`, a new table, or a new column.

- [ ] **Step 5: Enforce one candidate-generation mode in the backend**

For Gmail/Drive/Calendar only:

- V2 enabled: return refs and never call `_run_connector_agent_review` or project-assignment generation, including the no-change recovery path.
- V2 disabled: before legacy generation, ask preflight ownership logic whether the exact default-policy batch already has a V2 thread. Suppress V1 when it does; otherwise run current legacy generation and then mark `legacy_inline` on success.
- Slack: retain the current branch regardless of V2 setting.

V2 preflight accepts a new thread only when every current source marker is `v2_explicit` and its marker signature matches the requested/current signature. Legacy or pre-waterline evidence raises the existing `evidence_changed` code with no write.

- [ ] **Step 6: Run GREEN sync and compatibility tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_waterline.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_mail_document_agent_api.py backend/tests/test_integration_runtime_status.py -q
```

Expected: all selected tests pass; legacy `changed_source_ids` assertions remain valid and new refs are additive.

- [ ] **Step 7: Lint and commit**

```powershell
uv run --locked ruff check backend/app/ingestion/service.py backend/app/ingestion/sync.py backend/app/ingestion/source_versions.py backend/app/api/v1/integrations.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_review_v2_waterline.py backend/tests/test_mail_document_agent_api.py backend/tests/test_integration_runtime_status.py
git diff --check
git add backend/app/ingestion/service.py backend/app/ingestion/sync.py backend/app/ingestion/source_versions.py backend/app/api/v1/integrations.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_review_v2_waterline.py backend/tests/test_mail_document_agent_api.py backend/tests/test_integration_runtime_status.py
git commit -m "feat: add explicit review sync waterline"
```

---

### Task 4: Centralize Every Review Transition and Promote Exactly Once

**Files:**
- Create: `backend/app/review/__init__.py`
- Create: `backend/app/review/transitions.py`
- Modify: `backend/app/knowledge/promotion.py`
- Modify: `backend/app/api/v1/review.py`
- Create: `backend/tests/test_review_transitions.py`
- Modify: `backend/tests/test_review.py`
- Modify: `backend/tests/test_review_knowledge_promotion.py`
- Modify: `backend/tests/test_review_rbac.py`

**Interfaces:**

```python
ReviewAction = Literal['approve', 'reject', 'needs_more_evidence']


@dataclass(frozen=True)
class PromotionResult:
    target_type: str | None
    created_record_ids: tuple[int, ...]
    created_timeline_event_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReviewTransitionResult:
    item_id: int
    status: str
    replayed: bool
    promotion: PromotionResult | None


@dataclass(frozen=True)
class ReviewBatchTransitionResult:
    results: tuple[ReviewTransitionResult, ...]
    failed_items: tuple[dict[str, object], ...]
    skipped_items: tuple[dict[str, object], ...]


class InvalidReviewTransition(ValueError):
    code = 'invalid_state_transition'


class ReviewTransitionService:
    def transition(
        self,
        *,
        db: Session,
        item_id: int,
        action: ReviewAction,
        actor: DemoUser,
        note: str | None = None,
    ) -> ReviewTransitionResult: ...

    def transition_many(
        self,
        *,
        db: Session,
        item_ids: Sequence[int],
        action: ReviewAction,
        actor: DemoUser,
        note: str | None = None,
    ) -> ReviewBatchTransitionResult: ...
```

Routes retain the existing demo/production visibility query before calling the service. The service independently enforces existing Review RBAC plus exact actor permission-level membership, then mutates and flushes but does not commit. Each route records bounded audit metadata and commits the review state plus all promotion rows together.

- [ ] **Step 1: Write the state-machine and provenance tests**

Cover:

```python
def test_pending_approve_promotes_with_source_review_item_provenance(db_session) -> None: ...
def test_approve_replay_returns_existing_canonical_effect_ids(db_session) -> None: ...
def test_terminal_review_transition_is_rejected(db_session) -> None: ...
def test_needs_more_evidence_is_terminal_and_never_promotes(db_session) -> None: ...
def test_reject_records_reviewer_and_reviewed_at(db_session) -> None: ...
def test_bulk_transition_processes_ids_in_ascending_order(db_session) -> None: ...
def test_bulk_and_single_approve_create_effects_exactly_once(db_session) -> None: ...
def test_companion_timeline_provenance_is_exactly_once(db_session, item_type) -> None: ...
def test_transition_rechecks_exact_actor_permission_levels(db_session) -> None: ...
def test_transition_preserves_employee_reviewer_admin_rbac(db_session) -> None: ...
def test_transition_rechecks_evidence_before_approval(db_session) -> None: ...
def test_project_assignment_approval_has_no_knowledge_effect(db_session) -> None: ...
def test_legacy_approved_item_without_provenance_replays_without_new_effect(db_session) -> None: ...
```

Parameterize companion Timeline assertions for decision, history, and todo. Add route regressions proving single, bulk, and approve-agent-candidates all share replay semantics, and that terminal workflow-bound ReviewItems cannot be modified through `PATCH /review/{id}`.

- [ ] **Step 2: Run RED transition tests**

```powershell
uv run --locked pytest backend/tests/test_review_transitions.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_rbac.py -q
```

Expected: FAIL because routes directly mutate rows, promotion omits `source_review_item_id`, and terminal states can currently be rewritten.

- [ ] **Step 3: Make promotion provenance-aware and replayable**

Refactor the current promotion function so every main target and companion Timeline constructor receives:

```python
source_review_item_id=item.id
```

For an already approved V2 item, query each target table by `source_review_item_id`, order ids ascending, and return the canonical result rather than inserting. A pre-V2 approved item with no provenance returns `replayed=True`, `promotion=None`; it must not heuristically match or recreate knowledge. Keep the existing mapping:

- decision → `DecisionRecord` plus companion `TimelineEvent`
- history → `HistoryEvent` plus companion `TimelineEvent`
- timeline → `TimelineEvent`
- todo → `Todo` plus companion `TimelineEvent`
- project assignment → approved with no knowledge-table effect

Use the existing partial unique indexes as the final race backstop; do not add foreign keys or a migration.

- [ ] **Step 4: Implement the locked state machine**

Load all requested ids in ascending order with `SELECT ... FOR UPDATE` on PostgreSQL. SQLite uses the surrounding write transaction. For every row:

1. Require the route's existing demo/production visibility lookup, call existing `ensure_can_review_permission`, and independently verify `item.permission_level in actor.permission_levels`.
2. Revalidate source links/snippets and required structured fields before approval.
3. Permit mutation only from `pending_review`.
4. For `approved + approve`, return existing canonical ids with `replayed=True` and make no mutation/audit rewrite.
5. Reject every other terminal action with `InvalidReviewTransition`.
6. Set reviewer and reviewed timestamp for all three terminal outcomes.
7. Promote only approval and keep state plus all effects in the route's one transaction.

If a unique conflict occurs after another transaction wins, use a savepoint, refresh the item, and read existing provenance. Do not roll back unrelated items in the bulk transaction.

- [ ] **Step 5: Route every entry point through the service**

Replace direct mutations in:

- single approve
- single reject
- needs-more-evidence
- bulk approve/reject
- approve-agent-candidates

Preserve existing response fields and add the frozen replay/promotion fields. V2-bound audit metadata may contain actor, thread, graph version, action, result code, and bounded counts, but never item/source ids, URLs, snippets, prompts, or model output. Legacy unbound PATCH behavior remains compatible; a terminal workflow-bound item returns 409.

Keep `backend/tests/test_agent_workflow_models.py` in the green gate to prove the four existing provenance partial unique indexes remain the only knowledge-idempotency schema. Candidate/effect keys are not a fallback for knowledge promotion.

- [ ] **Step 6: Run GREEN transition and route tests**

```powershell
uv run --locked pytest backend/tests/test_review_transitions.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_rbac.py backend/tests/test_agent_workflow_models.py -q
```

Expected: all selected tests pass, approval replay returns the same ids, and every companion Timeline has the originating ReviewItem id.

- [ ] **Step 7: Lint and commit**

```powershell
uv run --locked ruff check backend/app/review/__init__.py backend/app/review/transitions.py backend/app/knowledge/promotion.py backend/app/api/v1/review.py backend/tests/test_review_transitions.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_rbac.py
git diff --check
git add backend/app/review/__init__.py backend/app/review/transitions.py backend/app/knowledge/promotion.py backend/app/api/v1/review.py backend/tests/test_review_transitions.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_rbac.py
git commit -m "feat: make review promotion exactly once"
```

---

### Task 5: Build Registry-Backed LangChain Adapters and Idempotent Drafting

**Files:**
- Create: `backend/app/agent_runtime/model_router.py`
- Create: `backend/app/agent_runtime/review_v2_agents.py`
- Create: `backend/app/agent_runtime/review_v2_drafting.py`
- Modify: `backend/app/agent_runtime/state.py`
- Modify: `backend/app/agent_runtime/__init__.py`
- Create: `backend/tests/test_review_v2_agents.py`
- Create: `backend/tests/test_review_v2_drafting.py`
- Test: `backend/tests/test_langchain_langgraph_dependency_compat.py`

**Interfaces:**

```python
class ReviewAgentAdapter(Protocol):
    manifest: AgentManifest
    estimated_output_tokens: int

    def preflight(self, packet: EvidencePacket) -> AgentCostBudgetDecision: ...

    def run(self, packet: EvidencePacket) -> AgentRunResult: ...


class ReviewAgentCatalog:
    @property
    def registry(self) -> AgentRegistry: ...

    def get(self, name: str) -> ReviewAgentAdapter: ...


@dataclass(frozen=True)
class ReviewDraftResult:
    review_item_ids: tuple[int, ...]
    review_status_counts: dict[ReviewItemResolutionStatus, int]


class ReviewDraftService:
    def preview_prepared(
        self,
        *,
        prepared: PreparedReviewRequest,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewWorkflowDryRunResponse: ...

    def draft(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewDraftResult: ...
```

The catalog registers exactly the five approved manifests. It pairs each manifest with one adapter; public lookup always goes through the Registry-backed catalog, not route-level imports.

- [ ] **Step 1: Write adapter, no-network, cost, lease, and replay tests**

Cover:

```python
def test_catalog_registers_only_exact_public_manifest_names() -> None: ...
def test_configured_mail_adapter_uses_existing_langchain_builder(monkeypatch) -> None: ...
def test_configured_memory_adapters_use_langchain_structured_output(monkeypatch) -> None: ...
def test_production_provider_absence_fails_closed_instead_of_using_rules() -> None: ...
def test_demo_catalog_uses_deterministic_models_without_network() -> None: ...
def test_preview_never_invokes_provider_or_writes_rows(db_session) -> None: ...
def test_over_budget_preflight_never_invokes_provider_or_writes_rows(db_session) -> None: ...
def test_draft_runs_provider_outside_transaction_and_lock(db_session) -> None: ...
def test_effect_replay_skips_model_and_reuses_agent_run_and_candidates(db_session) -> None: ...
def test_prompt_or_evidence_change_invalidates_effect_key(db_session) -> None: ...
def test_lease_expiry_discards_model_result_before_candidate_write(db_session) -> None: ...
def test_permission_or_signature_change_discards_model_result(db_session) -> None: ...
def test_candidate_without_evidence_is_rejected(db_session) -> None: ...
def test_draft_persists_token_cost_and_cache_metadata(db_session) -> None: ...
def test_new_exact_source_set_links_latest_needs_more_predecessor(db_session) -> None: ...
def test_predecessor_link_never_uses_fuzzy_or_partial_source_match(db_session) -> None: ...
```

Inject fake chat models and adapters. Assert call counts; no test may depend on an API key or network.

- [ ] **Step 2: Run RED agent/drafting tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_agents.py backend/tests/test_review_v2_drafting.py -q
```

Expected: FAIL because there is no central V2 catalog/model router/draft service.

- [ ] **Step 3: Build the replaceable model router using existing LangChain integrations**

For configured non-demo execution:

- Mail/document adapter constructs `MailDocumentAgent` with `build_langchain_mail_document_agent_model(...)`.
- Timeline/history/decision/todo adapters construct their existing agent classes with `LangChainMemoryExtractionModel(chat_model=..., expected_item_type=..., task_name=..., model_name=...)`, whose extraction uses LangChain `with_structured_output(...)`.
- OpenAI/Gemini chat construction remains behind settings and the existing LangChain provider packages.
- Missing production credentials or disabled model execution returns bounded `model_unavailable`; it must not silently choose deterministic rules.
- Local/demo construction may explicitly choose deterministic existing model classes.

Do not create direct provider HTTP calls, parse ad-hoc JSON model strings, or duplicate prompt rendering in routes/graph nodes.

- [ ] **Step 4: Build one exact permission-filtered EvidencePacket**

Load the workflow's immutable evidence refs, resolve current Source/DocumentVersion/parser data, check current signature and exact allowed permission levels, rank/dedupe/window it according to `COMPANY_MEMORY_SELECTION_POLICY_VERSION`, and construct `EvidencePacket`/`EvidenceMessage` with the source evidence required by existing agents.

The packet exists only in process memory during preview/draft. Store only its HMACs and bounded counts in workflow/checkpoint/audit records.

- [ ] **Step 5: Implement aggregate dry-run without provider invocation**

Each adapter's `preflight()` uses deterministic token estimation, existing `evaluate_agent_run_budget`, effect-cache lookup, and its declared output cap. Aggregate with the frozen budget rules. `preview_prepared()` accepts Task 2's in-memory `PreparedReviewRequest` and therefore writes no thread/evidence row. It may use `prepared.input_hash` plus full stored-request/evidence comparison to find an existing matching thread read-only and check its thread-bound effects; a new batch has no thread effect cache. The public dry-run path never creates or requires a persisted id. Actual start repeats this preflight; `over_budget` raises 409 `budget_exceeded` before thread/effect/checkpoint writes even if a client bypasses the disabled UI action.

- [ ] **Step 6: Implement the short lease and insert-or-return effects**

Use these phases:

1. Lock/CAS thread, verify status, current actor access, evidence version, and cancellation. For each selected adapter, derive `effect_key` from thread id, agent name, prompt version, model-route version, evidence HMAC, exact permission fingerprint, and selection policy.
2. Query existing AgentRuns by those keys before any provider call. Mark their agents cached; if every effect exists, return their existing candidate ids without claiming a lease or invoking a model.
3. For uncached agents only, assign opaque `lease_token`, expiration, and `drafting`; commit.
4. Build the exact packet and call only uncached adapters outside every application transaction.
5. Reopen a transaction, lock the thread, compare lease token/expiry/state version, and revalidate source signature, permission, cancellation, plus the still-missing effect keys.
6. Derive each `candidate_key` from effect key plus normalized structured candidate output.
7. Insert-or-return the thread-bound `AgentRun` and ReviewItems in the same transaction, preserving evidence metadata, strictest permission, and estimated/actual token/cost/cache fields. A concurrent winner is read back rather than duplicated.
8. For a new candidate, set `predecessor_review_item_id` only to the newest terminal `needs_more_evidence` item in the same security scope with the exact same agent name, item type, and normalized complete prefixed source-id set. Partial overlap, semantic similarity, or fuzzy matching never creates a predecessor link.
9. Set `checkpoint_pending`, clear the lease, increment state version, and commit.

Do not call the existing bridge helpers that commit internally. Reuse agent `run(packet)` results through the new graph-owned persistence boundary.

- [ ] **Step 7: Run GREEN adapter, drafting, and library compatibility tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_agents.py backend/tests/test_review_v2_drafting.py backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

Expected: all selected tests pass; fake provider call counts prove cache/lease behavior and current LangChain constructors remain compatible.

- [ ] **Step 8: Lint and commit**

```powershell
uv run --locked ruff check backend/app/agent_runtime/model_router.py backend/app/agent_runtime/review_v2_agents.py backend/app/agent_runtime/review_v2_drafting.py backend/app/agent_runtime/state.py backend/app/agent_runtime/__init__.py backend/tests/test_review_v2_agents.py backend/tests/test_review_v2_drafting.py backend/tests/test_langchain_langgraph_dependency_compat.py
git diff --check
git add backend/app/agent_runtime/model_router.py backend/app/agent_runtime/review_v2_agents.py backend/app/agent_runtime/review_v2_drafting.py backend/app/agent_runtime/state.py backend/app/agent_runtime/__init__.py backend/tests/test_review_v2_agents.py backend/tests/test_review_v2_drafting.py backend/tests/test_langchain_langgraph_dependency_compat.py
git commit -m "feat: add registry backed review drafting"
```

---

### Task 6: Compile the Real Review-Only LangGraph and Prove Its Interrupt Boundary

**Files:**
- Create: `backend/app/agent_runtime/review_v2_graph.py`
- Modify: `backend/app/agent_runtime/state.py`
- Modify: `backend/app/agent_runtime/graph_versions.py`
- Create: `backend/tests/test_review_v2_graph.py`
- Modify: `backend/tests/test_agent_runtime_graph_versions.py`
- Modify: `backend/tests/test_agent_runtime_checkpoint_execution.py`

**Interfaces:**

```python
def build_company_memory_review_v2_graph(
    saver: BaseCheckpointSaver,
) -> object: ...


def register_company_memory_review_v2(
    registry: GraphVersionRegistry,
) -> None: ...
```

Compile with the existing typed contracts:

```python
builder = StateGraph(
    ReviewGraphState,
    input_schema=ReviewGraphInput,
    output_schema=ReviewGraphOutput,
    context_schema=ReviewRuntimeContext,
)
```

The graph version registry key is exactly `('company-memory-review', 'company-memory-review-v2.0')`.

- [ ] **Step 1: Write actual graph topology and interrupt/resume tests**

Cover:

```python
def test_review_graph_registers_immutable_workflow_version() -> None: ...
def test_no_candidate_run_finishes_without_interrupt(in_memory_saver) -> None: ...
def test_candidate_run_interrupts_with_exact_safe_payload(in_memory_saver) -> None: ...
def test_resume_uses_same_thread_and_acknowledgement_only(in_memory_saver) -> None: ...
def test_pending_database_items_reinterrupt_defensively(in_memory_saver) -> None: ...
def test_needs_more_evidence_finishes_in_terminal_branch(in_memory_saver) -> None: ...
def test_approved_and_rejected_items_finish_completed(in_memory_saver) -> None: ...
def test_database_resolution_ignores_resume_payload_claims(in_memory_saver) -> None: ...
def test_checkpoint_contains_no_evidence_prompt_or_model_output(in_memory_saver) -> None: ...
def test_graph_has_no_rag_slack_neo4j_or_trusted_knowledge_node() -> None: ...
def test_candidate_set_is_immutable_after_confirmed_pause(in_memory_saver) -> None: ...
```

Use a fake draft service and an application DB fixture. Resume with the exact real LangGraph API, not a direct node call.

- [ ] **Step 2: Run the RED graph tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_graph.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py -q
```

Expected: FAIL because the V2 graph builder and registration do not exist.

- [ ] **Step 3: Implement the exact graph topology**

Register these small nodes and conditional edges:

```text
START
  -> validate_input
  -> collect_evidence_refs
  -> plan_agent_runs
  -> draft_review_candidates_transaction
  -> route_review_boundary
       -> no_candidates -> finalize_no_candidates -> END
       -> candidates_created -> await_human_review [interrupt]
  -> verify_review_resolution_from_postgres
       -> unresolved -> await_human_review
       -> needs_more_evidence -> finalize_needs_more_evidence -> END
       -> approved_or_rejected -> finalize_review_trace -> END
```

Each node returns only changed state fields. Session, registry, services, and actor data come from `ReviewRuntimeContext`; never serialize them into state. Once the first interrupt is confirmed, no normal resume path calls draft or appends a ReviewItem; only a new canonical source-version/policy/graph batch may create another candidate set.

- [ ] **Step 4: Use exact interrupt and resume acknowledgement payloads**

`await_human_review` is side-effect free and reads the current application thread `state_version` for:

```python
interrupt({
    'event': 'review_resolution_required',
    'state_version': current_state_version,
})
```

Tests resume the exact saver thread/root namespace with:

```python
Command(resume={
    'event': 'review_resolution_checked',
    'state_version': current_state_version,
})
```

The verification node ignores any approval/status claims in the acknowledgement and reloads bound ReviewItems plus current actor permission from PostgreSQL. Pending defensively routes back to the interrupt; needs-more terminates accordingly; all-approved/rejected completes.

- [ ] **Step 5: Run GREEN and verify the saved and returned interrupt exactly**

Use `invoke_and_confirm_checkpoint(..., durability='sync')` and `require_resumable_checkpoint(...)` from Deliverable B. Keep `checkpoint_ns=''`. A candidate run is not considered paused until the returned interrupt and saver tuple match exactly.

```powershell
uv run --locked pytest backend/tests/test_review_v2_graph.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py -q
```

Expected: all selected tests pass, including recursive sensitive-value inspection of the saved tuple.

- [ ] **Step 6: Lint and commit**

```powershell
uv run --locked ruff check backend/app/agent_runtime/review_v2_graph.py backend/app/agent_runtime/state.py backend/app/agent_runtime/graph_versions.py backend/tests/test_review_v2_graph.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py
git diff --check
git add backend/app/agent_runtime/review_v2_graph.py backend/app/agent_runtime/state.py backend/app/agent_runtime/graph_versions.py backend/tests/test_review_v2_graph.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py
git commit -m "feat: add interruptible review workflow graph"
```

---

### Task 7: Add Workflow Lifecycle Service, V2 Routes, and Review Filtering

**Files:**
- Create: `backend/app/agent_runtime/review_v2_service.py`
- Create: `backend/app/api/v1/orchestration_v2.py`
- Modify: `backend/app/api/v1/router.py`
- Modify: `backend/app/api/v1/review.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/agent_runtime/checkpointing.py`
- Create: `backend/tests/test_review_v2_service.py`
- Create: `backend/tests/test_review_v2_api.py`
- Modify: `backend/tests/test_review.py`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ReviewWorkflowStatus:
    thread_id: str
    status: str
    review_item_count: int
    review_status_counts: dict[ReviewItemResolutionStatus, int]
    durable: bool
    graph_version: str
    review_resolution_ready: bool
    checkpoint_resumable: bool
    resume_allowed: bool
    retry_allowed: bool
    created_at: datetime
    updated_at: datetime
    error_code: ReviewWorkflowErrorCode | None
    resume_error_code: ReviewWorkflowErrorCode | None


class ReviewWorkflowService:
    def diagnostic(self) -> ReviewWorkflowDiagnosticResponse: ...
    def dry_run(self, *, actor: DemoUser, request: ReviewWorkflowRunRequest) -> ReviewWorkflowDryRunResponse: ...
    def start(self, *, actor: DemoUser, request: ReviewWorkflowRunRequest) -> ReviewWorkflowStatus: ...
    def status(self, *, actor: DemoUser, thread_id: str) -> ReviewWorkflowStatus: ...
    def resume(self, *, actor: DemoUser, thread_id: str) -> ReviewWorkflowStatus: ...
    def cancel(self, *, actor: DemoUser, thread_id: str) -> ReviewWorkflowStatus: ...
```

The service owns all graph invocation, checkpoint readiness, lifecycle CAS, live Review projection, authorization, and bounded error mapping. API routes only validate/authenticate, call the service, and select HTTP status.

Before every tuple probe/invoke, the service calls an exact-mode guard:

```python
def require_thread_checkpoint_runtime(
    *,
    thread: AgentWorkflowThread,
    runtime: CheckpointRuntime,
) -> BaseCheckpointSaver:
    if thread.checkpoint_store != runtime.readiness.checkpoint_store:
        raise CheckpointUnavailableError('checkpoint runtime mode mismatch')
    if not runtime.readiness.ready or runtime.saver is None:
        raise CheckpointUnavailableError('checkpoint runtime unavailable')
    return runtime.saver
```

The route maps both internal messages to the one public `checkpoint_unavailable` code. A `memory` thread is never probed through Postgres and a Postgres thread is never probed through memory.

- [ ] **Step 1: Write service lifecycle, authorization, repair, and mode-matrix tests**

Cover:

```python
def test_initial_candidate_run_returns_only_after_confirmed_interrupt(db_session) -> None: ...
def test_initial_no_candidate_run_is_terminal(db_session) -> None: ...
def test_actual_start_rejects_over_budget_before_thread_write(db_session) -> None: ...
def test_status_projects_current_review_rows_not_checkpoint_counts(db_session) -> None: ...
def test_pending_resume_returns_review_unresolved_before_command_or_saver_write(db_session) -> None: ...
def test_same_thread_resume_completes_after_all_items_resolve(db_session) -> None: ...
def test_checkpoint_failed_repairs_same_application_thread_and_reuses_effects(db_session) -> None: ...
def test_valid_saved_tuple_reconciles_status_without_rewriting_checkpoint(db_session) -> None: ...
def test_missing_tuple_rotates_only_checkpoint_attempt_id_and_reuses_business_rows(db_session) -> None: ...
def test_failed_and_cancelled_threads_are_terminal_without_retry(db_session) -> None: ...
def test_draft_retry_budget_exhaustion_marks_failed_without_replacement(db_session) -> None: ...
def test_concurrent_resume_and_cancel_allow_one_cas_winner(db_session) -> None: ...
def test_owner_reviewer_admin_and_cross_owner_authorization_matrix(db_session) -> None: ...
def test_permission_revocation_fails_closed_without_deleting_knowledge(db_session) -> None: ...
def test_unsupported_graph_version_is_non_mutating(db_session) -> None: ...
def test_checkpoint_store_mismatch_fails_closed_without_tuple_probe(db_session) -> None: ...
def test_visible_workflow_filter_returns_only_bound_items(client, db_session) -> None: ...
def test_hidden_missing_and_zero_visible_workflow_filters_are_identical_empty_results(client, db_session) -> None: ...
def test_workflow_filter_runs_before_pagination_grouping_and_respects_status(client, db_session) -> None: ...
```

Add API tests for every diagnostic/new-run/existing-thread row in the approved feature-flag/readiness matrix, HTTP 200/202/400/404/409/503 mapping, safe error bodies, and absence of sensitive/internal fields.

- [ ] **Step 2: Run RED service/API tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_agent_runtime_lifespan.py -q
```

Expected: FAIL because the lifecycle service/routes/app registration do not exist.

- [ ] **Step 3: Build lifespan-scoped registries and immutable graph registration**

At application construction/startup:

- Create one `GraphVersionRegistry`, register `company-memory-review-v2.0` unconditionally, and store it on `app.state` so existing threads never fall back to V1.
- Create the settings-selected `ReviewAgentCatalog` and service dependencies; tests inject fakes.
- Keep the checkpoint runtime lifespan-scoped and close it once.
- Add `CheckpointRuntime.start(preserve_existing_review_threads: bool = False)`. When the new-run flag is false, application startup queries only non-terminal `company-memory-review` V2 rows; if at least one exists, it starts the database-appropriate runtime solely for those existing threads. Diagnostic still reports new-run `enabled=false`, `available=false`, `checkpoint_mode='disabled'`. If none exists, the runtime remains disabled as today.
- Separate "accept new V2 runs" from "service an existing immutable V2 thread" inside the lifecycle service. Existing status/cancel remains DB-backed; resume uses the exact-mode guard above. A missing/unready/mismatched saver returns `checkpoint_unavailable`, never V1 or memory fallback.
- In memory mode, restart loss yields `checkpoint_resumable=false` and resume 503.

Do not call `PostgresSaver.setup()` at startup.

- [ ] **Step 4: Implement live status projection and authorization**

For every status request:

1. Load the thread and all bound evidence refs.
2. Hide foreign scope or any invisible evidence as 404.
3. Load bound ReviewItems and count current statuses.
4. Set `review_resolution_ready` only when at least one item exists, no pending item remains, and the actor can see every item.
5. Probe the exact saver tuple for `checkpoint_resumable` without mutating it.
6. Set `resume_allowed` only for `awaiting_human_review` when both readiness flags are true.
7. Set `retry_allowed` only for authorized `checkpoint_failed` with a currently ready checkpoint runtime.

Owner may status/resume/cancel while retaining all evidence access. Same-scope authorized users may status/view. Non-owner ordinary users cannot resume/cancel; reviewer/admin may resume with all bound-item permissions; admin may cancel. Review approval RBAC remains separate.

- [ ] **Step 5: Implement initial start and confirmed pause**

Call preflight, then draft/graph through the immutable builder. If the exact thread already exists, return its current status without creating effects. For a new thread:

- allow at most two bounded internal draft attempts within the original start operation; provider/model exhaustion marks the same thread `failed`, returns HTTP 503 `model_unavailable`, and exposes no user retry/replacement;
- candidate count zero → commit terminal `completed`, HTTP 200;
- candidates present → move through `checkpoint_pending`, synchronously confirm interrupt, then separately CAS to `awaiting_human_review`, HTTP 202;
- saver unavailable before invocation → preserve business rows, set/retain the appropriate pre-pause state, and return HTTP 503 `checkpoint_unavailable`;
- saver write/interrupt confirmation failure → preserve candidates, set `checkpoint_failed`, and return HTTP 503 `checkpoint_failed`;
- checkpoint confirmed but application status CAS fails → return bounded 503, then let the next status/resume reconcile the valid tuple without another candidate effect.

- [ ] **Step 6: Implement explicit resume, repair, reconciliation, and cancel**

Before creating `Command` or invoking the graph, query live counts. Any pending item returns 409 `review_unresolved` with zero saver writes. Valid `awaiting_human_review` resumes the same `checkpoint_thread_id` and root namespace with acknowledgement only.

For `checkpoint_failed`:

- valid expected tuple → reconcile application status;
- missing/corrupt tuple → assign a new opaque `checkpoint_thread_id` attempt on the same application thread, reuse existing AgentRun/ReviewItem effects, and recreate/confirm the interrupt;
- unsupported graph version → return `runtime_version_unavailable` without mutation.

Cancel only allowed non-terminal statuses through state-version CAS. It sets cancellation metadata but deletes no ReviewItem, AgentRun, checkpoint, or knowledge row. Failed/cancelled/completed/needs-more are terminal and do not retry.

- [ ] **Step 7: Add the six V2 endpoints**

Create and include a router with exactly:

```text
GET  /api/v1/orchestration/v2/company-memory
POST /api/v1/orchestration/v2/company-memory/dry-run
POST /api/v1/orchestration/v2/company-memory/runs
GET  /api/v1/orchestration/v2/company-memory/runs/{thread_id}
POST /api/v1/orchestration/v2/company-memory/runs/{thread_id}/resume
POST /api/v1/orchestration/v2/company-memory/runs/{thread_id}/cancel
```

Do not import or call LangGraph in `orchestration_v2.py`.

- [ ] **Step 8: Add permission-first Review workflow filtering**

Add optional `workflow_thread_id` to `GET /api/v1/review`. Apply it only after existing visibility filters and before status selection, pagination, and grouping. A visible workflow returns only its bound items for the requested status. Foreign, invisible, missing, or zero-visible-item filters return the identical `items=[]`, `groups=[]`, `total_count=0`; the status endpoint separately returns 404.

- [ ] **Step 9: Run GREEN service/API/filter tests**

```powershell
uv run --locked pytest backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review.py backend/tests/test_agent_runtime_lifespan.py -q
```

Expected: all selected tests pass, pending resume never reaches the saver, and response scans find no internal/sensitive fields.

- [ ] **Step 10: Lint and commit**

```powershell
uv run --locked ruff check backend/app/agent_runtime/review_v2_service.py backend/app/api/v1/orchestration_v2.py backend/app/api/v1/router.py backend/app/api/v1/review.py backend/app/main.py backend/app/agent_runtime/checkpointing.py backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review.py backend/tests/test_agent_runtime_lifespan.py
git diff --check
git add backend/app/agent_runtime/review_v2_service.py backend/app/api/v1/orchestration_v2.py backend/app/api/v1/router.py backend/app/api/v1/review.py backend/app/main.py backend/app/agent_runtime/checkpointing.py backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review.py backend/tests/test_agent_runtime_lifespan.py
git commit -m "feat: expose review workflow v2 lifecycle"
```

---

### Task 8: Add the Typed Frontend V2 Contract and Route Wrappers

**Files:**
- Modify: `frontend/src/lib/api/types.ts`
- Create: `frontend/src/lib/api/reviewWorkflow.ts`
- Create: `frontend/e2e/review-hitl-v2-api.spec.ts`

**Interfaces:**

```typescript
export type ReviewWorkflowSourceRef = {
  source_type: "gmail" | "gmail_attachment" | "drive" | "calendar";
  source_id: string;
  version_or_signature: string;
};

export type ReviewWorkflowRunRequest = {
  source_refs: ReviewWorkflowSourceRef[];
  agent_names: string[];
  client_request_id?: string;
};

export type ReviewWorkflowErrorCode =
  | "invalid_input"
  | "not_found"
  | "idempotency_key_reused"
  | "evidence_changed"
  | "permission_denied"
  | "checkpoint_unavailable"
  | "checkpoint_failed"
  | "review_unresolved"
  | "runtime_version_unavailable"
  | "model_unavailable"
  | "budget_exceeded"
  | "concurrent_resume"
  | "invalid_state_transition";

export function getReviewWorkflowDiagnostic(): Promise<ReviewWorkflowDiagnostic>;
export function dryRunReviewWorkflow(request: ReviewWorkflowRunRequest): Promise<ReviewWorkflowDryRun>;
export function launchReviewWorkflow(request: ReviewWorkflowRunRequest): Promise<ReviewWorkflowStatus>;
export function getReviewWorkflowStatus(threadId: string): Promise<ReviewWorkflowStatus>;
export function resumeReviewWorkflow(threadId: string): Promise<ReviewWorkflowStatus>;
export function cancelReviewWorkflow(threadId: string): Promise<ReviewWorkflowStatus>;
```

- [ ] **Step 1: Write wrapper path/method/payload tests**

In the Playwright Node test, replace `globalThis.fetch` with a recording fake inside `try/finally` and restore it in `finally`. Because `apiUrl()` produces an absolute backend URL in a Node worker, compare `new URL(String(input)).pathname`, then assert:

- diagnostic and status use GET;
- dry-run, launch, resume, and cancel use POST;
- every path exactly matches the approved six endpoints;
- launch serializes only `source_refs`, `agent_names`, and optional `client_request_id`;
- safe bounded backend error JSON becomes a typed UI-consumable code and no raw server object is rendered by the wrapper.

- [ ] **Step 2: Run the RED frontend contract test**

```powershell
Set-Location frontend
npm.cmd run test:visual -- review-hitl-v2-api.spec.ts --project=chromium-desktop
```

Expected: FAIL because `reviewWorkflow.ts` and the V2 TypeScript types do not exist.

- [ ] **Step 3: Add exact additive TypeScript types**

Mirror the human-gated backend request, diagnostic, dry-run, lifecycle status, status-count, and exact `ReviewWorkflowErrorCode` union. Add:

```typescript
changed_source_refs?: ReviewWorkflowSourceRef[];
```

to `IntegrationSyncResponse` while retaining `changed_source_ids`. Preserve `ReviewApprovalResponse.promotion_result` and add `replayed` plus the new stable nullable `promotion` field. Preserve all existing bulk fields and add `replayed_items`.

- [ ] **Step 4: Implement one route wrapper module**

Use existing `apiGet`/`apiPost`; do not rewrite auth, CSRF, fetch transport, or API base handling. URL-encode `threadId`. Export one narrowly scoped error reader that accepts only the known `{detail: {code}}`/serialized-code shape produced by these wrappers. Unknown, malformed, or arbitrary `Error.message` content maps to one generic Korean failure; pages never render the raw message.

- [ ] **Step 5: Run GREEN contract and compile gates**

```powershell
npm.cmd run test:visual -- review-hitl-v2-api.spec.ts --project=chromium-desktop
npm.cmd run lint
npx.cmd tsc --noEmit
```

Expected: wrapper test, ESLint, and TypeScript checks pass.

- [ ] **Step 6: Commit**

```powershell
Set-Location ..
git diff --check
git add frontend/src/lib/api/types.ts frontend/src/lib/api/reviewWorkflow.ts frontend/e2e/review-hitl-v2-api.spec.ts
git commit -m "feat: add typed review workflow client"
```

---

### Task 9: Add the One-Click Cost Preview and Launch to Integrations

**Files:**
- Create: `frontend/src/app/integrations/ReviewCandidateLaunchPanel.tsx`
- Modify: `frontend/src/app/integrations/page.tsx`
- Create: `frontend/e2e/review-hitl-v2-integrations.spec.ts`
- Test: `frontend/e2e/integration-sync-modal.spec.ts`

**UX contract:**

- Stay in the existing sync completion modal; add no wizard, route, sidebar item, or agent-selection screen.
- Fetch diagnostic on page load.
- When the latest non-Slack completed sync has canonical changed refs and V2 is available, automatically run the no-cost dry-run and display source count, estimated input/output tokens, estimated cost, budget status, and cache/no-cost explanation.
- Provide one primary action: `검토 후보 만들기`.
- Build stable `client_request_id` as `review:{job_id}:{selection_policy_version}`; the backend canonical batch remains authoritative.
- Navigate to `/review?workflow_thread_id=<thread_id>` only after a confirmed 202 pause. Since the shared client returns parsed JSON rather than HTTP status, classify no-candidate only as `status === 'completed' && review_item_count === 0`; then keep the user in Integrations with `새 검토 후보 없음`. Never apply that copy to failed/cancelled zero-effect replays.

- [ ] **Step 1: Write desktop behavior tests before the component**

Mock integration status/sync, diagnostic, dry-run, and launch. Cover:

```typescript
test("shows cost and launches the exact completed source batch", async ({ page }) => {});
test("keeps the user on integrations when no candidates are created", async ({ page }) => {});
test("hides launch when readiness is disabled or unavailable", async ({ page }) => {});
test("does not offer V2 launch for Slack sync", async ({ page }) => {});
test("reuses one stable client request id after response loss", async ({ page }) => {});
test("does not label failed or cancelled zero effect replay as no candidates", async ({ page }) => {});
```

Assert the POST body uses the exact dry-run refs and server-provided `default_agent_names`; never reconstruct a signature from `changed_source_ids`.

- [ ] **Step 2: Run the RED Integrations test**

```powershell
Set-Location frontend
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- review-hitl-v2-integrations.spec.ts --project=chromium-desktop
```

Before this page test, start the existing Next dev server in a second terminal with `npm.cmd run dev -- --hostname 127.0.0.1 --port 3000` and wait for `http://127.0.0.1:3000/integrations`; stop it before any production build. Expected: FAIL because the completion modal has no V2 cost/launch panel.

- [ ] **Step 3: Implement readiness and exact sync-batch state**

Extend existing `SyncProgressState` with the latest job id, canonical refs, diagnostic, dry-run result, launch state, and safe user-facing error. Ignore stale polling responses by comparing job ids. A later sync replaces the launchable batch.

When V2 is enabled and available, a non-Slack completed result with refs automatically calls dry-run. When disabled/unavailable, preserve existing legacy completion UX and do not call V2 run.

- [ ] **Step 4: Build the pure launch panel in the existing modal**

Render directly below current sync result metrics. It must show:

- changed source count;
- token and USD estimates;
- budget state (`예산 초과` disables launch);
- `캐시 재사용 · 추가 모델 호출 없음` for cached and `추가 비용 없음` for no-input;
- memory-mode notice when diagnostic is non-durable;
- loading/disabled state preventing double submit;
- one launch button and safe retry guidance.

Do not show raw backend error codes or add a cancel control.

- [ ] **Step 5: Implement confirmed navigation/no-candidate handling**

For `awaiting_human_review`, navigate using the opaque thread id only after the launch wrapper resolves that confirmed status. For `completed` with zero items, keep the modal/page and show the no-candidate message. For a replayed waiting thread, use its current status and navigate only when it is actually awaiting review; failed/cancelled replays show safe terminal guidance.

- [ ] **Step 6: Run GREEN and legacy modal regressions**

```powershell
npm.cmd run lint
npm.cmd run build
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- review-hitl-v2-integrations.spec.ts integration-sync-modal.spec.ts --project=chromium-desktop
```

Run lint/build with the dev server stopped, restart the server with the command from Step 2, run Playwright, then stop it. Expected: new flow and existing sync modal pass; production build succeeds.

- [ ] **Step 7: Commit**

```powershell
Set-Location ..
git diff --check
git add frontend/src/app/integrations/ReviewCandidateLaunchPanel.tsx frontend/src/app/integrations/page.tsx frontend/e2e/review-hitl-v2-integrations.spec.ts
git commit -m "feat: launch review workflow from integrations"
```

---

### Task 10: Add Workflow Context and Explicit Completion to Review

**Files:**
- Create: `frontend/src/app/review/ReviewWorkflowContextPanel.tsx`
- Modify: `frontend/src/app/review/page.tsx`
- Create: `frontend/e2e/review-hitl-v2-review.spec.ts`
- Test: `frontend/e2e/review-bulk-actions.spec.ts`
- Test: `frontend/e2e/review-agent-metadata.spec.ts`

**UX contract:**

- Parse `workflow_thread_id` alongside the existing `itemId` deep link.
- Resolve the URL query before the first Review list request; never fetch or flash the unfiltered global queue first.
- Append the workflow filter to every Review list page request.
- Load workflow status in parallel; server-side current counts, not the pending-only visible list length, drive progress.
- Never call resume from approve/reject/needs-more actions.
- Show exactly one lifecycle action: `검토 완료` when waiting and ready, or `다시 시도` only for `checkpoint_failed` with `retry_allowed=true`.
- Keep existing bulk/group/evidence drawer behavior and no new navigation depth.

- [ ] **Step 1: Write desktop and mobile workflow-context tests**

Cover:

```typescript
test("filters review items and enables explicit completion only when ready", async ({ page }) => {});
test("review item actions never auto resume the workflow", async ({ page }) => {});
test("needs more evidence ends with guidance and no retry", async ({ page }) => {});
test("checkpoint failed is the only retryable state", async ({ page }) => {});
test("failed and cancelled states show safe terminal guidance", async ({ page }) => {});
test("memory mode explains process local durability", async ({ page }) => {});
test("workflow panel stacks above queue actions on mobile", async ({ page }) => {});
test("the first review request already contains the workflow filter", async ({ page }) => {});
test("status 404 hides the thread id and raw detail behind generic copy", async ({ page }) => {});
test("an empty hidden workflow keeps group and bulk controls safe", async ({ page }) => {});
```

For the no-auto-resume test, fulfill the resume route with a counter and prove it remains zero after approve, reject, and needs-more actions.

- [ ] **Step 2: Run the RED Review test**

```powershell
Set-Location frontend
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- review-hitl-v2-review.spec.ts --project=chromium-desktop
```

Before this page test, start the existing Next dev server in a second terminal with `npm.cmd run dev -- --hostname 127.0.0.1 --port 3000` and wait for the Review route; stop it before any production build. Expected: FAIL because Review ignores `workflow_thread_id` and has no lifecycle panel/action.

- [ ] **Step 3: Add permission-safe filtering and live status refresh**

Initialize/query-gate `workflow_thread_id` before `loadItems` can issue its first request, then add it to every paginated request while preserving `itemId` scroll behavior. Fetch status in parallel, refresh it after each Review mutation, and keep pagination/grouping sourced from the filtered Review endpoint. A 404 shows a generic unavailable message that includes neither the opaque thread id nor raw backend detail; an empty hidden-filter response leaves grouping/bulk controls disabled and stable.

- [ ] **Step 4: Build the pure context panel**

Place it directly below the Review heading. Render Korean labels, `완료 수 / 전체 수`, durability notice, and status-specific guidance:

- `awaiting_human_review`: `검토 완료`, disabled until `review_resolution_ready && resume_allowed`;
- `checkpoint_failed`: `다시 시도`, shown only when `retry_allowed`;
- `completed`: completion confirmation;
- `needs_more_evidence`: new evidence/sync guidance, no resume/retry;
- `failed`/`cancelled`: terminal guidance, no resume/retry.

The action calls only the explicit service wrapper. It disables during the request and refreshes status after success/failure.

- [ ] **Step 5: Preserve item actions without auto-resume**

Keep `runStatusAction` responsible only for Review transitions and local/list refresh. Do not add an effect that watches pending count or statuses and resumes automatically. Bulk partial failures remain visible through live server counts.

- [ ] **Step 6: Run GREEN desktop/mobile and existing Review regressions**

```powershell
npm.cmd run lint
npm.cmd run build
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- review-hitl-v2-review.spec.ts review-bulk-actions.spec.ts review-agent-metadata.spec.ts --project=chromium-desktop
npm.cmd run test:visual -- review-hitl-v2-review.spec.ts --project=chromium-mobile
```

Run lint/build with the dev server stopped, restart it with the command from Step 2, run both Playwright projects, then stop it. Expected: all selected desktop/mobile tests and production build pass; the primary action is reachable without a new screen.

- [ ] **Step 7: Commit**

```powershell
Set-Location ..
git diff --check
git add frontend/src/app/review/ReviewWorkflowContextPanel.tsx frontend/src/app/review/page.tsx frontend/e2e/review-hitl-v2-review.spec.ts
git commit -m "feat: complete review workflows explicitly"
```

---

### Task 11: Prove PostgreSQL Recovery, Concurrency, Regressions, and Rollout Evidence

**Files:**
- Create: `backend/tests/test_review_v2_postgres.py`
- Create: `backend/tests/test_review_transition_postgres.py`
- Modify: `docs/portfolio-log.md`
- Modify: `docs/superpowers/runbooks/session-handoff.md`
- Modify: `plan.md`

**Release gate:** This task does not add feature behavior. It independently proves the approved behavior against a disposable PostgreSQL database, runs all non-Slack and frontend gates, records exact evidence, and leaves V2 disabled by default.

- [ ] **Step 1: Write the PostgreSQL restart/recovery and race tests**

Cover with fake/deterministic adapters only:

```python
def test_postgres_interrupt_survives_pool_and_app_restart_then_resumes_same_thread() -> None: ...
def test_postgres_saved_tuple_contains_no_source_or_model_content() -> None: ...
def test_checkpoint_commit_then_status_failure_reconciles_without_duplicate_effects() -> None: ...
def test_draft_commit_then_checkpoint_failure_repairs_without_duplicate_effects() -> None: ...
def test_concurrent_exact_batch_launches_create_one_thread() -> None: ...
def test_concurrent_single_and_bulk_approval_create_one_target_and_companion() -> None: ...
def test_concurrent_resume_has_one_state_version_winner() -> None: ...
def test_terminal_review_race_returns_bounded_conflict_without_effect() -> None: ...
```

The fixture must validate the database identity before mutation, generate unique test thread/source ids, close saver/pool/app A completely, create independent B resources, resume at root namespace, and clean only its generated rows in `finally`. It must not skip when `PARAWORKS_TEST_POSTGRES_URL` is configured. The approval race asserts one `replayed=false` winner, one `replayed=true` observer with identical canonical ids, no leaked `IntegrityError`, and exactly-once companion provenance for decision/history/todo. Reject/needs-more terminal races must return bounded 409 and insert no effect.

- [ ] **Step 2: Run the PostgreSQL gate and preserve the output**

```powershell
uv run --locked pytest backend/tests/test_review_v2_postgres.py backend/tests/test_review_transition_postgres.py -q
```

Expected: all selected tests pass against the disposable PostgreSQL target with no skip. A missing `PARAWORKS_TEST_POSTGRES_URL` is an explicit release blocker, not evidence of success.

- [ ] **Step 3: Run the complete Deliverable C focused backend suite**

```powershell
uv run --locked pytest backend/tests/test_review_v2_schemas.py backend/tests/test_legacy_company_memory_metadata.py backend/tests/test_review_v2_canonical_sources.py backend/tests/test_review_v2_preflight.py backend/tests/test_review_v2_waterline.py backend/tests/test_review_transitions.py backend/tests/test_review_v2_agents.py backend/tests/test_review_v2_drafting.py backend/tests/test_review_v2_graph.py backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review_v2_postgres.py backend/tests/test_review_transition_postgres.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_rbac.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_mail_document_agent_api.py backend/tests/test_integration_runtime_status.py backend/tests/test_orchestration_api.py -q
```

Expected: zero failures and zero live-provider calls.

- [ ] **Step 4: Re-run the complete Deliverable B runtime/checkpoint gate**

```powershell
uv run --locked pytest backend/tests/test_agent_preflight.py backend/tests/test_agent_runtime_state.py backend/tests/test_agent_runtime_fingerprints.py backend/tests/test_agent_workflow_models.py backend/tests/test_agent_runtime_migration.py backend/tests/test_agent_runtime_checkpointing.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_agent_runtime_bootstrap.py backend/tests/test_agent_runtime_retention.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_checkpoint_execution.py backend/tests/test_agent_runtime_postgres_checkpoint.py backend/tests/test_db_init.py backend/tests/test_db_schema_operations.py backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

Expected: all selected tests pass and no schema migration beyond `2f6a8b9c0d1e` appears.

- [ ] **Step 5: Run the non-Slack backend regression gate**

```powershell
uv run --locked pytest backend/tests -q --deselect=backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_runs_real_agent_services --deselect=backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_skips_agents_that_exceed_cost_budget --deselect=backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_uses_cache_when_evidence_is_unchanged --deselect=backend/tests/test_oauth_pkce.py::test_slack_oauth_pkce_generation --deselect=backend/tests/test_oauth_pkce.py::test_slack_callback_with_custom_redirect_uri_and_pkce --deselect=backend/tests/test_oauth_pkce.py::test_api_endpoints_support_redirect_uri --deselect=backend/tests/test_orchestration_api.py::test_company_memory_orchestration_api_runs_agent_services --deselect=backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_company_memory_emits_review_checkpoint_without_paid_calls --deselect=backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_cache_hit_does_not_duplicate_agent_runs_or_review_items --deselect=backend/tests/test_slack_oauth.py::test_slack_sync_endpoint_uses_installed_connection_token_without_exposing_it
```

Expected: every selected test passes; the existing optional pgvector environment test may remain skipped. Do not add deselections.

- [ ] **Step 6: Run the full backend suite and compare only the deferred Slack baseline**

```powershell
uv run --locked pytest backend/tests -q
```

Expected: exactly the same ten user-deferred Slack-related test ids fail and every new/non-Slack test passes. If any different failure appears, stop and fix it before documentation/commit. Do not hide, delete, weaken, or mark the Slack failures expected.

- [ ] **Step 7: Run backend lock/lint/diff gates**

```powershell
uv lock --check
uv run --locked ruff check backend/app backend/tests
git diff --check
```

Expected: lock is current, Ruff reports all checks passed, and no whitespace errors exist.

- [ ] **Step 8: Run all affected frontend gates on both viewports**

```powershell
Set-Location frontend
npm.cmd run lint
npm.cmd run build
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- review-hitl-v2-api.spec.ts review-hitl-v2-integrations.spec.ts review-hitl-v2-review.spec.ts --project=chromium-desktop
npm.cmd run test:visual -- review-hitl-v2-integrations.spec.ts review-hitl-v2-review.spec.ts --project=chromium-mobile
npm.cmd run test:visual -- integration-sync-modal.spec.ts review-bulk-actions.spec.ts review-agent-metadata.spec.ts --project=chromium-desktop
Set-Location ..
```

Run lint/build with no dev server active. Then start `npm.cmd run dev -- --hostname 127.0.0.1 --port 3000` in a second terminal, wait for the app, run all Playwright commands, and stop the server. Expected: lint/build and every selected desktop/mobile Playwright test pass.

- [ ] **Step 9: Run a deterministic two-screen smoke with V2 enabled**

Use SQLite/demo memory mode and fake/deterministic adapters. Verify:

1. Gmail/Drive/Calendar sync returns canonical refs without legacy candidates.
2. Integrations shows cost and launches once.
3. Candidate run reaches a confirmed interrupt and Review deep link.
4. Review filter shows only bound visible items.
5. Approval/rejection/needs-more changes DB only and never auto-resumes.
6. Explicit `검토 완료` resumes the same thread and reaches the correct terminal state.
7. Repeating launch/approval/resume creates no duplicate AgentRun, ReviewItem, knowledge, or companion Timeline effect.

Record bounded counts/status/version, never raw source/model content.

- [ ] **Step 10: Update product truth and handoff with actual evidence**

In `plan.md`, mark Deliverable C implemented only after every required gate above passes and keep Slack last. In `docs/portfolio-log.md`, record the two-screen UX, real interrupt/resume, shared-batch/exactly-once protection, actual LangChain adapter path, and exact verification counts. In `docs/superpowers/runbooks/session-handoff.md`, record commit ids, flag/rollback behavior, PostgreSQL test target requirements, remaining known failures, and the next approved/unapproved boundary.

Do not copy planned expectations as if they were observed results.

- [ ] **Step 11: Commit the release evidence**

```powershell
git diff --check
git add backend/tests/test_review_v2_postgres.py backend/tests/test_review_transition_postgres.py plan.md docs/portfolio-log.md docs/superpowers/runbooks/session-handoff.md
git commit -m "test: verify review workflow v2 release"
git status --short
```

Expected: commit succeeds and the worktree is clean. Do not push, merge, enable the production flag, or open a PR unless the user separately requests it.

## Final Implementation Review Checklist

- [ ] Every task's RED command was observed before its production edit and the corresponding GREEN command was rerun after it.
- [ ] No new migration, table, column, foreign key, outbox, broker, CDC, streaming consumer, Slack change, RAG call, Neo4j integration, or Knowledge Map code was added.
- [ ] Production model construction reaches the existing LangChain adapters; tests/demo remain deterministic and no live provider was called.
- [ ] The compiled workflow uses actual LangGraph `interrupt()` and same-thread `Command(resume=...)` at root checkpoint namespace.
- [ ] Checkpoint/audit/V2 summary scans contain no source text/ref, ReviewItem id, prompt/model output, raw error, or credential.
- [ ] PostgreSQL ReviewItem rows and current permission are authoritative at status and resume.
- [ ] Scope-shared exact batches, creator-only client keys, effect/candidate keys, and provenance constraints prevent every specified duplicate.
- [ ] All five Review action paths share one locked transition service and terminal transitions obey the approved state table.
- [ ] Integrations and Review are the only primary screens; there is no agent selector, extra wizard, auto-resume, or Agent Runs navigation.
- [ ] V2 remains disabled by default, PostgreSQL never falls back to memory, and existing V2 threads never fall back to V1.
- [ ] Focused C, complete B, PostgreSQL, non-Slack, frontend, and full-suite comparison evidence is current and recorded truthfully.
- [ ] The ten deferred Slack failures remain visible and Slack remediation remains the final phase.

## Execution Handoff

After this plan is reviewed and explicitly authorized, choose one execution mode:

1. **Subagent-Driven Development (recommended):** Stay in this task, invoke `superpowers:subagent-driven-development`, give each independent task a fresh implementation worker, and run spec/code review checkpoints after every task.
2. **Inline Plan Execution:** Stay in this task, invoke `superpowers:executing-plans`, execute the tasks sequentially with the named verification and commit checkpoints.

No product implementation begins merely because this planning document exists. The user's execution-mode choice is the coding authorization boundary.
