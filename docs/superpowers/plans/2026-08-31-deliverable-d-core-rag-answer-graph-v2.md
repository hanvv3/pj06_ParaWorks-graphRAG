# Deliverable D Core RAG Answer Graph V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ParaWorks' hard-coded RAG orchestration path with a real LangChain 1.x retriever port and a real LangGraph 1.x answer graph while preserving the V1 `/ask`, `/search`, and Assistant user experience, evidence authority, permissions, and exact cost accounting.

**Architecture:** Keep the existing V1 service intact behind a deployment-static facade. Add a separately versioned, request-local `StateGraph` registered by exact workflow/version, LangChain `Runnable` retrievers for keyword and pgvector, a canonical evidence resolver, strict structured output, an exact-two-component cost ledger, and two-phase fresh projection. Cut over `/ask`, then `/search`, then Assistant through `disabled | shadow | enforce`; D Core performs no answer reuse and does not depend on Redis, Neo4j, Slack, or CDC.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL + pgvector, SQLite provider-free smoke mode, LangChain `>=1.3.17,<1.4.0`, LangGraph `>=1.2.11,<1.3.0`, `langchain-openai >=1.6.0,<1.7.0`, Pydantic 2, tiktoken, pytest 9, Ruff, Next.js 16, React 19, TypeScript, ESLint, and Playwright.

**Spec:** `docs/superpowers/specs/2026-08-30-deliverable-d-core-rag-answer-graph-v2-design.md`

## Global Constraints

- This document is the implementation plan, not implementation authorization. Do not change production code until the user separately approves this plan.
- Execute every behavior change test-first: add one bounded failing test, run it and capture the expected failure, add the minimum production code, rerun green, then commit. Never combine several unobserved RED steps into one implementation batch.
- Start every newly created Python module in this plan with `from __future__ import annotations`; keep cross-module DTO/protocol annotations import-safe and prove import/startup readiness before behavior tests.
- Use the actual libraries. The V2 retrieval adapters must implement LangChain `Runnable[RetrievalRequest, RetrievalResult]`; orchestration must compile an actual LangGraph `StateGraph` with conditional edges. A Python `if` chain or custom callable must not be represented as a LangGraph implementation.
- Preserve exact graph identity: `company-memory-rag-answer`, `company-memory-rag-answer-v2.0`, `rag-graph-state:v2`, and prompt `rag-answer:v2`. Keep public agent name `rag_orchestrator_agent`.
- Keep `GraphVersionRegistry`/checkpointing and the sealed five-manifest Review registry unchanged. D Core owns separate `RagGraphRegistry` and application manifest registries and receives no checkpointer.
- Preserve V1 success key sets and nullability for `/ask`, `/search`, and Assistant. Do not expose slot IDs, serving identities, graph versions, fallback traces, internal HMACs, or cost children.
- Filter by `SecurityScope` and canonical permission before any provider call. Model output never creates citation authority; only a fresh `CanonicalEvidenceProjector` may emit V1 citations and search rows.
- Trusted knowledge is first. Raw `gmail | gmail_attachment | drive | calendar` evidence may only be `source_observation` after exact current-version/parser/signature checks. Pending candidates, Slack, source-less output, unknown permissions, and incomplete provenance remain excluded.
- D Core has no answer-cache lookup/write/reuse. Do not add Redis L2, ordinary LangChain cache, process-memory answer cache, Neo4j GraphRAG, CDC/streaming, Slack reconstruction, or D.1 behavior.
- Automated tests must never call a live LLM, embeddings API, Slack, Gmail, Drive, Calendar, OAuth provider, or public network. Use fake LangChain chat models, fake HTTP transports, deterministic embeddings, and local PostgreSQL/SQLite fixtures.
- Production provider identities are frozen to OpenAI direct standard-global only: `text-embedding-3-small` with 1,536 dimensions for `query_embedding`, and `gpt-5.4-mini-2026-03-17` with `reasoning_effort="none"` for `answer_generation`. SDK retry, provider/model fallback, tools, tracing, callbacks, streaming, storage, and global cache are disabled.
- Each request may dispatch `query_embedding` at most once and `answer_generation` at most once. A future V2 `AgentRun` has exactly two immutable component rows, even when either component is terminal zero.
- The frozen first live gate remains 30 sanitized cases, at most 30 generation calls, 10 embedding calls, 40 total provider dispatches, USD `0.012000` per case, and USD `0.360000` committed reserve total. The user's USD 100 balance is availability, not permission to enlarge this gate.
- Do not run the live gate merely because implementation is green. First commit the clean runner and fixture, run every provider-free gate, produce a zero-call preview bound to exact commit/fixture/DB/authority identities, show its maximum cost, and obtain a fresh explicit execution confirmation. Crash, partial execution, rerun, production traffic, raw-observation reindexing, D.1, or E each requires fresh authorization.
- Preserve SQLite only as a single-process, provider-free keyword smoke oracle. PostgreSQL + pgvector remains the production write and paid-call authority.
- Preserve unrelated user and teammate changes. Check `git status --short` before and after each task; stage only files listed by the current task.
- Every task commit must be independently green. Use the suggested commit subject unless the implementation reveals a narrower truthful subject.
- Command convention: every abbreviated `uv run ...` command below must be executed as `uv run --no-cache --locked ...`. The explicit flags in Tasks 26–27 are the canonical form; no task may substitute a mutable dependency cache or an unlocked environment.

## Frozen Public and Internal Identities

```python
RAG_WORKFLOW_NAME = 'company-memory-rag-answer'
RAG_GRAPH_VERSION = 'company-memory-rag-answer-v2.0'
RAG_STATE_SCHEMA_VERSION = 'rag-graph-state:v2'
RAG_PROMPT_VERSION = 'rag-answer:v2'
RAG_RUN_CONTRACT_VERSION = 'rag-run:v2'
RAG_OUTPUT_SCHEMA_NAME = 'rag_answer_blocks_v1'
RAG_RENDER_CAPABILITY = 'rag-v2-plain-text-citations:v1'
RAG_LIVE_GATE_CONTRACT_VERSION = 'rag-live-gate:v1'
RAG_LIVE_FIXTURE_VERSION = 'rag-live-quality-30:v1'
```

```text
Cutover order: disabled -> keyword shadow -> pgvector shadow -> /ask enforce
               -> /search enforce -> Assistant enforce -> D Core green
Next after D Core green: separately design/plan/approve D.1 PostgreSQL answer cache.
```

---

## Phase A — Contracts, Identity, and Storage

### Task 1: Freeze V2 Contracts, Configuration, and Separate Registries

**Files:**

- Create: `backend/app/agent_runtime/rag_v2_contracts.py`
- Create: `backend/app/agent_runtime/rag_v2_registry.py`
- Create: `backend/tests/test_rag_v2_contracts.py`
- Create: `backend/tests/test_rag_v2_registry.py`
- Modify: `backend/app/agent_runtime/__init__.py`
- Modify: `backend/app/agents/rag_orchestrator_agent/agent.py`
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/main.py`
- Modify: `.env.example`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`
- Modify: `backend/tests/test_langchain_langgraph_dependency_compat.py`

**Interfaces:**

```python
RagSurface = Literal['search', 'ask', 'assistant']
RagMode = Literal['disabled', 'shadow', 'enforce']
RagCutoverStage = Literal['none', 'ask', 'search', 'assistant']
RagRetrievalBackend = Literal['keyword', 'pgvector']
RagEffectiveBackend = Literal['deterministic_lexical', 'pgvector']

class ProviderDispatchPermit(Protocol):
    """One-use runtime dispatch capability; live-gate composition is separate."""
    def consume_at_dispatch(self) -> None: ...

@dataclass(frozen=True, slots=True)
class RagGraphRegistration:
    workflow_name: str
    graph_version: str
    state_schema_version: str
    graph: CompiledStateGraph

class RagGraphRegistry:
    def register(self, registration: RagGraphRegistration) -> None: ...
    def resolve(self, workflow_name: str, graph_version: str) -> CompiledStateGraph: ...

def build_rag_manifest_registry() -> AgentRegistry: ...
def resolved_rag_mode(settings: Settings) -> RagMode: ...
def resolved_rag_stage(settings: Settings) -> RagCutoverStage: ...
def resolved_rag_backend(settings: Settings) -> RagRetrievalBackend: ...
```

- [ ] Add RED tests for frozen literal values, exact manifest fields, invalid configuration rejection, legacy `LANGGRAPH_RAG_V2_ENABLED=true` mapping only to `shadow`, canonical-mode precedence, and `RAG_RETRIEVAL_BACKEND` precedence over `RAG_USE_PGVECTOR_SEARCH`.
- [ ] Add RED registry tests for exact version resolution, duplicate registration failure, unknown/missing version mapping to `runtime_version_unavailable`, no `latest` alias, and no mutable request data on a registration.
- [ ] Add RED lifespan tests proving `app.state.rag_graph_registry` and `app.state.agent_manifest_registry` are separate from `app.state.agent_graph_registry` and the sealed exact-five `app.state.review_agent_registry`; RAG must be absent from the Review registry.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_contracts.py backend/tests/test_rag_v2_registry.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_langchain_langgraph_dependency_compat.py -q` and confirm failures are imports/contracts, not unrelated setup failures.
- [ ] Implement the immutable aliases/constants, exact-version registry, manifest builder, and settings resolution. Keep the existing `langgraph_rag_v2_enabled` field as a read-only migration alias; add canonical `langgraph_rag_v2_mode`, `langgraph_rag_v2_stage`, and `rag_retrieval_backend` fields with fail-closed validators.
- [ ] Change the RAG manifest to exact D Core values: input `RagGraphInput`, output `RagGraphOutput`, prompt versions `('rag-answer:v1', 'rag-answer:v2')`, permissions `('public', 'internal', 'restricted')`, and unchanged stable public name.
- [ ] Expose the separate manifest registry in application composition without adding RAG to Review services. Register the production graph only after Task 14 supplies the compiled graph; until then registry/lifespan tests inject an explicit test graph factory.
- [ ] Document only non-secret rollout variables in `.env.example`; leave real `.env` untouched and defaults `mode=disabled`, `stage=none`, `backend=keyword`.
- [ ] Rerun the focused pytest command and require all green.
- [ ] Run `uv run ruff check backend/app/agent_runtime/rag_v2_contracts.py backend/app/agent_runtime/rag_v2_registry.py backend/app/agents/rag_orchestrator_agent/agent.py backend/app/core/config.py backend/app/main.py backend/tests/test_rag_v2_contracts.py backend/tests/test_rag_v2_registry.py`.
- [ ] Commit with `git commit -m "feat: freeze rag v2 runtime contracts"`.

### Task 2: Prepare Safe Request Text and Security Scope

**Files:**

- Create: `backend/app/agent_runtime/rag_v2_identity.py`
- Create: `backend/app/agents/rag_orchestrator_agent/v2_input.py`
- Create: `backend/tests/test_rag_v2_identity.py`
- Create: `backend/tests/test_rag_v2_input.py`
- Create: `backend/tests/test_rag_v2_security_scope.py`
- Modify: `backend/app/agent_runtime/__init__.py`
- Reuse unchanged: `backend/app/agent_runtime/auto_review_input_safety.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class SecurityScope:
    contract_version: Literal['rag-security-scope:v1']
    principal_subject: str  # transient; never persisted
    workspace_scope_id: str
    resource_scope_mode: Literal['all_current_scope', 'constrained']
    project_constraints: tuple[str, ...]  # canonical project_key:<value>
    source_constraints: tuple[str, ...]  # canonical source_pk:<positive id>
    allowed_permission_levels: tuple[Literal['public', 'internal', 'restricted'], ...]
    auth_policy_version: str
    permission_policy_version: str

@dataclass(frozen=True, slots=True)
class PreparedRagRequestText:
    caller_text: str
    normalized_current_user_text: str
    retrieval_query_text: str
    answer_question_text: str
    query_context_version: Literal['direct-query:v1', 'assistant-context:v1']
    current_text_hmac: str
    retrieval_query_hmac: str
    answer_question_hmac: str

@dataclass(frozen=True, slots=True)
class AssistantContextMessage:
    message_id: int
    created_at: datetime
    role: Literal['user', 'assistant']
    content: str

class StrictUnicodeScalarValidator:
    @staticmethod
    def validate(value: str) -> str: ...

class RagPublicCitationUrlValidator:
    policy_version = 'rag-public-citation-url:v1'
    def validate(self, value: str) -> str: ...

def exact_utf8_bytes(value: str) -> dict[str, int | str]: ...
class RagSecurityScopeResolver(Protocol):
    def resolve(self, *, db: Session, actor: DemoUser) -> SecurityScope: ...
def security_scope_fingerprint(scope: SecurityScope, *, settings: Settings) -> str: ...
def prepare_direct_request_text(text: str, *, key: bytes) -> PreparedRagRequestText: ...
def prepare_assistant_request_text(
    current_text: str,
    prior_messages: Sequence[AssistantContextMessage],
    *,
    key: bytes,
) -> PreparedRagRequestText: ...
```

- [ ] Add RED Unicode tests for high/low/reversed surrogate, U+0000, strict UTF-8 failure, non-BMP scalar counting, NFC helper copy, exact caller-byte preservation, decomposed/composed HMAC distinction, and normalized-copy equality.
- [ ] Add RED URL tests for absolute HTTP(S), required host, exact stored bytes, and rejection of userinfo, controls, whitespace, protocol-relative, `javascript:`, `data:`, `file:`, invalid ports, and ambiguous parsing.
- [ ] Add RED direct-input tests proving `/ask`/`/search` whitespace and 4,001+ characters remain accepted by the V1 input contract, caller text is the exact echo/query, and normalized text is only scanner/fingerprint support.
- [ ] Add RED Assistant context-builder golden tests for chronological `(created_at,id)`, lowercase `user|assistant`, current duplicate removal, prior-assistant exclusion, last six unique prior user rows, exact `최근 대화:`/`user: `/`현재 질문: ` literals, LF join, 500/501 scalar ellipsis, oldest-whole-line drop at 8,000 scalars, and no summary authority.
- [ ] Add RED credential tests that reuse `credential-scan:v1`: scan caller and normalized copies, remove a credential-bearing prior whole message, reject a credential-bearing current turn, and rescan the final contextual query. Assert no plaintext is returned in exceptions.
- [ ] Add RED scope tests for server resolution, exact `all_current_scope|constrained` empty semantics, canonical constraint order/namespaces, exact allowed permissions, scope fingerprint mutation, raw-principal non-persistence, and serialized-scope/fingerprint mismatch causing zero query/embedding/DB calls.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_identity.py backend/tests/test_rag_v2_input.py backend/tests/test_rag_v2_security_scope.py -q` and confirm the new contracts fail before implementation.
- [ ] Implement exact-byte HMAC envelopes with existing `keyed_fingerprint`; never pass caller strings through the existing NFC-normalizing generic helper when exact bytes are required.
- [ ] Implement `SecurityScope` validation with exact contract/policy versions, transient principal, exact workspace, explicit `all_current_scope|constrained` mode, ordered canonical `project_key:*` and `source_pk:*` constraints, sorted exact permission tuple, and narrowing-only semantics. Bind explicit empty lists with the mode; never treat missing/null/empty as aliases or persist the raw principal/scope payload.
- [ ] Implement direct and Assistant builders as pure functions. Keep prior Assistant text and `conversation.summary` out of both V2 provider inputs; keep prior user context retrieval-only and current user text answer-only.
- [ ] Map typed internal failures only in later facade tasks; do not change V1 routes in this task.
- [ ] Rerun the focused tests and `uv run ruff check backend/app/agent_runtime/rag_v2_identity.py backend/app/agents/rag_orchestrator_agent/v2_input.py backend/tests/test_rag_v2_identity.py backend/tests/test_rag_v2_input.py backend/tests/test_rag_v2_security_scope.py`.
- [ ] Commit with `git commit -m "feat: prepare safe rag v2 request identity"`.

### Task 3: Add D Serving Projection Persistence

**Files:**

- Create: `backend/app/models/rag_serving.py`
- Create: `backend/migrations/versions/d1a2b3c4e5f6_add_rag_serving_projection.py`
- Create: `backend/tests/test_rag_v2_models.py`
- Create: `backend/tests/test_rag_v2_migration.py`
- Modify: `backend/app/models/vector_index.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/tests/test_agent_runtime_migration.py`

**Interfaces:**

```python
class RagServingCorpusGeneration(Base):
    __tablename__ = 'rag_serving_corpus_generations'
    # actor-independent singleton: corpus_generation, vector_index_generation,
    # embedding/index-policy identities and key verifier

class RagLexicalServingProjection(Base):
    __tablename__ = 'rag_lexical_serving_projections'
    # serving identity/version/content/citation hashes plus lowercase searchable fields
```

- [ ] Add RED model tests for the actor-independent corpus singleton, lexical projection identity/contract fields, D provenance/readiness fields on `VectorIndexState`, generation monotonicity, and immutable serving identity/version HMACs.
- [ ] Add RED migration tests for the first D revision with `revision='d1a2b3c4e5f6'`, `down_revision='9d7f3a1c6e20'`, exact serving/index tables/columns/indexes/checks/FKs, and no data-destructive backfill.
- [ ] Add RED PostgreSQL tests for singleton enforcement, non-negative generation counters, lexical projection uniqueness, D vector-state provenance/readiness constraints, and named `rag_python_round6_binary64_v1`/`rag_python_lexical_score_v1` functions. Do not activate mutation guard triggers in this first revision: existing writers are updated in Task 5 and Task 11's later revision installs the deferred database guards, preserving an independently green Task 3 commit.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_models.py backend/tests/test_rag_v2_migration.py backend/tests/test_agent_runtime_migration.py -q` and observe schema failures.
- [ ] Implement only serving/index projection persistence in this revision. Do not add cost, provider-safety, release, AgentRun, or Assistant-integrity fields here; those belong to the runtime-safety revision in Task 11.
- [ ] Add exact database constraints for canonical serving-document identity, policy/HMAC lengths, known permissions/support modes, and generation counters. Historical vector rows stay readable and are not silently retagged as D-ready.
- [ ] Rerun focused tests on SQLite and the existing disposable PostgreSQL migration fixture.
- [ ] Run `uv run ruff check backend/app/models/rag_serving.py backend/app/models/vector_index.py backend/migrations/versions/d1a2b3c4e5f6_add_rag_serving_projection.py backend/tests/test_rag_v2_models.py backend/tests/test_rag_v2_migration.py`.
- [ ] Commit with `git commit -m "feat: add rag serving projection schema"`.

---

## Phase B — Canonical Evidence and Retrieval

### Task 4: Resolve Canonical Raw and Trusted Serving Evidence

**Files:**

- Create: `backend/app/rag/serving_contracts.py`
- Create: `backend/app/rag/source_observations.py`
- Create: `backend/app/rag/trusted_evidence.py`
- Create: `backend/tests/test_rag_source_observations.py`
- Create: `backend/tests/test_rag_trusted_evidence.py`
- Modify: `backend/app/knowledge/trusted_serving_eligibility.py`
- Reuse unchanged: `backend/app/ingestion/source_authority.py`
- Reuse unchanged: `backend/app/knowledge/serving_text.py`
- Reuse unchanged: `backend/app/agent_runtime/keyed_mutation_guard.py`

**Interfaces:**

```python
SupportMode = Literal['trusted_fact', 'source_observation']
ServingKind = Literal['raw_chunk', 'trusted_knowledge']

@dataclass(frozen=True, slots=True)
class RawServingVersionEnvelope:
    serving_document_id: str  # chunk:<DocumentChunk.id>
    source_row_id: int
    public_source_id: str
    document_id: int
    document_version_id: int
    current_document_version_id: int
    document_chunk_id: int
    parser_run_id: int
    external_revision: str | None
    server_content_signature_schema: str
    server_content_signature: str
    parser_policy_version: str
    parser_version: str
    chunk_policy_version: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    effective_permission: Literal['public', 'internal', 'restricted']

@dataclass(frozen=True, slots=True)
class RawChunkProvenance:
    branch: Literal['raw_chunk']
    raw_version: RawServingVersionEnvelope

@dataclass(frozen=True, slots=True)
class TrustedEvidenceLinkIdentity:
    trusted_knowledge_evidence_link_id: int
    approval_link_id: int
    canonical_source_kind: str
    canonical_source_id: str
    canonical_version_or_signature: str
    evidence_hash: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str

@dataclass(frozen=True, slots=True)
class SelectedCitationChild:
    trusted_knowledge_evidence_link_id: int
    source_row_id: int
    canonical_source_type: str
    canonical_version_or_signature: str
    review_item_source_pair_ordinal: int

@dataclass(frozen=True, slots=True)
class ExplicitApprovalProvenance:
    branch: Literal['explicit_approval']
    approval_link_id: int
    review_item_id: int
    security_scope_id: str
    promotion_effect_kind: str
    resolution_source: Literal['human', 'auto_policy']
    claim_fingerprint: str
    approval_permission_level: Literal['public', 'internal', 'restricted']
    approval_fingerprint_key_version: str
    approval_fingerprint_key_material_verifier: str
    evidence_links: tuple[TrustedEvidenceLinkIdentity, ...]
    selected_citation_child: SelectedCitationChild

@dataclass(frozen=True, slots=True)
class LegacyHumanProvenance:
    branch: Literal['legacy_human_base']
    legacy_binding: Literal['review_item']
    legacy_evidence_pairs_hmac: str
    legacy_review_item_permission_level: Literal['public', 'internal', 'restricted']
    legacy_source_review_item_id: int

@dataclass(frozen=True, slots=True)
class TrustedServingVersionEnvelope:
    serving_document_id: str
    knowledge_type: Literal['decision_record', 'history_event', 'timeline_event', 'todo']
    knowledge_id: int
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    effective_permission: Literal['public', 'internal', 'restricted']
    provenance: ExplicitApprovalProvenance | LegacyHumanProvenance

ServingVersionEnvelope = RawServingVersionEnvelope | TrustedServingVersionEnvelope

@dataclass(frozen=True, slots=True)
class ServingEvidenceIdentity:
    serving_document_id: str
    serving_kind: ServingKind
    public_source_id: str
    public_source_type: str | None
    effective_permission: Literal['public', 'internal', 'restricted']
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    serving_version_fingerprint: str
    version_envelope: ServingVersionEnvelope

@dataclass(frozen=True, slots=True)
class ServingEvidence:
    serving_document_id: str
    serving_kind: ServingKind
    public_source_id: str
    public_source_type: str | None
    support_mode: SupportMode
    model_content: str
    title: str
    effective_permission: Literal['public', 'internal', 'restricted']
    serving_identity_hmac: str
    serving_version_fingerprint: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    version_envelope: ServingVersionEnvelope
    provenance: RawChunkProvenance | ExplicitApprovalProvenance | LegacyHumanProvenance

@dataclass(frozen=True, slots=True)
class IndexableSourceObservation:
    identity: ServingEvidenceIdentity
    evidence: ServingEvidence
    raw_version: RawServingVersionEnvelope

@dataclass(frozen=True, slots=True)
class TrustedServingEnvelope:
    identity: ServingEvidenceIdentity
    evidence: ServingEvidence
    trusted_version: TrustedServingVersionEnvelope

@dataclass(frozen=True, slots=True)
class EvidenceAccessClassification:
    global_eligibility: Literal['eligible', 'ineligible']
    resource_scope: Literal['in_scope', 'out_of_scope', 'invalid_scope']
    permission_visibility: Literal['visible', 'denied_known', 'unknown_permission']

class CanonicalSourceObservationResolver:
    def resolve_for_index(self, chunk_id: int) -> IndexableSourceObservation | None: ...

class CanonicalSourceObservationEligibilityService:
    def classify_access(self, scope: SecurityScope, observation: IndexableSourceObservation) -> EvidenceAccessClassification: ...

class TrustedServingEnvelopeResolver:
    def resolve_for_index(self, knowledge_type: str, knowledge_id: int) -> TrustedServingEnvelope | None: ...

class TrustedEvidenceAuthorizer:
    def classify_access(self, scope: SecurityScope, evidence: TrustedServingEnvelope) -> EvidenceAccessClassification: ...

class ServingEvidenceResolver:
    def resolve_candidate(self, *, db: Session, identity: ServingEvidenceIdentity, scope: SecurityScope) -> ServingEvidence | None: ...
```

- [ ] Add RED raw-evidence tests for exact current `DocumentVersion`/parser-run/chunk lineage, server content signature, canonical nonblank snippet, source kinds `gmail | gmail_attachment | drive | calendar`, source URL validation, strictest source/document/chunk permission, project/source scope, and Slack/pending/stale/tombstoned exclusion.
- [ ] Add RED trusted-evidence tests for all-active-link global eligibility, actor authorization before supporting-link selection, deterministic human-before-auto link choice, all-child source constraint, strictest permission across target/provenance/ReviewItem, and no leakage of an unauthorized higher-priority link.
- [ ] Add RED explicit citation-child tests for ordered evidence links, lowest exact ReviewItem pair ordinal, current Source/version match, exact snippet/signature, knowledge type remaining the public source type, and whole-candidate exclusion on ambiguity or drift.
- [ ] Add RED canonical identity mapping tests for `chunk:{id}` -> exact `Source.source_id`, `decision:` -> `decision_record:`, and same-but-separate trusted internal/public IDs. Assert the exact `serving_kind`, `public_source_id`, `public_source_type`, raw/trusted version-envelope type, and one-to-one identity/evidence/envelope consistency.
- [ ] Add RED legacy-human tests for the one allowed linked `ReviewItem` branch and rejection of `source_review_item_id=null`, C.5 candidate contracts, empty/mismatched arrays, unknown permission, or synthetic source/version reconstruction.
- [ ] Add RED golden HMAC tests for raw/trusted envelopes, provenance, model content, canonical citation, evidence-link set, selected child, key rotation, dynamic UTF-8 envelopes, and one-field/order/null/type mutation.
- [ ] Run `uv run pytest backend/tests/test_rag_source_observations.py backend/tests/test_rag_trusted_evidence.py -q` and confirm the resolvers are absent/failing.
- [ ] Implement one resolver shared by keyword, pgvector, final projection, shadow, indexing guards, and Assistant revalidation. It may return classifications, but only the visible branch may carry public citation bytes.
- [ ] Reuse `TrustedServingEligibilityService` for global eligibility without weakening C.5. Add a D-specific envelope method rather than changing legacy eligibility semantics.
- [ ] Keep vector metadata as lookup hints only. Reject document/chunk identity disagreement rather than repairing it.
- [ ] Rerun the focused suite and `uv run ruff check backend/app/rag/serving_contracts.py backend/app/rag/source_observations.py backend/app/rag/trusted_evidence.py backend/app/knowledge/trusted_serving_eligibility.py backend/tests/test_rag_source_observations.py backend/tests/test_rag_trusted_evidence.py`.
- [ ] Commit with `git commit -m "feat: resolve canonical rag serving evidence"`.

### Task 5: Add Incremental Raw Observation and Lexical Projection Indexing

**Files:**

- Create: `backend/app/rag/serving_generation.py`
- Create: `backend/app/rag/lexical_projection.py`
- Create: `backend/app/rag/vector_validation.py`
- Create: `backend/app/rag/index_readiness.py`
- Create: `backend/tests/test_rag_v2_indexing.py`
- Create: `backend/tests/test_rag_v2_index_readiness.py`
- Modify: `backend/app/rag/indexing.py`
- Modify: `backend/app/agent_runtime/keyed_mutation_guard.py`
- Modify: `backend/app/documents/service.py`
- Modify: `backend/app/ingestion/service.py`
- Modify: `backend/app/knowledge/promotion.py`
- Modify: `backend/app/knowledge/trusted_provenance.py`
- Modify: `backend/app/review/transitions.py`
- Modify: `backend/app/review/auto_review_quality_revoke.py`
- Modify: `backend/app/review/auto_review_source_reconciliation.py`
- Modify: `backend/app/rag/serving_locks.py`
- Modify: `backend/app/rag/pgvector_store.py`
- Modify: `backend/app/tasks/rag_indexing.py`
- Modify: `backend/app/rag/reindexing.py`
- Modify: `backend/tests/test_rag_indexing.py`
- Modify: `backend/tests/test_pgvector_store.py`

**Interfaces:**

```python
RAG_INDEX_POLICY_VERSION = 'rag-v2-serving-index:v1'
RAG_LEXICAL_COMPAT_VERSION = 'rag-keyword-lexical-compat:v1'
RAG_COSINE_POLICY_VERSION = 'pgvector-cosine-indexable:v1'

@dataclass(frozen=True, slots=True)
class RagIndexMutationResult:
    indexed_count: int
    skipped_count: int
    tombstoned_count: int
    saved_embedding_calls: int
    corpus_generation: int
    vector_index_generation: int

@dataclass(frozen=True, slots=True)
class RagServingIndexReadiness:
    ready: bool
    corpus_generation: int
    vector_index_generation: int
    expected_document_count: int
    live_vector_count: int
    tombstone_count: int
    mismatch_count_capped_at_20: int
    embedding_model: str
    embedding_dimensions: int
    index_policy_version: str
    readiness_snapshot_hmac: str

class RagV2ServingIndexReadinessService:
    def inspect(self, *, db: Session) -> RagServingIndexReadiness: ...

CanonicalFloat32Vector: TypeAlias = tuple[float, ...]  # validator guarantees exact expected length, finite IEEE-754 float32 values, and at least one nonzero

class CosineIndexableVectorValidator:
    policy_version = 'pgvector-cosine-indexable:v1'
    def validate(self, vector: Sequence[object], *, expected_dimensions: int) -> CanonicalFloat32Vector: ...
    def validate_batch(self, vectors: Sequence[Sequence[object]], *, expected_dimensions: int) -> tuple[CanonicalFloat32Vector, ...]: ...
```

- [ ] Add RED tests proving `build_rag_index_documents()` still preserves legacy documents while the D-specific builder includes only eligible raw observations and trusted knowledge with exact D provenance metadata.
- [ ] Add RED incremental tests for stable content-hash skip before provider work, batch embedding only after skip checks, one invalid/zero/non-finite vector rejecting the entire batch, no raw live write on SQLite, and reported indexed/skipped/tombstoned/saved counts.
- [ ] Add RED generation-lock tests for writer order `AUTO_REVIEW_KEY_GENERATION_LOCK_ID -> AutoReviewRuntimeKeyState -> RagServingCorpusGeneration`, same-transaction corpus increment on raw/trusted eligibility/content/permission/revoke plus lexical projection refresh, and vector-index increment only on D vector write or tombstone.
- [ ] Add RED service-integration and SQLite transaction-assertion tests for every source/document/version/chunk, trusted promotion/reaffirm/revoke/permission, and D vector mutation path. These make all existing writers generation-safe before Task 11 activates equivalent PostgreSQL deferred guards; direct-SQL trigger refusal is deliberately owned by Task 11.
- [ ] Add RED lexical-projection tests for exact Python-lowered title/searchable bytes, content/version/citation HMACs, named scorer versions, literal `%`/`_`, duplicate term ordinality, and stale/missing projection exclusion.
- [ ] Add RED actor-independent pgvector readiness tests for expected/live/tombstone sets, exact embedding model/dimensions/hash/cosine policy, mismatch count capped at 20, no raw IDs in traces, and `legacy_unbound` not being retagged.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_indexing.py backend/tests/test_rag_v2_index_readiness.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py -q` and observe the new policy/readiness failures.
- [ ] Implement D-specific document construction and post-provider write guards while reusing `index_changed_vector_documents()` and existing serving locks. Never make a production embedding call in a test.
- [ ] Update every canonical evidence mutation path so the source/trusted mutation, lexical projection refresh bound to the resulting `corpus_generation`, and shared corpus-generation increment commit in the same transaction. Only D vector write/tombstone increments `vector_index_generation`; lexical projection writes never do. Add lock-order and application transaction assertions for source, promotion, reaffirmation, revoke, permission, parser, vector, tombstone, and legacy service writers; Task 11 alone activates/tests direct-SQL database triggers.
- [ ] Make `dry_run=false` raw observation reindex remain PostgreSQL+pgvector-only and separately operator-authorized; this task builds code/tests but does not execute a live reindex.
- [ ] Rerun the focused tests, then `uv run ruff check backend/app/rag/serving_generation.py backend/app/rag/lexical_projection.py backend/app/rag/vector_validation.py backend/app/rag/index_readiness.py backend/app/rag/indexing.py backend/app/rag/serving_locks.py backend/app/rag/pgvector_store.py backend/tests/test_rag_v2_indexing.py backend/tests/test_rag_v2_index_readiness.py`.
- [ ] Commit with `git commit -m "feat: index canonical rag v2 evidence incrementally"`.

### Task 6: Port Keyword Retrieval to a LangChain Runnable

**Files:**

- Create: `backend/app/rag/retrieval.py`
- Create: `backend/app/rag/keyword_retriever.py`
- Create: `backend/tests/test_rag_v2_keyword_retriever.py`
- Modify: `backend/app/rag/search_store.py`
- Reuse unchanged: `backend/app/rag/serving_contracts.py`
- Reuse unchanged: `backend/app/rag/source_observations.py`
- Reuse unchanged: `backend/app/rag/trusted_evidence.py`

**Interfaces:**

```python
RagPaidComponent = Literal['query_embedding', 'answer_generation']

@dataclass(frozen=True, slots=True)
class StrictProviderUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass(frozen=True, slots=True)
class PreparedPaidCallBudget:
    component: RagPaidComponent
    estimated_input_tokens: int
    maximum_output_tokens: int
    reserved_cost_usd: Decimal
    cost_policy_snapshot_hmac: str
    estimator_input_hmac: str

@dataclass(frozen=True, slots=True)
class QueryEmbeddingCostInput:
    retrieval_query_utf8: bytes
    model_config_snapshot_hmac: str

@dataclass(frozen=True, slots=True)
class AnswerGenerationCostInput:
    exact_messages_json: bytes
    exact_response_schema_json: bytes
    model_config_snapshot_hmac: str

class EmbeddingUsageParserPort(Protocol):
    def parse_usage(self, usage: object) -> StrictProviderUsage: ...

class RagCostPolicyPort(Protocol):
    def prepare_query_embedding(self, value: QueryEmbeddingCostInput) -> PreparedPaidCallBudget: ...
    def charge_actual(self, component: RagPaidComponent, usage: StrictProviderUsage) -> Decimal: ...

@dataclass(frozen=True, slots=True)
class PreparedQueryEmbedding:
    retrieval_query_hmac: str
    transient_query_utf8: bytes  # request-local only; never logged/persisted
    corpus_generation: int
    vector_index_generation: int
    readiness_snapshot_hmac: str
    model_config_snapshot_hmac: str
    provider_policy_snapshot_hmac: str
    estimated_input_tokens: int
    reserved_cost_usd: Decimal
    attempt_fence_hmac: str
    budget: PreparedPaidCallBudget

@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    evidence: ServingEvidence
    relevance_score: float
    matched_terms: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class QueryEmbeddingReceipt:
    attempted: bool
    input_tokens: int
    actual_cost_usd: Decimal
    latency_ms: int
    outcome: str
    model_config_snapshot_hmac: str
    provider_policy_snapshot_hmac: str

@dataclass(frozen=True, slots=True)
class QueryEmbeddingCallResult:
    prepared: PreparedQueryEmbedding
    vector: CanonicalFloat32Vector
    attempted: Literal[True]
    validated_input_tokens: int
    actual_cost_usd: Decimal
    receipt: QueryEmbeddingReceipt

@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    retrieval_query_text: str
    security_scope: SecurityScope
    security_scope_fingerprint: str
    query_embedding_result: QueryEmbeddingCallResult | None
    candidate_scan_limit: Literal[50]
    visible_limit: Literal[5, 8]
    relevance_policy_version: Literal['rag-retrieval-policy:v2.0']

@dataclass(frozen=True, slots=True)
class SanitizedRetrievalTrace:
    candidate_window_count: int
    visible_count: int
    hidden_match_count: int
    provider_attempt_count: int
    latency_ms: int
    fallback_category: str | None

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    configured_backend: RagRetrievalBackend
    effective_backend: RagEffectiveBackend
    visible: tuple[RetrievalCandidate, ...]
    hidden_match_count: int
    hidden_count_capped: bool
    top_candidate_window_hmac: str
    query_embedding_receipt: QueryEmbeddingReceipt | None
    trace: SanitizedRetrievalTrace

class KeywordEvidenceRetriever(Runnable[RetrievalRequest, RetrievalResult]):
    def invoke(self, input: RetrievalRequest, config: RunnableConfig | None = None) -> RetrievalResult: ...

class RagRetrieverRegistry:
    def register(self, backend: RagRetrievalBackend, retriever: Runnable[RetrievalRequest, RetrievalResult]) -> None: ...
    def resolve(self, backend: RagRetrievalBackend) -> Runnable[RetrievalRequest, RetrievalResult]: ...
```

- [ ] Add RED tests asserting `KeywordEvidenceRetriever` is an actual LangChain `Runnable`, accepts immutable typed input, and has no mutable result side channel.
- [ ] Add RED scorer golden tests for exact comma/period replacement, `split`, lower, pre-lower length `>=3`, duplicate/order preservation, exact phrase/title/coverage bonuses, binary64 six-place round, and score `>0` relevance.
- [ ] Add RED PostgreSQL/Python oracle parity tests for literal `%`/`_`, duplicate terms, Unicode lower/split, direct 1,001+ terms without dropping/refusal, statement timeout with no partial results, and no pre-score LIMIT. Assistant's contextual-query-only 1,000-term preflight belongs to Task 14 because the authoritative retriever request intentionally carries no surface.
- [ ] Add RED two-stage bound tests: resource/relevance top 50 first; known-permission denial second; visible 5 for search or 8 for answer; public hidden count capped at 20; unknown permission excluded from candidate and count.
- [ ] Add RED ordering tests for trusted tier before raw, then descending score, then stable serving ID; ensure denied identities never appear in public objects/logs.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_keyword_retriever.py -q` and confirm RED.
- [ ] Implement the production SQL adapter and SQLite Python oracle behind the same Runnable/result contract. Apply resource scope and a complete-superset coarse predicate in SQL before exact scoring; never materialize an unbounded production corpus in Python.
- [ ] Register keyword by explicit backend name and reject duplicate/unknown registry entries without fallback.
- [ ] Rerun the focused test and `uv run ruff check backend/app/rag/retrieval.py backend/app/rag/keyword_retriever.py backend/app/rag/search_store.py backend/tests/test_rag_v2_keyword_retriever.py`.
- [ ] Commit with `git commit -m "feat: port keyword retrieval to langchain runnable"`.

### Task 7: Port Pgvector Retrieval with One Strict Query Embedding

**Files:**

- Create: `backend/app/agents/rag_orchestrator_agent/v2_embedding.py`
- Create: `backend/tests/test_rag_v2_embedding.py`
- Create: `backend/tests/test_rag_v2_pgvector_retriever.py`
- Create: `backend/app/rag/pgvector_retriever.py`
- Modify: `backend/app/rag/retrieval.py`
- Modify: `backend/app/rag/embeddings.py`
- Modify: `backend/app/rag/pgvector_store.py`
- Modify: `backend/app/rag/search_store.py`
- Modify: `backend/tests/test_embedding_provider.py`
- Modify: `backend/tests/test_pgvector_integration.py`

**Interfaces:**

```python
class StrictQueryEmbeddingAdapter:
    def __init__(self, *, usage_parser: EmbeddingUsageParserPort, cost_policy: RagCostPolicyPort) -> None: ...
    def prepare(self, request: RetrievalRequest, readiness: RagServingIndexReadiness) -> PreparedQueryEmbedding: ...
    def dispatch_once(self, prepared: PreparedQueryEmbedding, permit: ProviderDispatchPermit) -> QueryEmbeddingCallResult: ...

class PgVectorEvidenceRetriever(Runnable[RetrievalRequest, RetrievalResult]):
    def invoke(self, input: RetrievalRequest, config: RunnableConfig | None = None) -> RetrievalResult: ...
```

- [ ] Add RED envelope/vector matrix tests: top object `list`, exact model, exactly one item, object `embedding`, integer index 0, exactly 1,536 finite non-bool values, float32 conversion, no overflow/underflow-to-all-zero, at least one nonzero coordinate, and canonical big-endian float32 SHA-256.
- [ ] Add RED adapter-boundary tests with fake `EmbeddingUsageParserPort` and `RagCostPolicyPort`: the adapter passes raw usage once, uses only the returned strict usage/Decimal charge, never coerces/recomputes them, and builds no SDK client or fallback. Task 9 supplies the production parsers/policy before graph composition.
- [ ] Add RED parse-precedence tests: usage overrun, then response identity, then vector, then usage/storage; ensure exactly one typed result, no vector/SQL/fallback on invalid response, and no fabricated usage failure for a response-less transport exception.
- [ ] Add RED retrieval tests for cosine score `>=0.25`, exact two-stage 50/permission/5-or-8 bounds, one immutable carrier shared by legacy/V2 shadow only when query bytes are identical, and one provider attempt maximum.
- [ ] Add RED readiness-fence tests proving `PreparedQueryEmbedding` binds the exact pre-call corpus/vector generation pair and anti-join readiness HMAC. Recheck the same tuple immediately before and after pgvector SQL; any drift discards the whole vector result, preserves validated embedding actual cost, recomputes keyword with the same `SecurityScope`, sets `serving_corpus_changed_during_pgvector_query`, and performs no second embedding.
- [ ] Add RED fallback tests proving only pgvector storage/search runtime failure after a valid carrier or the explicit post-embedding corpus-drift case takes the same-request keyword conditional edge; embedding failure/invalid payload/pre-call readiness failure never keyword-fallbacks.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_embedding.py backend/tests/test_rag_v2_pgvector_retriever.py backend/tests/test_embedding_provider.py backend/tests/test_pgvector_integration.py -q` and confirm RED without network access.
- [ ] Implement raw-result adapters with injected fake transport and explicit model/config identity. The actual durable cost claim/permit is supplied by Task 12; tests use a fake one-use permit and assert second dispatch fails.
- [ ] Update pgvector SQL to accept a validated vector carrier, not a query string or vector metadata authority. Fresh-read the generation/anti-join tuple on both sides of SQL, project through the canonical resolver, and return effective backend/fallback trace internally only.
- [ ] Rerun focused suites and `uv run ruff check backend/app/agents/rag_orchestrator_agent/v2_embedding.py backend/app/rag/retrieval.py backend/app/rag/pgvector_retriever.py backend/app/rag/embeddings.py backend/app/rag/pgvector_store.py backend/tests/test_rag_v2_embedding.py backend/tests/test_rag_v2_pgvector_retriever.py`.
- [ ] Commit with `git commit -m "feat: port pgvector retrieval with strict embeddings"`.

### Task 8: Rank, Slot, and Project Evidence from Canonical Rows

**Files:**

- Create: `backend/app/rag/evidence_projection.py`
- Create: `backend/tests/test_rag_v2_projection.py`
- Modify: `backend/app/rag/serving_contracts.py`
- Modify: `backend/app/rag/source_observations.py`
- Modify: `backend/app/rag/trusted_evidence.py`
- Modify: `backend/app/rag/retrieval.py`

**Interfaces:**

```python
EvidenceSlotId: TypeAlias = Literal['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']

@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    slot_id: EvidenceSlotId  # assigned contiguously from E1
    support_mode: SupportMode
    evidence: ServingEvidence
    relevance_score: float
    matched_terms: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class V1EvidenceProjection:
    citations: tuple[dict[str, object], ...]
    source_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    source_snippets: tuple[str, ...]
    search_results: tuple[dict[str, object], ...]
    projection_hmac: str

@dataclass(frozen=True, slots=True)
class PreparedModelInfluenceObservation:
    slot_id: EvidenceSlotId
    support_mode: SupportMode
    lookup_identity: ServingEvidenceIdentity
    effective_permission: Literal['public', 'internal', 'restricted']
    serving_identity_hmac: str
    serving_version_fingerprint: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    approval_provenance_hmac: str | None
    evidence_link_set_hmac: str | None
    observation_hmac: str

@dataclass(frozen=True, slots=True)
class ModelInfluenceDependencySnapshot:
    observation: PreparedModelInfluenceObservation
    dependency_role: Literal['selected_citation', 'unselected_model_influence']
    fresh_lookup_identity: ServingEvidenceIdentity
    dependency_hmac: str

class CanonicalEvidenceProjector:
    def project_search(self, ...) -> V1EvidenceProjection: ...
    def project_selected(self, ...) -> V1EvidenceProjection: ...
    def prepare_model_influence(self, ...) -> tuple[PreparedModelInfluenceObservation, ...]: ...
    def finalize_model_influence_dependencies(
        self,
        observations: tuple[PreparedModelInfluenceObservation, ...],
        selected_slot_ids: tuple[EvidenceSlotId, ...],
        ...,
    ) -> tuple[ModelInfluenceDependencySnapshot, ...]: ...
```

- [ ] Add RED slot tests for trusted-first deterministic order, max eight, contiguous `E1..En`, whole-record tail removal and renumbering, immutable prepared slots, and model-selected IDs being a subset of prepared IDs.
- [ ] Add RED HMAC goldens for citation, selected set, ordered search set including empty set, model-influence child/set, prepared influence observation, hidden membership, score binary64 bits, matched-term order, nullable key presence, and one-byte/order/role mutation. Prove the pre-generation observation binds the canonical lookup identity/version but has no dependency role; assign `selected_citation | unselected_model_influence` only after validated selected IDs exist.
- [ ] Add RED projection tests proving every public URL/ID/snippet/type/permission comes from a fresh canonical row, model-provided citation objects are ignored, search projects all visible bounded candidates, answer projects selected slots only, and all provider-visible slots are retained as influence dependencies.
- [ ] Add RED revalidation tests for corpus/index generation, source/version/permission/revoke/provenance drift, selected-membership drift, hidden-membership drift, invalid Unicode/URL/score, and entire-answer/search redaction rather than partial stale projection.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_projection.py -q` and confirm RED.
- [ ] Implement projection as a transaction-bound service; its DTO is immutable and fully assembled before the transaction commits. Routes must never rebuild it from vector metadata or state after commit.
- [ ] Implement public hidden count/capped semantics without persisting denied IDs; keep the capped flag internal.
- [ ] Rerun the focused suite and `uv run ruff check backend/app/rag/evidence_projection.py backend/app/rag/serving_contracts.py backend/app/rag/source_observations.py backend/app/rag/trusted_evidence.py backend/tests/test_rag_v2_projection.py`.
- [ ] Commit with `git commit -m "feat: project rag evidence from canonical rows"`.

---

## Phase C — Structured Generation, Cost Authority, and LangGraph

### Task 9: Freeze Strict Provider Usage and Cost Policy

**Files:**

- Create: `backend/app/agent_runtime/provider_usage.py`
- Create: `backend/app/agent_runtime/rag_cost_policy.py`
- Create: `backend/tests/test_provider_usage.py`
- Create: `backend/tests/test_rag_v2_cost_policy.py`
- Modify: `backend/app/agent_runtime/auto_review_validator.py`
- Modify: `backend/app/agent_runtime/model_router.py`
- Modify: `backend/tests/test_auto_review_validator.py`
- Modify: `backend/tests/test_auto_review_model_router.py`

**Interfaces:**

```python
RagRunProductOutcome = Literal[
    'supported', 'search_projected', 'no_match', 'hidden_only',
    'safety_filter_empty', 'insufficient_evidence', 'evidence_unavailable',
]
AssistantSafePersistedErrorOutcome = Literal[
    'budget_exceeded', 'retriever_not_configured', 'retriever_unavailable',
    'runtime_version_unavailable', 'model_unavailable', 'model_provider_failed',
    'provider_response_identity_invalid', 'provider_embedding_payload_invalid',
    'provider_usage_overrun', 'provider_safety_unavailable',
    'structured_output_invalid', 'citation_validation_failed',
    'unexpected_internal_error',
]
RagResultOutcome = RagRunProductOutcome | AssistantSafePersistedErrorOutcome
AssistantProductOutcome = Literal[
    'supported', 'no_match', 'hidden_only', 'safety_filter_empty',
    'insufficient_evidence', 'evidence_unavailable',
]
AssistantPersistedOutcome = AssistantProductOutcome | AssistantSafePersistedErrorOutcome
AssistantApplicationOutcome = AssistantPersistedOutcome | Literal[
    'persistence_failed', 'commit_unknown', 'permission_denied',
    'owner_not_found', 'invalid_input', 'input_safety_blocked',
    'input_scanner_unavailable',
]
RagCannedMessageIdentity = Literal[
    'rag-canned-no-evidence:v1', 'rag-canned-evidence-unavailable:v1',
    'rag-canned-budget-failure:v1', 'rag-canned-generation-failure:v1',
]

class StrictChatUsageParser:
    def parse_message(self, message: object) -> StrictProviderUsage: ...

class StrictEmbeddingUsageParser:
    def parse_usage(self, usage: object) -> StrictProviderUsage: ...

class RagCostPolicy:
    def prepare_query_embedding(self, value: QueryEmbeddingCostInput) -> PreparedPaidCallBudget: ...
    def prepare_answer_generation(self, value: AnswerGenerationCostInput) -> PreparedPaidCallBudget: ...
    def charge_actual(self, component: RagPaidComponent, usage: StrictProviderUsage) -> Decimal: ...
```

- [ ] Add RED strict chat usage tests for required `usage_metadata` and optional `token_usage|usage`, exact input/prompt and output/completion alias agreement, non-bool non-negative integers, exact total equality, and rejection of missing/type/negative/conflicting simultaneous aliases.
- [ ] Add RED strict embedding usage tests for required prompt/total equality, optional equal input alias, zero-only output aliases, and rejection of missing/string/float/bool/negative/conflicting/nonzero output metadata.
- [ ] Add RED boundary tests distinguishing a response-less transport exception from a returned-response usage-contract violation; do not fabricate a missing usage object when no response exists.
- [ ] Add RED cost goldens for `cl100k_base` query embedding with no normalization/chat overhead and exact 7,999/8,000 pass versus 8,001 pre-claim refusal, decomposed/repeated-whitespace byte preservation, `o200k_base + 16 + 512` answer estimation, exact direct-standard model/config snapshots, six-place `ROUND_CEILING`, full reserve, actual replacement, USD `0.012000` component ceiling, and unclamped representable overrun.
- [ ] Add RED tests that one config/price/tokenizer/estimator/endpoint/service-tier mutation changes policy HMAC and produces a zero-call readiness refusal.
- [ ] Run `uv run pytest backend/tests/test_provider_usage.py backend/tests/test_rag_v2_cost_policy.py -q` and confirm RED.
- [ ] Implement shared strict parsers and frozen Decimal policy. Refactor the C.5 validator to call `StrictChatUsageParser` while preserving its existing accepted/rejected behavior and cost outputs.
- [ ] Rerun `uv run pytest backend/tests/test_provider_usage.py backend/tests/test_rag_v2_cost_policy.py backend/tests/test_auto_review_validator.py backend/tests/test_auto_review_model_router.py -q`.
- [ ] Run `uv run ruff check backend/app/agent_runtime/provider_usage.py backend/app/agent_runtime/rag_cost_policy.py backend/app/agent_runtime/auto_review_validator.py backend/app/agent_runtime/model_router.py backend/tests/test_provider_usage.py backend/tests/test_rag_v2_cost_policy.py`.
- [ ] Commit with `git commit -m "feat: freeze rag provider usage and cost policy"`.

### Task 10: Build the Strict LangChain Structured Answer Boundary

**Files:**

- Create: `backend/app/agents/rag_orchestrator_agent/v2_answer_schema.py`
- Create: `backend/app/agents/rag_orchestrator_agent/v2_answer.py`
- Create: `backend/tests/test_rag_v2_answer_schema.py`
- Create: `backend/tests/test_rag_v2_answer_model.py`
- Create: `backend/tests/test_rag_answer_model_router.py`
- Modify: `backend/app/agent_runtime/model_router.py`
- Modify: `backend/app/agents/rag_orchestrator_agent/llm.py`
- Modify: `backend/tests/test_auto_review_model_router.py`
- Modify: `backend/tests/test_rag_orchestrator_llm.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class PreparedAnswerInvocation:
    messages: tuple[tuple[Literal['system', 'user'], str], ...]
    evidence_slots: tuple[EvidenceSlot, ...]
    model_influence: tuple[PreparedModelInfluenceObservation, ...]
    answer_question_hmac: str
    retrieval_query_hmac: str
    rendered_input_hmac: str
    generation_estimator_input_hmac: str
    encoded_input_tokens: int
    framed_input_tokens: int
    reserved_cost_usd: Decimal
    model_config_snapshot_hmac: str
    provider_policy_snapshot_hmac: str
    budget: PreparedPaidCallBudget

AnswerBlockUncertaintyReason: TypeAlias = Literal[
    'source_observation_not_promoted_to_trusted_knowledge',
]

@dataclass(frozen=True, slots=True)
class ValidatedAnswerBlock:
    block_ordinal: int
    text: str
    evidence_slot_ids: tuple[EvidenceSlotId, ...]
    support_mode: SupportMode
    confidence_score: Decimal
    uncertainty_reason: AnswerBlockUncertaintyReason | None
    block_result_hmac: str

@dataclass(frozen=True, slots=True)
class ValidatedAnswerBlocks:
    blocks: tuple[ValidatedAnswerBlock, ...]
    insufficient_reason: str | None
    selected_slot_ids: tuple[EvidenceSlotId, ...]
    assembled_answer: str
    assembled_answer_hmac: str | None
    answer_block_audit_set_hmac: str | None

@dataclass(frozen=True, slots=True)
class ProviderAnswerEnvelope:
    raw_message: object
    returned_model: str
    returned_object: str
    returned_service_tier: str
    usage: StrictProviderUsage
    latency_ms: int

@dataclass(frozen=True, slots=True)
class RoutedRagAnswerModel:
    model: Runnable[Sequence[BaseMessage], AIMessage]
    provider: Literal['openai']
    model_name: Literal['gpt-5.4-mini-2026-03-17']
    model_config_snapshot_hmac: str

def build_rag_answer_model_route(*, settings: Settings) -> RoutedRagAnswerModel: ...

class StructuredRagAnswerModel:
    def __init__(self, *, routed_model: RoutedRagAnswerModel, cost_policy: RagCostPolicy) -> None: ...
    def prepare(self, *, question: str, slots: tuple[EvidenceSlot, ...], ...) -> PreparedAnswerInvocation: ...
    def invoke_once(self, prepared: PreparedAnswerInvocation, permit: ProviderDispatchPermit) -> ProviderAnswerEnvelope: ...
    def validate(self, envelope: ProviderAnswerEnvelope, prepared: PreparedAnswerInvocation) -> ValidatedAnswerBlocks: ...
```

- [ ] Add RED golden tests for the hand-authored `rag_answer_blocks_v1` provider JSON schema and exact insertion-order wrapper bytes. Assert LangChain receives exact `response_format={'type':'json_schema','json_schema':<literal>}` through `with_structured_output(..., method='json_schema', strict=True, include_raw=True)`.
- [ ] Add RED startup-readiness tests rejecting generated Pydantic title/description, missing/extra/reordered keys, changed required/strict/additionalProperties/null unions/minItems/maxItems/enums, or renderer/joiner/schema HMAC drift before a provider call.
- [ ] Add RED renderer goldens for exact two messages, compact `QUESTION_JSON`/`EVIDENCE_JSON`, untrusted-data instruction, evidence order, escaping, static bytes, LF/joiner literals, and credential scans over question/frame/whole evidence/final messages.
- [ ] Add RED estimator tests using compact sorted JSON, exact schema object, `o200k_base`, no Unicode normalization, `allowed_special=set()`, `disallowed_special=()`, `+16+512`, max 12,000 serialized chars, max 10,000 framed input tokens, and whole-evidence tail drop with slot renumbering. Static-frame-only overflow must be startup failure.
- [ ] Add RED semantic validation matrix for 0/1/8/9 blocks; per-block text 1/1,200/1,201; total text 2,400/2,401; reason 1/400/401; blocks/reason XOR; NUL/surrogate/whitespace; no coercion/extra fields; E9/unselected slots; per-block duplicate/cross-block first-use; and support-mode trust mismatch.
- [ ] Add RED confidence tests deriving `0.950000` with null uncertainty for all-trusted blocks and `0.700000` with the exact low-confidence reason when any raw slot supports a block. Reject model-supplied confidence or permission.
- [ ] Add RED usage and response-identity tests for exact returned model/object/service tier, strict chat usage aliases/equality, output token limit, no retry/fallback, response-less transport distinction, and output/schema/citation failure preserving valid actual usage.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_answer_schema.py backend/tests/test_rag_v2_answer_model.py backend/tests/test_rag_answer_model_router.py backend/tests/test_rag_orchestrator_llm.py -q` and observe RED.
- [ ] Implement the exact OpenAI Responses configuration only in the model-router boundary: direct `https://api.openai.com/v1`, model `gpt-5.4-mini-2026-03-17`, `reasoning_effort='none'`, `service_tier='default'`, max 512 output tokens, 30-second timeout, `max_retries=0`, omitted temperature/top-p/seed, `store=False`, no tools/streaming/cache/callbacks/tracing/fallback. `v2_answer.py` consumes the injected routed model and must not construct an SDK/provider client.
- [ ] Keep `rag-answer:v1` legacy model behavior unchanged. V2 must fail closed on missing credentials/readiness; it must never fall back to `DeterministicRagOrchestratorModel` in enforce mode.
- [ ] Rerun focused tests and `uv run ruff check backend/app/agent_runtime/model_router.py backend/app/agents/rag_orchestrator_agent/v2_answer_schema.py backend/app/agents/rag_orchestrator_agent/v2_answer.py backend/app/agents/rag_orchestrator_agent/llm.py backend/tests/test_rag_v2_answer_schema.py backend/tests/test_rag_v2_answer_model.py backend/tests/test_rag_answer_model_router.py`.
- [ ] Commit with `git commit -m "feat: add strict langchain rag answer model"`.

### Task 11: Add Runtime Safety, Cost, and Assistant Integrity Schema

**Files:**

- Create: `backend/app/models/rag_runtime.py`
- Create: `backend/migrations/versions/e2b3c4d5f6a7_add_rag_runtime_safety.py`
- Create: `backend/tests/test_rag_v2_runtime_models.py`
- Create: `backend/tests/test_rag_v2_runtime_migration.py`
- Modify: `backend/app/models/agent_runs.py`
- Modify: `backend/app/models/assistant.py`
- Modify: `backend/app/models/auto_review.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/tests/test_agent_runtime_migration.py`

**Interfaces:**

```python
class AgentRunCostComponent(Base):
    __tablename__ = 'agent_run_cost_components'
    # unique (agent_run_id, component); exact query_embedding + answer_generation

class RagProviderSafetyAuthority(Base):
    __tablename__ = 'rag_provider_safety_authorities'

class RagProviderReadiness(Base):
    __tablename__ = 'rag_provider_readiness'

class RagProviderSafetyTransition(Base):
    __tablename__ = 'rag_provider_safety_transitions'

class RagAdvisoryLockKey(Base):
    __tablename__ = 'rag_advisory_lock_key_registry'
```

- [ ] Add RED migration tests for `revision='e2b3c4d5f6a7'`, `down_revision='d1a2b3c4e5f6'`, one Alembic head, cost/provider/advisory tables, AgentRun fields, Assistant content/result fields, and dependency scope/role/HMAC/provenance fields.
- [ ] Add RED future-parent tests for `run_contract_version='rag-run:v2'`, phases `admission | cost_finalized_pending_projection | final | admission_only`, non-null `NUMERIC(24,6)` total, optional projection-owner fence, and historical null rows with no backfill.
- [ ] Add RED component tests for exact order/names, terminal-zero versus `not_attempted`, dispatching reserve, terminal actual/reserved, abandoned unknown, overrun, immutable terminal rows, BIGINT token range, NUMERIC storage, one dispatch maximum, and target provider/model/config/cost/estimator identities.
- [ ] Add RED PostgreSQL deferred-trigger tests for exact two children, unique component, child sum/parent total equality, phase/status/completed-at/outcome matrix, final-error/admission-only constraints, and all-or-nothing parent/children transitions.
- [ ] Add RED generation-guard migration tests for deferred commit-time triggers on every canonical raw/trusted eligibility/content/version/permission mutation and D vector write/tombstone. Direct SQL and a deliberately generation-unsafe legacy writer must fail at commit; Task 5's updated writers must pass while incrementing corpus or vector generation in the same transaction. Lexical projection refresh is bound to corpus generation and never increments vector generation.
- [ ] Add RED provider/advisory tests for singleton/exact-two active family rows, state/version/generation/checksum constraints, append-only transitions, signed int4 key pairs, static/dynamic namespaces, two-argument advisory lock use, collision fail-stop, and no salt retry.
- [ ] Add RED Assistant tests for exact-write/legacy-trim modes, assembled/canned XOR, result/content HMACs, dependency marker/schema, child scope/role, and historical null-marker readability.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_runtime_models.py backend/tests/test_rag_v2_runtime_migration.py backend/tests/test_agent_runtime_migration.py -q` and confirm RED.
- [ ] Implement the second additive migration, runtime models, and activation of generation-guard triggers after Task 5 has made every existing writer compliant. Do not alter first-revision serving table/function ownership and do not add any live-release tables yet.
- [ ] Preserve Float/token compatibility mirrors as non-authoritative; never infer new fixed-precision values from historical rows.
- [ ] Rerun focused SQLite/PostgreSQL migration suites and `uv run ruff check backend/app/models/rag_runtime.py backend/app/models/agent_runs.py backend/app/models/assistant.py backend/app/models/auto_review.py backend/migrations/versions/e2b3c4d5f6a7_add_rag_runtime_safety.py backend/tests/test_rag_v2_runtime_models.py backend/tests/test_rag_v2_runtime_migration.py`.
- [ ] Commit with `git commit -m "feat: add rag runtime safety schema"`.

### Task 12: Enforce Durable Cost Claims and Provider Safety

**Files:**

- Create: `backend/app/agent_runtime/rag_runtime_contracts.py`
- Create: `backend/app/agent_runtime/rag_safety_identity.py`
- Create: `backend/app/agent_runtime/rag_advisory_locks.py`
- Create: `backend/app/agent_runtime/durable_file_authority.py`
- Create: `backend/app/agent_runtime/rag_cost_ledger.py`
- Create: `backend/app/agent_runtime/rag_provider_safety.py`
- Create: `backend/app/agent_runtime/rag_provider_transport.py`
- Create: `backend/tests/test_rag_advisory_locks.py`
- Create: `backend/tests/test_rag_advisory_locks_postgres.py`
- Create: `backend/tests/test_durable_file_authority.py`
- Create: `backend/tests/test_rag_v2_costs.py`
- Create: `backend/tests/test_rag_v2_provider_safety.py`
- Create: `backend/tests/test_rag_v2_provider_transport.py`
- Modify: `backend/app/agent_runtime/provider_send_fence.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/tests/test_provider_send_fence.py`

**Interfaces:**

```python
RagComponentClassification = Literal[
    'validated_success', 'pre_send_refusal', 'response_less_failure',
    'response_identity_invalid', 'embedding_payload_invalid',
    'usage_contract_invalid', 'usage_storage_invalid', 'known_overrun',
    'structured_output_invalid', 'citation_validation_failed',
    'evidence_validation_failed',
]

RagComponentTerminalOutcome = Literal[
    'component_succeeded', 'retriever_unavailable', 'model_provider_failed',
    'provider_response_identity_invalid', 'provider_embedding_payload_invalid',
    'provider_usage_overrun', 'provider_safety_unavailable',
    'structured_output_invalid', 'citation_validation_failed',
    'evidence_unavailable', 'abandoned_unknown',
]

RagProviderSafetyAction = Literal['unchanged', 'block_overrun', 'block_remediation']

RagRunTerminalOutcome = RagResultOutcome | Literal[
    'serving_index_not_ready', 'shadow_match', 'shadow_mismatch',
    'persistence_failed', 'abandoned_unknown', 'live_corpus_snapshot_changed',
]

RagAdmissionSourceWindow: TypeAlias = Literal[
    'rag-v2:admission:enforce:ask:keyword',
    'rag-v2:admission:enforce:ask:pgvector',
    'rag-v2:admission:enforce:search:keyword',
    'rag-v2:admission:enforce:search:pgvector',
    'rag-v2:admission:enforce:assistant:keyword',
    'rag-v2:admission:enforce:assistant:pgvector',
    'rag-v2:admission:shadow:ask:pgvector',
    'rag-v2:admission:shadow:search:pgvector',
    'rag-v2:admission:shadow:assistant:pgvector',
]

RagTerminalSourceWindow: TypeAlias = Literal[
    'rag-v2:ask:keyword', 'rag-v2:ask:pgvector',
    'rag-v2:search:keyword', 'rag-v2:search:pgvector',
    'rag-v2:assistant:keyword', 'rag-v2:assistant:pgvector',
    'rag-v2:shadow:ask:keyword', 'rag-v2:shadow:ask:pgvector',
    'rag-v2:shadow:search:keyword', 'rag-v2:shadow:search:pgvector',
    'rag-v2:shadow:assistant:keyword', 'rag-v2:shadow:assistant:pgvector',
    'rag-v2:final-error:ask:keyword', 'rag-v2:final-error:ask:pgvector',
    'rag-v2:final-error:search:keyword', 'rag-v2:final-error:search:pgvector',
    'rag-v2:final-error:assistant:keyword', 'rag-v2:final-error:assistant:pgvector',
]

@dataclass(frozen=True, slots=True)
class RagRunAdmission:
    agent_run_id: int
    surface: RagSurface
    mode: Literal['shadow', 'enforce']
    cutover_stage: RagCutoverStage
    configured_backend: RagRetrievalBackend
    query_context_version: Literal['direct-query:v1', 'assistant-context:v1']
    current_text_hmac: str
    retrieval_query_hmac: str
    security_scope_fingerprint: str
    query_embedding_provider_policy_snapshot_hmac: str | None
    answer_provider_policy_snapshot_hmac: str | None
    admission_cache_identity_hmac: str
    source_window: RagAdmissionSourceWindow
    status: Literal['running']
    run_record_phase: Literal['admission']
    component_order: tuple[Literal['query_embedding'], Literal['answer_generation']]
    total_reserved_cost_usd: Decimal
    total_charged_cost_usd: Decimal
    runtime_cost_snapshot_hmac: str

@dataclass(frozen=True, slots=True)
class StrictProviderOutcome:
    component: RagPaidComponent
    classification: RagComponentClassification
    terminal_outcome: RagComponentTerminalOutcome
    provider_dispatch_started: bool
    provider_response_received: bool
    strict_usage: StrictProviderUsage | None
    actual_cost_usd: Decimal | None
    safety_action: RagProviderSafetyAction

@dataclass(frozen=True, slots=True)
class RagComponentFinal:
    agent_run_id: int
    component: RagPaidComponent
    terminal_outcome: RagComponentTerminalOutcome | None
    dispatch_state: Literal['terminal', 'abandoned_unknown']
    attempted: bool
    dispatch_count: Literal[0, 1]
    reserved_input_tokens: int
    reserved_output_tokens: int
    actual_input_tokens: int | None
    actual_output_tokens: int | None
    reserved_cost_usd: Decimal
    charged_cost_usd: Decimal
    charge_basis: Literal['zero', 'actual', 'reserved']
    overrun: bool
    provider: str
    model: str
    authorized_model_config_version: str
    authorized_model_config_snapshot_hmac: str
    authorized_cost_policy_version: str
    authorized_token_estimator_version: str
    authorized_policy_snapshot_hmac: str
    dispatch_fence_hmac: str | None
    process_instance_hmac: str | None
    parent_status: Literal['running', 'complete', 'failed']
    parent_run_record_phase: Literal['admission', 'cost_finalized_pending_projection', 'final', 'admission_only']
    parent_total_charged_cost_usd: Decimal
    projection_owner_fence_hmac: str | None
    runtime_cost_snapshot_hmac: str

@dataclass(frozen=True, slots=True)
class RagRunTerminal:
    agent_run_id: int
    status: Literal['complete', 'failed']
    run_record_phase: Literal['final', 'admission_only']
    outcome: RagRunTerminalOutcome
    admission_cache_identity_hmac: str
    source_window: RagTerminalSourceWindow | RagAdmissionSourceWindow
    cache_key: str
    total_reserved_cost_usd: Decimal
    total_charged_cost_usd: Decimal
    component_finals: tuple[RagComponentFinal, RagComponentFinal]
    projection_owner_fence_hmac: str | None
    runtime_cost_snapshot_hmac: str
    terminal_identity_hmac: str | None
    completed_at: datetime

@dataclass(frozen=True, slots=True)
class AuthorizedProviderPolicySnapshot:
    component: RagPaidComponent
    provider: str
    model: str
    reasoning_or_config_identity: str
    authorized_model_config_version: str
    authorized_model_config_snapshot_hmac: str
    authorized_cost_policy_version: str
    authorized_token_estimator_version: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    authorized_policy_snapshot_hmac: str

@dataclass(frozen=True, slots=True)
class RagProviderSafetyBinding:
    policy_snapshot: AuthorizedProviderPolicySnapshot
    authority_uuid: UUID
    designated_environment_id: str
    envelope_digest: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    global_safety_generation: int
    family_state: Literal['ready']
    family_state_version: int
    family_safety_generation: int
    provider_safety_snapshot_hmac: str

class CommittedRagDispatchGrant(ProviderDispatchPermit, Protocol):
    @property
    def agent_run_id(self) -> int: ...
    @property
    def component(self) -> RagPaidComponent: ...
    @property
    def reserved_cost_usd(self) -> Decimal: ...
    @property
    def dispatch_fence_hmac(self) -> str: ...
    @property
    def provider_safety_snapshot_hmac(self) -> str: ...
    def consume_at_dispatch(self) -> None: ...

class RagCostLedger:
    def create_admission(self, ...) -> RagRunAdmission: ...
    def claim_component(self, *, run_id: int, component: RagPaidComponent, prepared: PreparedPaidCallBudget) -> CommittedRagDispatchGrant: ...
    def finalize_component(self, *, grant: CommittedRagDispatchGrant, outcome: StrictProviderOutcome) -> RagComponentFinal: ...
    def finalize_projectionless_failure(self, ...) -> RagRunTerminal: ...

class RagProviderSafetyService:
    def require_ready(self, connection: Connection, component: RagPaidComponent, policy_snapshot: AuthorizedProviderPolicySnapshot) -> RagProviderSafetyBinding: ...
    def revalidate_before_send(self, ...) -> RagProviderSafetyBinding: ...
    def revalidate_after_response(self, ...) -> RagProviderSafetyBinding: ...
    def block_overrun(self, ...) -> None: ...
    def block_remediation(self, ...) -> None: ...
```

- [ ] Add RED ledger transition tests that apply the frozen Task 9 policy to full reserve, actual replacement, terminal zero, response-less reserved failure, and unclamped representable overrun without recomputing prices in the store. Only exact terminal-zero (`attempted=false`, dispatch count zero, zero charge) has `terminal_outcome=None`; every attempted/dispatching terminal component requires an allowlisted non-null outcome.
- [ ] Add RED identity goldens for model-config/provider-policy snapshots, security scope, admission, runtime cost, final product/shadow/error, projection owner, process/dispatch fences, and implementation-plan reference. Mutating one key/type/order/byte/version must change or reject the identity. Require the exact nine admission and eighteen terminal source-window literals. `run_record_phase='admission_only'` preserves an admission literal, exact `cache_key='rag-v2-admission:' + admission_cache_identity_hmac`, and `terminal_identity_hmac=None`; `final` requires a product/shadow/final-error literal, its matching cache-key prefix, and the non-null exact corresponding final identity HMAC. Reject legacy `ask:<text>`, unknown/dynamic values, phase/window/identity mismatch, a new abandoned terminal domain, and any persisted raw-query substring.
- [ ] Add RED claim tests proving `status=running` parent plus exact-two children commits before permit creation; the selected child becomes `dispatching` with full reserve and dispatch count one; no DB commit means no permit/provider call; a consumed or copied permit cannot dispatch twice.
- [ ] Add RED parser/accounting matrix for response-less transport reserve, strict valid actual, storage-invalid reserve/remediation, returned identity invalid, vector invalid, usage invalid, known overrun, structured/citation/evidence failure with valid usage, sibling terminal-zero, and parent total after every transition.
- [ ] Add RED provider-safety tests for exact two families, external-first blocker, DB/external generation equality, family rebind versus supersession, restart persistence, rollback not clearing blockers, and other-worker zero-call after block.
- [ ] Add RED lock-order assertions for both paths. Ordinary runtime is `provider stable sidecar -> provider-safety singleton/families -> projection owner -> evidence barrier -> C.5 key/corpus -> AgentRun/cost -> optional Assistant`; live release is `provider stable sidecar -> release stable sidecar/advisory -> provider-safety singleton/families -> release ledger/authorization/case/dispatch rows -> projection owner -> evidence barrier -> C.5 key/corpus -> AgentRun/cost -> optional Assistant`.
- [ ] Add RED stable-sidecar tests on Windows/POSIX abstraction: never-replaced sidecar, atomic data-file replace, process crash OS unlock, fresh HMAC/generation read, path/ACL/regular-file validation, symlink/reparse/hardlink/case-fold alias rejection, and no unsupported-lock fallback.
- [ ] Add RED pre-send evidence fence interleavings for writer-before-lock, writer waiting, evidence change between recheck and send, exact shared advisory unlock true, cancellation/exception cleanup, invalid unlock connection close, and no pooled lock inheritance.
- [ ] Run `uv run pytest backend/tests/test_rag_advisory_locks.py backend/tests/test_durable_file_authority.py backend/tests/test_rag_v2_costs.py backend/tests/test_rag_v2_provider_safety.py backend/tests/test_rag_v2_provider_transport.py backend/tests/test_provider_send_fence.py -q` and confirm RED.
- [ ] Implement provider-safety path settings using `PARAWORKS_PROVIDER_SAFETY_LATCH_PATH` plus its stable `.lock` file. Ordinary disabled/non-cutover startup validates only safe configuration strings and must not create authority artifacts.
- [ ] Build fenced OpenAI transports that receive a store-owned one-use permit; provider body/response bytes never enter the cost store or logs. Disable retries/callbacks/cache/tracing at both LangChain and transport layers.
- [ ] Implement the two exact lock orders above. In particular, live release never locks release DB rows before provider-safety rows; only the release sidecar/advisory precedes provider safety. Add executable assertions rather than relying on comments.
- [ ] Rerun focused suites and `uv run ruff check backend/app/agent_runtime/rag_runtime_contracts.py backend/app/agent_runtime/rag_safety_identity.py backend/app/agent_runtime/rag_advisory_locks.py backend/app/agent_runtime/durable_file_authority.py backend/app/agent_runtime/rag_cost_ledger.py backend/app/agent_runtime/rag_provider_safety.py backend/app/agent_runtime/rag_provider_transport.py backend/app/agent_runtime/provider_send_fence.py backend/tests/test_rag_v2_costs.py backend/tests/test_rag_v2_provider_safety.py backend/tests/test_rag_v2_provider_transport.py`.
- [ ] Commit with `git commit -m "feat: enforce rag provider cost authority"`.

### Task 13: Finalize Fresh Projections and SQLite Smoke Atomically

**Files:**

- Create: `backend/app/agent_runtime/rag_finalization.py`
- Create: `backend/app/agent_runtime/rag_sqlite_smoke.py`
- Create: `backend/app/assistant/evidence_persistence.py`
- Create: `backend/tests/test_rag_v2_finalization.py`
- Create: `backend/tests/test_rag_v2_sqlite_smoke.py`
- Create: `backend/tests/test_assistant_evidence_writer_v2.py`
- Modify: `backend/app/rag/serving_locks.py`
- Modify: `backend/app/assistant/service.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class RagProjectionPending:
    parent_agent_run_id: int
    projection_owner_fence_hmac: str
    security_scope_fingerprint: str
    prepared_corpus_generation: int
    prepared_vector_index_generation: int | None
    terminal_cost_snapshot_hmac: str

@dataclass(frozen=True, slots=True)
class CanonicalRagProjection:
    outcome: RagResultOutcome
    answer_text: str | None
    evidence: V1EvidenceProjection
    model_influence: tuple[ModelInfluenceDependencySnapshot, ...]
    hidden_match_count: int
    effective_backend: RagEffectiveBackend
    result_hmac: str

@dataclass(frozen=True, slots=True)
class AssistantProjectionTarget:
    conversation_id: int
    user_message_id: int
    owner_user_id: str

@dataclass(frozen=True, slots=True)
class AssistantFinalizationRecord:
    assistant_message_id: int
    parent_agent_run_id: int
    application_outcome: AssistantPersistedOutcome
    finalization_kind: Literal['substantive', 'canned_safe', 'terminal_failure']

RagFinalProjection = CanonicalRagProjection | AssistantFinalizationRecord

@dataclass(frozen=True, slots=True)
class PreparedRagFinalization:
    product_kind: Literal['answer', 'search']
    tentative_outcome: RagResultOutcome
    prepared_text: PreparedRagRequestText
    security_scope: SecurityScope
    query_embedding_result: QueryEmbeddingCallResult | None
    retrieval_result: RetrievalResult
    evidence_slots: tuple[EvidenceSlot, ...]
    model_influence_observations: tuple[PreparedModelInfluenceObservation, ...]
    selected_slot_ids: tuple[EvidenceSlotId, ...]
    validated_answer: ValidatedAnswerBlocks | None
    canned_message_identity: RagCannedMessageIdentity | None

class RagFinalizationService:
    def finalize_direct(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
    ) -> CanonicalRagProjection: ...
    def finalize_assistant(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        target: AssistantProjectionTarget,
    ) -> AssistantFinalizationRecord: ...
    def finalize_provider_free_safe(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection: ...
    def finalize_paid_embedding_only_safe(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection: ...
    def finalize_pre_generation_evidence_changed(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection: ...
    def finalize_inter_component_failure(self, ...) -> RagRunTerminal: ...
    def recover_dead_projection_owner(self, run_id: int) -> RagRunTerminal | None: ...

class SQLiteRagSmokeCoordinator:
    def run_keyword(
        self,
        request: PreparedRagRequestText,
        *,
        scope: SecurityScope,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection: ...

@dataclass(frozen=True, slots=True)
class AssistantMessageProjection:
    content: str
    metadata: Mapping[str, object]
    evidence: V1EvidenceProjection
    permission_level: str | None
    permission_notice: str | None
    hidden_match_count: int
    assembled_answer_hmac: str | None
    canned_message_identity: RagCannedMessageIdentity | None
    result_hmac: str
    write_mode: Literal['rag_v2_exact']

@dataclass(frozen=True, slots=True)
class LegacyAssistantMessageProjection:
    content: str
    metadata: Mapping[str, object]
    evidence: V1EvidenceProjection
    write_mode: Literal['legacy_trimmed']

class AssistantEvidenceWriter:
    def append_final(
        self,
        *,
        db: Session,
        conversation: AssistantConversation,
        projection: AssistantMessageProjection,
        pending: RagProjectionPending,
    ) -> AssistantMessage: ...

    def append_legacy_evidence(
        self,
        *,
        db: Session,
        conversation: AssistantConversation,
        projection: LegacyAssistantMessageProjection,
    ) -> AssistantMessage: ...
```

- [ ] Add RED normal two-phase tests: component costs become immutable terminal and parent pending first with product bytes/message absent; fresh corpus transaction then atomically writes parent final plus immutable DTO or Assistant message/dependencies.
- [ ] Add RED zero-provider `/search` and pre-generation safe-answer tests for parent + exact-two terminal-zero, no provider authority/artifacts, projection-owner fence, fresh hidden/corpus projection, safe 200 despite blocked provider families, and no pending residue.
- [ ] Add RED paid-embedding-only and pre-generation-evidence-changed tests preserving exact terminal embedding actual/fence/vector observation, terminal-zero answer child, no generation call, and correct canned projection values.
- [ ] Add RED carrier/final revalidation tests proving `PreparedRagFinalization` contains the exact transient query, `SecurityScope`, optional validated query vector, original retrieval result, answer/search projection inputs, complete pre-generation influence observations, and post-validation selected IDs. Rebuild `RetrievalRequest` without provider dispatch, fresh-resolve every canonical lookup identity/version, recompute model-influence full set, selected subset, hidden membership, security scope, and corpus/index generation, and map successful-open-transaction drift to `evidence_unavailable`; lock/read/connection/commit failure must create no DTO/message.
- [ ] Add RED lifecycle tests proving `PreparedModelInfluenceObservation` exists before generation without a role, the finalizer alone derives each dependency role from validated selected IDs after fresh resolution, and neither request/query plaintext, `SecurityScope.principal_subject`, query vector, preliminary retrieval bytes, nor prepared observations are persisted or logged.
- [ ] Add RED Assistant-writer tests that `rag_v2_exact` validates nonblank but preserves exact leading/trailing UTF-8 bytes, legacy mode keeps current `.strip()`, assembled/canned identity is XOR, parent result/content HMAC and full dependency set commit in the same transaction, and no caller can request V2 exact mode directly.
- [ ] Add RED target tests that `AssistantProjectionTarget.owner_user_id` is the existing string owner identifier. The finalizer fresh-loads and locks the conversation and user message, verifies exact owner/conversation/message relations, and fails without a write on mismatch; no synthetic conversation-version field is introduced.
- [ ] Add RED projection-owner recovery tests: no mutation while owner session lock is live; dead-session reacquire plus exact fence/CAS can close `persistence_failed`; timeout/process probing alone cannot; provider output is never retried.
- [ ] Add RED SQLite tests for one process-local reentrant mutex, `BEGIN IMMEDIATE`, parent/exact-two/final product in one transaction, writer-before/after serialization, drift canned answer, transaction crash rollback, and zero externally committed pending rows.
- [ ] Add RED file-backed SQLite process-lifetime OS lock tests for one never-replaced lock identity and second-process/symlink/hardlink/path mismatch refusal. In-memory SQLite is test-only; pgvector/live/paid/release modes must refuse before mutation/call.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_finalization.py backend/tests/test_rag_v2_sqlite_smoke.py -q` and confirm RED.
- [ ] Implement all finalizers over the shared `CanonicalEvidenceProjector` and injected retriever/readiness boundaries. `PreparedRagFinalization` is immutable, request-local, graph-neutral, and non-persistent; its preliminary retrieval/answer bytes are comparison inputs, never public authority. Both direct and Assistant finalizers rerun retrieval with the already validated optional vector and zero provider dispatch, fresh-resolve identities, then build and commit the canonical projection. Pass only the committed immutable projection upward and prohibit route-side reconstruction.
- [ ] Keep Assistant V2 message creation behind the finalizer. `AssistantEvidenceWriter` may validate/add/flush rows but never commit; `RagFinalizationService` alone owns the product transaction and commit. Existing helper functions remain available only to legacy/non-RAG flows until Task 18 moves the route.
- [ ] Rerun focused suites and `uv run ruff check backend/app/agent_runtime/rag_finalization.py backend/app/agent_runtime/rag_sqlite_smoke.py backend/app/assistant/evidence_persistence.py backend/app/rag/serving_locks.py backend/tests/test_rag_v2_finalization.py backend/tests/test_rag_v2_sqlite_smoke.py backend/tests/test_assistant_evidence_writer_v2.py`.
- [ ] Commit with `git commit -m "feat: finalize rag projections atomically"`.

### Task 14: Compile the Actual LangGraph and Application Facade

**Files:**

- Create: `backend/app/agent_runtime/rag_graph.py`
- Create: `backend/app/agent_runtime/rag_v2_state.py`
- Create: `backend/app/agent_runtime/rag_v2_composition.py`
- Create: `backend/app/agent_runtime/rag_application.py`
- Create: `backend/app/agent_runtime/rag_rollout.py`
- Create: `backend/tests/test_rag_v2_graph.py`
- Create: `backend/tests/test_rag_application_facade.py`
- Create: `backend/tests/test_rag_rollout.py`
- Modify: `backend/app/agent_runtime/rag_v2_registry.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`

**Interfaces:**

```python
RagFallbackCategory = Literal[
    'pgvector_storage_runtime_failure',
    'serving_corpus_changed_during_pgvector_query',
    'serving_corpus_changed_during_answer_revalidation',
    'pgvector_hidden_recompute_runtime_failure',
]

@dataclass(frozen=True, slots=True)
class RagRunTrace:
    node_counts: Mapping[str, int]
    node_latency_ms: Mapping[str, int]
    provider_attempt_counts: Mapping[RagPaidComponent, int]
    bounded_candidate_counts: Mapping[str, int]
    domain_hmacs: Mapping[str, str]

@dataclass(frozen=True, slots=True)
class SanitizedRagToolEvent:
    event_kind: str
    outcome: str
    latency_ms: int
    domain_hmac: str

class RagGraphInput(TypedDict):
    prepared_text: PreparedRagRequestText

class RagGraphOutput(TypedDict):
    outcome: RagResultOutcome
    answer_blocks: ValidatedAnswerBlocks | None
    selected_slot_ids: tuple[EvidenceSlotId, ...]
    evidence_projection: V1EvidenceProjection
    model_influence: tuple[ModelInfluenceDependencySnapshot, ...]
    hidden_match_count: int
    effective_backend: RagEffectiveBackend
    fallback_category: RagFallbackCategory | None
    charged_cost_usd: Decimal
    sanitized_trace: RagRunTrace

class RagGraphState(TypedDict, total=False):
    # request-local text/HMAC/scope/backend/candidates/slots/blocks/cost/projection fields only

class SafeRagRunRecorder(Protocol):
    def record(self, trace: RagRunTrace) -> None: ...

class SafeRagToolRecorder(Protocol):
    def record(self, event: SanitizedRagToolEvent) -> None: ...

@dataclass(frozen=True, slots=True)
class RagRouteDecision:
    public_owner: Literal['legacy', 'v2']
    shadow_retrieval: bool
    cutover_surface: bool

class RagRolloutPolicy:
    def decide(self, *, mode: RagMode, stage: RagCutoverStage, surface: RagSurface) -> RagRouteDecision: ...

class RagApplicationFacade:
    def execution_owner(self, surface: RagSurface) -> Literal['legacy', 'shadow', 'v2']: ...
    def invoke_graph(
        self,
        *,
        actor: DemoUser,
        surface: RagSurface,
        prepared_text: PreparedRagRequestText,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagGraphOutput: ...

@dataclass(frozen=True, slots=True)
class RagRuntimeContext:
    session_factory: Callable[[], Session]
    actor: DemoUser
    surface: RagSurface
    security_scope_resolver: RagSecurityScopeResolver
    retrievers: RagRetrieverRegistry
    query_embedding_adapter: StrictQueryEmbeddingAdapter
    index_readiness: RagV2ServingIndexReadinessService
    evidence_resolver: ServingEvidenceResolver
    answer_model: StructuredRagAnswerModel
    cost_policy: RagCostPolicy
    cost_ledger: RagCostLedger
    provider_safety: RagProviderSafetyService
    finalizer: RagFinalizationService
    rollout_policy: RagRolloutPolicy
    run_recorder: SafeRagRunRecorder
    tool_recorder: SafeRagToolRecorder
    settings: Settings

@dataclass(frozen=True, slots=True)
class RagRuntimeDependencies:
    security_scope_resolver: RagSecurityScopeResolver
    retrievers: RagRetrieverRegistry
    query_embedding_adapter: StrictQueryEmbeddingAdapter
    index_readiness: RagV2ServingIndexReadinessService
    evidence_resolver: ServingEvidenceResolver
    answer_model: StructuredRagAnswerModel
    cost_policy: RagCostPolicy
    cost_ledger: RagCostLedger
    provider_safety: RagProviderSafetyService
    finalizer: RagFinalizationService
    rollout_policy: RagRolloutPolicy
    run_recorder: SafeRagRunRecorder
    tool_recorder: SafeRagToolRecorder

@dataclass(frozen=True, slots=True)
class RagRuntimeBundle:
    compiled_graph: CompiledStateGraph
    graph_registry: RagGraphRegistry
    manifest_registry: AgentRegistry
    facade: RagApplicationFacade

def build_company_memory_rag_answer_v2_graph() -> CompiledStateGraph: ...
def build_rag_v2_runtime(*, settings: Settings, session_factory: Callable[[], Session], dependencies: RagRuntimeDependencies) -> RagRuntimeBundle: ...
```

```text
START
  -> validate_input
  -> resolve_current_permission_context
  -> select_configured_backend
  -> preflight_retrieval_paid_cost_ceiling
       -> keyword_retrieval
       -> prepare_and_call_query_embedding -> pgvector_retrieval
            -> keyword_retrieval_on_pgvector_runtime_failure
  -> canonical_permission_guard
  -> rank_and_bound_evidence
  -> assemble_server_evidence_slots
  -> route_surface
       -> commit_zero_provider_search_costs_and_mark_projection_pending
            -> project_visible_search_results
            -> finalize_run_and_search_projection -> END
       -> commit_paid_or_prepared_cost_components_and_mark_projection_pending
            -> project_visible_search_results
            -> finalize_run_and_search_projection -> END
       -> provider_free_safe_outcome_finalizer -> END
       -> paid_embedding_only_safe_outcome_finalizer -> END
       -> prepare_and_preflight_answer_invocation
  -> generate_structured_answer_blocks
  -> validate_claim_evidence_refs
  -> commit_cost_components_and_mark_projection_pending
  -> revalidate_model_influence_and_selected_evidence
  -> recompute_bounded_hidden_count_and_selected_membership
  -> project_selected_server_citations
  -> finalize_run_and_answer_projection_or_assistant_message
  -> END
```

- [ ] Add RED topology tests that inspect an actual compiled `StateGraph` and require the exact spec §6 node names: `validate_input`, `resolve_current_permission_context`, `select_configured_backend`, `preflight_retrieval_paid_cost_ceiling`, `keyword_retrieval`, `prepare_and_call_query_embedding`, `pgvector_retrieval`, `keyword_retrieval_on_pgvector_runtime_failure`, `canonical_permission_guard`, `rank_and_bound_evidence`, `assemble_server_evidence_slots`, `route_surface`, both search pending/finalization branches, both safe-outcome finalizers, answer preparation/generation/validation/cost-pending/revalidation/hidden-recompute/citation-projection, and final answer/Assistant persistence. Require backend/surface/safe/failure conditional edges, START/END reachability, and no checkpointer/saver.
- [ ] Add RED tests proving compiled-graph reuse with two concurrent invocations does not share state/runtime actor/session/model/cost data and does not persist question/evidence/model output in graph checkpoints or metadata.
- [ ] Add RED execution tests with fake Runnable retrievers and fake structured model for supported, no-match, hidden-only, safety-filter-empty, model-insufficient, pre/post-generation evidence change, embedding failure, provider/schema/citation failure, and search projection.
- [ ] Add RED facade routing matrix for `disabled | shadow | enforce` and stage `none | ask | search | assistant`; non-cutover paths are exact legacy-only, enforce has no background shadow, and routes cannot select graph/backend/version. Add Task-14-owned exhaustive `RagRolloutPolicy` tests so the facade never depends on a later task.
- [ ] Add RED graph-preflight tests for the Assistant-only contextual retrieval maximum of 1,000 terms using the server-owned runtime `surface` plus `PreparedRagRequestText.query_context_version`; 1,001 terms must return budget refusal with zero DB/retriever/provider work, while direct `/ask` and `/search` remain unbounded by this rule.
- [ ] Add RED application-composition tests proving exact graph registration regardless of rollout mode, fatal static schema/frame/config readiness blocking startup, checkpoint-only failure not blocking RAG, and app state exposing only immutable registries/facade.
- [ ] Run `uv run pytest backend/tests/test_rag_v2_graph.py backend/tests/test_rag_application_facade.py backend/tests/test_rag_rollout.py backend/tests/test_agent_runtime_lifespan.py -q` and confirm RED.
- [ ] Implement each graph node as a small typed function and every multi-branch transition as `add_conditional_edges`. Construct exactly `StateGraph(RagGraphState, input_schema=RagGraphInput, output_schema=RagGraphOutput, context_schema=RagRuntimeContext).compile()` with no checkpointer. The graph must invoke LangChain Runnables/model boundaries built in prior tasks; do not call feature implementations from API routes.
- [ ] Compile once at application composition without a checkpointer, register the exact workflow/version, and inject request-local runtime context through LangGraph's runtime/config mechanism without mutating the compiled graph.
- [ ] Implement the facade as the sole selector of legacy versus V2 ownership. Preserve the existing legacy `answer_question_with_rag` service unchanged for disabled/non-cutover paths.
- [ ] Rerun focused suites and `uv run ruff check backend/app/agent_runtime/rag_graph.py backend/app/agent_runtime/rag_v2_state.py backend/app/agent_runtime/rag_v2_composition.py backend/app/agent_runtime/rag_application.py backend/app/agent_runtime/rag_rollout.py backend/app/agent_runtime/rag_v2_registry.py backend/app/main.py backend/tests/test_rag_v2_graph.py backend/tests/test_rag_application_facade.py backend/tests/test_rag_rollout.py`.
- [ ] Commit with `git commit -m "feat: compile rag answer langgraph"`.

---

## Phase D — V1 API and Same-Screen Assistant UX

### Task 15: Freeze V1 DTOs and Route `/ask` and `/search` Through the Facade

**Files:**

- Create: `backend/app/schemas/rag.py`
- Create: `backend/app/api/v1/rag_delivery.py`
- Create: `backend/tests/test_rag_v1_response_contracts.py`
- Create: `backend/tests/test_rag_api_delivery.py`
- Modify: `backend/app/agent_runtime/rag_application.py`
- Modify: `backend/app/schemas/ask.py`
- Modify: `backend/app/schemas/search.py`
- Modify: `backend/app/api/v1/ask.py`
- Modify: `backend/app/api/v1/search.py`
- Modify: `backend/tests/test_ask_api.py`
- Modify: `backend/tests/test_search_permissions.py`
- Modify: `backend/tests/test_search_retrieval_backend.py`

**Interfaces:**

```python
DirectRagApplicationOutcome = RagResultOutcome | Literal[
    'permission_denied', 'invalid_input', 'input_safety_blocked',
    'input_scanner_unavailable', 'persistence_failed',
]

class ExactV1Projection(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

class RagCitationResponse(ExactV1Projection):
    source_id: str
    source_url: str
    source_type: str | None
    permission_level: str
    source_snippet: str
    relevance_score: float
    matched_terms: list[str]

PublicRagErrorCode = Literal[
    'input_safety_blocked', 'input_scanner_unavailable', 'permission_denied',
    'budget_exceeded', 'runtime_version_unavailable', 'retriever_not_configured',
    'retriever_unavailable', 'model_unavailable', 'provider_safety_unavailable',
    'provider_response_identity_invalid', 'provider_usage_overrun',
    'provider_embedding_payload_invalid', 'model_provider_failed',
    'structured_output_invalid', 'citation_validation_failed',
    'persistence_failed', 'unexpected_internal_error',
]

@dataclass(frozen=True, slots=True)
class RagPublicErrorProjection:
    code: PublicRagErrorCode  # HTTP mapper alone wraps this as exact {'detail': {'code': code}}

class RagTokenUsageResponse(ExactV1Projection):
    input_tokens: int
    output_tokens: int
    total_tokens: int

class AskV1Projection(ExactV1Projection):
    agent_name: str
    prompt_version: str
    question: str
    answer: str
    source_ids: list[str]
    source_links: list[str]
    source_snippets: list[str]
    citations: list[RagCitationResponse]
    permission_level: str | None
    hidden_match_count: int
    permission_notice: str | None
    agent_run_id: int | None
    cache_key: str
    model_name: str
    estimated_cost_usd: float
    token_usage: RagTokenUsageResponse

class SearchCostPolicyResponse(ExactV1Projection):
    embedding_query_call: bool
    paid_llm_call: Literal[False]
    requires_pgvector_flag: Literal[True]

class SearchResultResponse(ExactV1Projection):
    id: int
    source_id: str
    text: str
    source_snippet: str
    source_url: str
    source_type: str | None
    permission_level: str
    relevance_score: float
    matched_terms: list[str]
    citation: RagCitationResponse
    parser_status: str | None
    parser_status_reason: str | None
    revision_id: str | None

class SearchV1Projection(ExactV1Projection):
    retrieval_backend: Literal['deterministic_lexical', 'pgvector']
    cost_policy: SearchCostPolicyResponse
    hidden_match_count: int
    results: list[SearchResultResponse]
    permission_notice: Literal['Some sources may be hidden by permissions.'] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

@dataclass(frozen=True, slots=True)
class DirectRagDeliveryResult(Generic[T]):
    public_status: Literal[200, 403, 409, 422, 500, 502, 503]
    body_kind: Literal['success', 'permission_error', 'validation_error', 'budget_error', 'runtime_error', 'generation_error', 'persistence_error']
    application_outcome: DirectRagApplicationOutcome
    projection: T | None
    error: RagPublicErrorProjection | None

class RagApplicationFacade:
    def invoke_ask(self, *, actor: DemoUser, caller_text: str) -> DirectRagDeliveryResult[AskV1Projection]: ...
    def invoke_search(self, *, actor: DemoUser, caller_text: str) -> DirectRagDeliveryResult[SearchV1Projection]: ...
```

- [ ] Add RED recursive key-set tests for `/ask`, `/search`, and shared citation: required-nullable keys are always present; citation/search URL is required non-null; only top-level `/search.permission_notice` is conditionally omitted; no internal graph/slot/identity/fallback key can appear.
- [ ] Add RED public error decoder/OpenAPI tests for the exact `PublicRagErrorCode` union, including `input_safety_blocked`, unknown-code rejection, FastAPI validation `detail:[...]` preservation, and typed `{'detail': {'code': ...}}` only where the spec permits it.
- [ ] Add RED `/ask` result table tests for supported, no-match, hidden-only, safety-filter-empty, pre-generation evidence unavailable, insufficient, post-generation evidence unavailable, exact caller echo, empty four arrays, nullable permission, hidden/notice values, actual/fake-zero cost semantics, and error status mapping.
- [ ] Add RED `/search` tests for effective backend after fallback, legacy opaque result ID, hidden count/notice omission, zero generation calls, nullable parser/source fields, and provider-free keyword cost.
- [ ] Add RED route-boundary tests proving routes import/call only `app.state.rag_application_facade`, do not construct a retriever/store/model/graph, and copy already-committed projections without re-reading DB state.
- [ ] Add RED scanner-readiness outage tests for direct `/ask` and `/search`: typed 503, zero retrieval/model/provider calls, and zero AgentRun/product mutation. Distinguish scanner unavailable from a detected unsafe input and from post-dispatch generation failure.
- [ ] Run `uv run pytest backend/tests/test_rag_v1_response_contracts.py backend/tests/test_rag_api_delivery.py -q` and confirm RED.
- [ ] Implement explicit Pydantic response models and one mapper from typed facade result to HTTP. Remove direct pgvector builder, agent-service, retrieval, settings, and DB composition from `/ask` and `/search` routes.
- [ ] Preserve exact disabled/non-cutover V1 values and status bodies. V2 enforce may change prompt version value to `rag-answer:v2`, never the key meaning.
- [ ] Rerun `uv run pytest backend/tests/test_rag_v1_response_contracts.py backend/tests/test_rag_api_delivery.py backend/tests/test_ask_api.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py -q`.
- [ ] Run `uv run ruff check backend/app/schemas/rag.py backend/app/schemas/ask.py backend/app/schemas/search.py backend/app/api/v1/rag_delivery.py backend/app/api/v1/ask.py backend/app/api/v1/search.py backend/tests/test_rag_v1_response_contracts.py backend/tests/test_rag_api_delivery.py`.
- [ ] Commit with `git commit -m "refactor: route direct rag surfaces through facade"`.

### Task 16: Add Assistant Delivery Algebra and Render Capability Guard

**Files:**

- Create: `backend/app/assistant/delivery.py`
- Create: `backend/app/assistant/capability.py`
- Create: `backend/tests/test_assistant_delivery_contract.py`
- Create: `backend/tests/test_assistant_capability.py`
- Modify: `backend/app/schemas/assistant.py`
- Modify: `backend/app/api/v1/assistant.py`
- Modify: `backend/tests/test_assistant_api.py`

**Interfaces:**

```python
AssistantBodyKind = Literal[
    'assistant_message', 'budget_error', 'generation_error',
    'permission_error', 'owner_not_found', 'validation_error',
    'persistence_error', 'reconciliation_required',
]

@dataclass(frozen=True, slots=True)
class AssistantDeliveryResult:
    application_outcome: AssistantApplicationOutcome
    assistant_message_id: int | None
    body_kind: AssistantBodyKind
    delivery_state: Literal['committed_success', 'committed_safe_failure', 'committed_run_failure', 'not_persisted', 'commit_unknown']
    parent_agent_run_id: int | None
    public_status: Literal[200, 403, 404, 409, 422, 500, 502]

class AssistantMessageResponse(ExactV1Projection):
    id: int
    conversation_id: int
    role: str
    content: str
    citations: list[RagCitationResponse]
    source_ids: list[str]
    source_links: list[str]
    source_snippets: list[str]
    permission_level: str | None
    hidden_match_count: int
    permission_notice: str | None
    agent_run_id: int | None
    metadata: dict[str, object]
    created_at: str

def require_rag_render_capability(raw_headers: Sequence[tuple[bytes, bytes]]) -> None: ...
```

- [ ] Add RED exhaustive construction tests for the approved delivery-state/status/body/ID/outcome legal matrix and rejection of every cross-combination. `commit_unknown` cannot infer IDs; safe 200 is committed success, not failure.
- [ ] Add RED raw-ASGI-header tests for exact one `X-ParaWorks-Rag-Render-Capability: rag-v2-plain-text-citations:v1`; missing, duplicate, comma-folded, or wrong value returns exact `409 {'detail': {'code': 'client_upgrade_required'}}`.
- [ ] Add RED guard-precedence tests: authentication -> Pydantic/strict Unicode -> owner-hidden 404 -> capability 409 -> scanner -> mutation/RAG. First-turn valid body checks capability before conversation creation; malformed input and foreign owner retain 422/404.
- [ ] Add RED GET tests: conversation list requires capability only when a live V2 row contributes to a returned summary, and message GET requires it only when the target projection contains a live V2 row. Cover stale/redacted/non-contributing/unrelated V2 rows and legacy-only payloads; invalid capability must expose zero contributing V2 body bytes.
- [ ] Add RED recursive Assistant serialization tests requiring every `citations[]` item to use `RagCitationResponse`'s exact seven keys with required non-null URL and no slot/support/HMAC/internal identity fields.
- [ ] Add RED scanner-readiness tests for first-turn and existing-conversation Assistant POST: generic 500, no conversation INSERT/UPDATE, no user/assistant message, no AgentRun, and no provider call. This is a pre-user `not_persisted` boundary, never a committed safe-failure writer.
- [ ] Add RED response-header tests for capability-dependent Assistant POST/GET success and errors: exact `Cache-Control: private, no-store` and `Vary: X-ParaWorks-Rag-Render-Capability`.
- [ ] Run `uv run pytest backend/tests/test_assistant_delivery_contract.py backend/tests/test_assistant_capability.py -q` and confirm RED.
- [ ] Implement the validated algebra and refusal-only capability guard. It never enables a stage, chooses legacy, or claims signed-client identity.
- [ ] Add only the guard/helper integration needed for RED tests; Task 18 performs V2 sole-writer route cutover.
- [ ] Rerun focused tests plus `uv run pytest backend/tests/test_assistant_api.py -k "capability or owner or validation" -q`.
- [ ] Run `uv run ruff check backend/app/assistant/delivery.py backend/app/assistant/capability.py backend/app/schemas/assistant.py backend/app/api/v1/assistant.py backend/tests/test_assistant_delivery_contract.py backend/tests/test_assistant_capability.py`.
- [ ] Commit with `git commit -m "feat: enforce assistant rag render capability"`.

### Task 17: Read and Revalidate Assistant Evidence Integrity

**Files:**

- Create: `backend/app/assistant/evidence_reader.py`
- Create: `backend/tests/test_assistant_evidence_v2.py`
- Modify: `backend/app/assistant/service.py`
- Modify: `backend/tests/test_assistant_models.py`
- Modify: `backend/tests/test_assistant_service.py`
- Reuse unchanged: `backend/app/assistant/evidence_persistence.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class AssistantMessageView:
    id: int
    conversation_id: int
    role: str
    content: str
    citations: tuple[RagCitationResponse, ...]
    source_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    source_snippets: tuple[str, ...]
    permission_level: str | None
    hidden_match_count: int
    permission_notice: str | None
    agent_run_id: int | None
    metadata: Mapping[str, object]
    created_at: str

class AssistantEvidenceReader:
    def project_message(
        self,
        *,
        db: Session,
        actor: DemoUser,
        message: AssistantMessage,
    ) -> AssistantMessageView: ...
```

- [ ] Add RED writer/reader integration tests for `rag_v2_exact` byte preservation versus legacy `.strip()`, assembled/canned XOR, linked final AgentRun/result/content HMACs, exact child/set HMAC, all influence dependencies, and zero-child canned messages.
- [ ] Add RED whole-message redaction tests for key version, content, mode/origin, AgentRun swap, result HMAC, permission, source version/signature, approval/revoke, evidence-link set, selected child, dependency order/role/scope, citation HMAC, or one-byte mutation.
- [ ] Add RED compatibility tests for historical null-marker rows, valid `legacy_v1_only` integrity snapshots that never grant V2 eligibility, and disabled/non-cutover `legacy_unbound` POST/GET retaining exact V1 values.
- [ ] Add RED serialization/context tests proving `serialize_message`, list GET, summary source selection, and `eligible_context_messages` use the same reader; hidden count/notice alone is not an evidence-backed redaction trigger.
- [ ] Add RED privacy tests removing public `metadata.question`, graph/backend/fallback/internal IDs, and raw exception text while preserving safe `agent_name`/`prompt_version` and exact legacy failure triple values.
- [ ] Run `uv run pytest backend/tests/test_assistant_evidence_v2.py backend/tests/test_assistant_models.py backend/tests/test_assistant_service.py -q` and confirm RED.
- [ ] Implement a single reader over the Task 13 writer and Task 4 resolver. It must revalidate before every GET/context use and return a whole safe `evidence_unavailable` projection on any mismatch.
- [ ] Do not let integrity HMAC success synthesize missing provenance or promote knowledge.
- [ ] Rerun focused tests and `uv run ruff check backend/app/assistant/evidence_reader.py backend/app/assistant/service.py backend/tests/test_assistant_evidence_v2.py backend/tests/test_assistant_models.py backend/tests/test_assistant_service.py`.
- [ ] Commit with `git commit -m "feat: revalidate assistant rag evidence"`.

### Task 18: Make the Facade the Assistant RAG Sole Writer

**Files:**

- Modify: `backend/app/api/v1/assistant.py`
- Modify: `backend/app/agent_runtime/rag_application.py`
- Modify: `backend/app/assistant/service.py`
- Modify: `backend/tests/test_assistant_api.py`
- Modify: `backend/tests/test_assistant_delivery_contract.py`
- Modify: `backend/tests/test_assistant_capability.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class PreparedAssistantIngress:
    conversation_id: int | None
    owner_user_id: str
    prepared_text: PreparedRagRequestText
    prior_context_hmac: str
    capability_version: Literal['rag-v2-plain-text-citations:v1']

class RagApplicationFacade:
    def prepare_assistant_ingress(
        self,
        *,
        actor: DemoUser,
        conversation_id: int | None,
        caller_text: str,
    ) -> PreparedAssistantIngress: ...

    def invoke_assistant(
        self,
        *,
        actor: DemoUser,
        conversation_id: int,
        user_message_id: int,
        prepared_ingress: PreparedAssistantIngress,
        tool_recorder: SafeRagToolRecorder | None = None,
    ) -> AssistantDeliveryResult: ...
```

- [ ] Add RED tests that validation/owner/capability/scanner/prior-context checks occur before user/conversation/provider mutation; invalid prior context yields `not_persisted` 502 with both IDs null and leaves seeded rows byte/count unchanged. Repeat the scanner-unavailable first-turn/existing-conversation matrix here against the completed facade/sole-writer path and require generic 500 plus zero conversation/message/AgentRun/provider mutation.
- [ ] Add RED post-user tests for provider-free pre-dispatch failure, inter-component failure after terminal embedding, provider/schema/citation/overrun failure, supported and canned success, phase-1 final-message rollback, commit-before-ACK, and post-commit exception. Assert exact one final parent/message where committed and no second append in route/catch.
- [ ] Add RED mapper tests proving the route trusts `AssistantDeliveryResult` rather than reinterpreting exception/outcome; `commit_unknown` triggers generic 500 and GET reconciliation only.
- [ ] Add RED branch-isolation tests keeping contact lookup, recipient clarification, email draft/composer, and non-RAG legacy writers unchanged. Email RAG context must go through a facade compatibility method, not a direct feature-agent import.
- [ ] Add RED V2 failure metadata tests: `status='failed'`, bounded application outcome, `failure_class='RagV2SafeFailure'`; safe-200 canned messages have no failure triple; legacy remains exact.
- [ ] Run `uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_delivery_contract.py backend/tests/test_assistant_capability.py -q` and observe sole-writer failures.
- [ ] Implement the sole-writer cutover by moving the non-email enforce RAG branch to `prepare_assistant_ingress` plus `invoke_assistant`. Remove V2 route-side success append and catch-path safe append; look up the committed message ID read-only for response projection.
- [ ] Preserve route-owned writes only for disabled/non-cutover legacy and explicitly non-RAG flows.
- [ ] Rerun `uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_assistant_delivery_contract.py backend/tests/test_assistant_capability.py backend/tests/test_assistant_evidence_v2.py -q`.
- [ ] Run `uv run ruff check backend/app/api/v1/assistant.py backend/app/agent_runtime/rag_application.py backend/app/assistant backend/tests/test_assistant_api.py`.
- [ ] Commit with `git commit -m "refactor: make rag facade the assistant answer writer"`.

### Task 19: Correct Frontend DTOs and Capability Transport

**Files:**

- Create: `frontend/src/lib/api/assistant.ts`
- Create: `frontend/tests/rag-v2-types.fixture.ts`
- Create: `frontend/e2e/rag-v2-assistant.spec.ts`
- Modify: `frontend/src/lib/api/client.ts`
- Modify: `frontend/src/lib/api/types.ts`
- Modify: `frontend/src/app/search/page.tsx`

**Interfaces:**

```ts
export const RAG_RENDER_CAPABILITY = "rag-v2-plain-text-citations:v1" as const;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string | null,
    message: string,
  ) { super(message); }
}
```

- [ ] Add a compile-time fixture requiring nullable-but-present `AskResponse.permission_level`, `permission_notice`, `agent_run_id`; `AssistantConversation.summary`; Assistant message nullable fields; search `source_type/parser_status/parser_status_reason/revision_id`; and `RagCitation.source_type`. Require non-null search/citation `source_url`.
- [ ] Add RED Playwright request capture in `frontend/e2e/rag-v2-assistant.spec.ts` proving Assistant GET/POST sends the capability header exactly once and other APIs do not silently claim it.
- [ ] Add RED error decoding for the exact server code allowlist; unknown code or raw English response text must not become user-visible copy.
- [ ] Run `Push-Location frontend; npx tsc --noEmit; $env:PLAYWRIGHT_MANAGED_SERVER='1'; npm run test:visual -- e2e/rag-v2-assistant.spec.ts --project=chromium-desktop; Pop-Location` and observe the nullable/required/capability transport failures.
- [ ] Implement the DTO corrections, `ApiError`, and typed Assistant wrappers. Do not make fields optional when the backend returns explicit null.
- [ ] Rerun `Push-Location frontend; npx tsc --noEmit; npm run lint; $env:PLAYWRIGHT_MANAGED_SERVER='1'; npm run test:visual -- e2e/rag-v2-assistant.spec.ts --project=chromium-desktop; Pop-Location`.
- [ ] Commit with `git commit -m "fix: align frontend rag v1 contracts"`.

### Task 20: Render V2 Safely and Reconcile Uncertain Same-Screen Delivery

**Files:**

- Create: `frontend/src/lib/rag/presentation.ts`
- Create: `frontend/src/lib/assistant/searchHandoff.ts`
- Create: `frontend/src/lib/assistant/reconciliation.ts`
- Modify: `frontend/e2e/rag-v2-assistant.spec.ts`
- Modify: `frontend/src/components/layout/AppShell.tsx`
- Modify: `frontend/src/app/search/page.tsx`
- Modify: `frontend/e2e/assistant-memory.spec.ts`
- Modify: `frontend/e2e/visual-smoke.spec.ts`

**Interfaces:**

```ts
export function isSafeCitationUrl(value: string): boolean;
export const EphemeralSearchHandoff: {
  put(raw: string): void;
  consume(): string | null;
};
export type DeliveryStatus = "sending" | "persisted" | "unknown";
```

- [ ] Add RED Playwright cases for V2/unknown RAG prompt rendering as a React text node with `white-space: pre-wrap` and `overflow-wrap:anywhere`; Markdown, raw HTML, `javascript:`, and URL-like model text stay literal. V1 and non-RAG rendering remains unchanged.
- [ ] Add RED citation tests for absolute HTTP(S), no userinfo/whitespace/control, `rel='noopener noreferrer'`, and invalid server citation becoming a non-clickable source label rather than a content link. For V2 and unknown-RAG rows, `message.source_links` must produce zero anchors/URLs; only validated `RagCitation.source_url` is clickable. Preserve the legacy source-link section unchanged for proven V1 rows.
- [ ] Add RED handoff tests proving AppShell stores raw input in consume-once module memory, navigates only to `/search`, never writes query/history/session/local storage/analytics, discards inbound `?q=`, and safely loses state on reload without auto-send.
- [ ] Add RED optimistic reconciliation tests: request token + conversation guard; 403/404/422 removes only owned optimistic row; 500/502 performs guarded GET; authoritative rows replace exactly once; refetch failure leaves one `unknown` row; unknown retry is GET-only; stale response cannot overwrite a new conversation. Split 409 by typed code: `budget_exceeded` is a committed safe failure and must guarded-GET authoritative user/assistant rows, while `client_upgrade_required` allows hard reload only, never resend/retry; POST remains disabled until a successful GET proves absence.
- [ ] Add RED Korean-copy and placement tests for input safety, budget, hidden permission, evidence unavailable, client-upgrade reload, 500 unknown state, and 502 generation failure. Assert machine codes/raw English never render, and permission/hidden-only notice is visible immediately below the answer before the evidence accordion is opened; citations/snippets remain inside the existing accordion.
- [ ] Run `Push-Location frontend; $env:PLAYWRIGHT_MANAGED_SERVER='1'; npm run test:visual -- e2e/rag-v2-assistant.spec.ts e2e/visual-smoke.spec.ts --project=chromium-desktop; Pop-Location` and confirm RED.
- [ ] Implement safe renderer, URL validator, ephemeral handoff, typed 409 reconciliation, and notice placement without a new page/modal/wizard or additional click depth. Keep citations/snippets in the existing accordion and move only permission/hidden notice directly below the answer.
- [ ] Rerun `Push-Location frontend; npm run lint; npm run build; $env:PLAYWRIGHT_MANAGED_SERVER='1'; npm run test:visual -- e2e/rag-v2-assistant.spec.ts e2e/assistant-memory.spec.ts e2e/visual-smoke.spec.ts --project=chromium-desktop; npm run test:visual -- e2e/rag-v2-assistant.spec.ts --project=chromium-mobile; Pop-Location`.
- [ ] Commit with `git commit -m "feat: render and reconcile rag v2 safely"`.

---

## Phase E — Shadow, Safety Operations, and Release Proof

### Task 21: Implement Staged Rollout and Retrieval-Only Shadow

**Files:**

- Modify: `backend/app/agent_runtime/rag_rollout.py`
- Create: `backend/app/rag/shadow.py`
- Create: `backend/tests/fixtures/rag_v2_provider_free_golden_60.json`
- Create: `backend/tests/test_rag_v2_provider_free_golden.py`
- Modify: `backend/tests/test_rag_rollout.py`
- Create: `backend/tests/test_rag_shadow.py`
- Modify: `backend/app/agent_runtime/rag_application.py`
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_settings_env.py`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`

**Interfaces:**

```python
ShadowDeltaKind = Literal[
    'v2_source_observation_delta', 'trust_tier_reorder_delta',
    'raw_public_identity_repair_delta', 'bounded_hidden_delta',
    'assistant_context_security_delta',
]

@dataclass(frozen=True, slots=True)
class LegacyRetrievalCandidateObservation:
    ordinal: int
    legacy_serving_document_id: str  # request-local only; never persisted
    visibility: Literal['visible', 'denied_known']
    legacy_public_source_id: str | None
    support_class: Literal['trusted_candidate', 'raw_candidate']
    effective_permission: Literal['public', 'internal', 'restricted']
    relevance_score: float
    matched_terms: tuple[str, ...]
    public_projection_hmac: str | None
    candidate_identity_hmac: str

@dataclass(frozen=True, slots=True)
class LegacyRetrievalObservation:
    surface: RagSurface
    configured_backend: RagRetrievalBackend
    effective_backend: RagEffectiveBackend
    query_context_version: Literal['direct-query:v1', 'assistant-context:v1']
    retrieval_query_hmac: str
    security_scope_fingerprint: str
    candidate_window: tuple[LegacyRetrievalCandidateObservation, ...]
    candidate_window_hmac: str
    visible_result_set_projection_hmac: str
    hidden_match_count: int
    hidden_count_capped: bool
    query_embedding_attempt_fence_hmac: str | None
    public_agent_run_correlation_hmac: str | None
    latency_ms: int
    observation_hmac: str

@dataclass(frozen=True, slots=True)
class ShadowComparison:
    surface: RagSurface
    configured_backend: RagRetrievalBackend
    effective_backend: RagEffectiveBackend
    outcome: Literal['shadow_match', 'shadow_mismatch']
    security_scope_fingerprint: str
    legacy_observation_hmac: str
    legacy_public_run_correlation_hmac: str | None
    v2_top_candidate_window_hmac: str
    legacy_candidate_count: int
    v2_candidate_count: int
    common_cohort_count: int
    common_cohort_exact_match_count: int
    legacy_hidden_match_count: int
    v2_hidden_match_count: int
    v2_source_observation_delta_count: int
    trust_tier_reorder_delta_count: int
    raw_public_identity_repair_delta_count: int
    bounded_hidden_delta_count: int
    assistant_context_security_delta_count: int
    trusted_comparison_projection_unavailable_count: int
    unclassified_shadow_mismatch_count: int
    common_cohort_hmac: str
    intended_delta_set_hmac: str
    comparison_hmac: str
    latency_ms: int

class ShadowComparator:
    def compare(self, *, legacy: LegacyRetrievalObservation, v2: RetrievalResult, scope: SecurityScope) -> ShadowComparison: ...
```

- [ ] Add RED exhaustive `mode × stage × surface` tests for public owner, V2 retrieval, generation count, embedding count, AgentRun/Audit owner, stage cumulative order, disabled-stage ignore, no request/header override, and no enforce background shadow.
- [ ] Add RED keyword-shadow tests for legacy public bytes/run unchanged, V2 generation/provider/cost owner zero, canonical comparison only, and rollback immediately returning to legacy-only.
- [ ] Add RED pgvector-shadow tests for one shared immutable query embedding only when legacy/V2 query bytes match, one internal exact-two cost owner, terminal-zero generation child, legacy public run/body untouched, no internal run ID leakage, and strict usage/safety override.
- [ ] Add RED serving-index-not-ready shadow tests: validated vector may serve legacy only, V2 comparison/Audit zero, internal parent final `serving_index_not_ready`, and no generic pending projection.
- [ ] Add RED Assistant prior-assistant context tests producing only `assistant_context_security_delta`, no V2 comparison/shared carrier/internal cost, and at most one standalone legacy embedding.
- [ ] Add RED comparison tests for common cohort and exact five allowed deltas: `v2_source_observation_delta`, `trust_tier_reorder_delta`, `raw_public_identity_repair_delta`, `bounded_hidden_delta`, `assistant_context_security_delta`. Any other membership/rank/projection/permission/relevance mismatch is red.
- [ ] Add RED provider-free golden-manifest tests requiring at least 60 unique deterministic/fake cases across all three surfaces, keyword/pgvector, trusted/raw, public/internal/restricted, no-match/hidden/stale/revoked/hard-negative and V1 parity cohorts. Assert an exact collected-case count, fake model/embedding transports, zero network/provider calls, zero permission/revoke/stale leaks, and zero invalid/missing evidence slots.
- [ ] Add RED trace tests persisting only aggregate counts/latency/cost/domain HMACs, never raw query/evidence/IDs/tuples.
- [ ] Add RED retained-blocker matrix: disabled/non-cutover/rollback legacy remains exact; shadow D work stops and advancement is red; enforce cutover maps component safety error; rollback never clears the blocker.
- [ ] Run `uv run pytest backend/tests/test_rag_rollout.py backend/tests/test_rag_shadow.py backend/tests/test_settings_env.py backend/tests/test_agent_runtime_lifespan.py -q` and confirm RED.
- [ ] Implement deployment-static policy, read-only comparator, and the sanitized provider-free golden fixture. Shadow performs retrieval comparison only; never dual generation.
- [ ] Rerun focused tests and `uv run pytest backend/tests/test_rag_v2_provider_free_golden.py -q`; then run `uv run ruff check backend/app/agent_runtime/rag_rollout.py backend/app/rag/shadow.py backend/app/agent_runtime/rag_application.py backend/tests/test_rag_rollout.py backend/tests/test_rag_shadow.py backend/tests/test_rag_v2_provider_free_golden.py`.
- [ ] Commit with `git commit -m "feat: add staged rag v2 rollout"`.

### Task 22: Add Provider-Free Safety Administration

**Files:**

- Create: `backend/app/admin/rag_provider_safety.py`
- Create: `backend/tests/test_rag_provider_safety_admin.py`
- Create: `backend/tests/test_rag_provider_safety_postgres.py`
- Modify: `backend/app/agent_runtime/rag_provider_safety.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`

**Interfaces:**

```text
uv run python -m backend.app.admin.rag_provider_safety provider-safety-status
uv run python -m backend.app.admin.rag_provider_safety provider-safety-init
uv run python -m backend.app.admin.rag_provider_safety provider-safety-bootstrap-recovery
uv run python -m backend.app.admin.rag_provider_safety provider-safety-mark-rebind-required
uv run python -m backend.app.admin.rag_provider_safety provider-safety-rebind
uv run python -m backend.app.admin.rag_provider_safety provider-safety-reset
uv run python -m backend.app.admin.rag_provider_safety provider-safety-supersede
```

- [ ] Add RED initialization tests for absent latch, trusted stable sidecar, exact-empty authority/readiness/history, zero D paid attempts, active key/config/policy identities, exact two ready families at generation zero/state version one, one bootstrap transition, and second initialization refusal.
- [ ] Add RED failure tests for file-first crash recovery only when DB/attempts remain exact empty; partial/corrupt/mismatched artifacts require isolated new authority and fresh preview rather than repair.
- [ ] Add RED reset/rebind/supersede tests for authenticated reviewed command input, monotonic global/family generations, append-only history, blocker preservation, no cross-family widening, and zero provider call.
- [ ] Add RED production/live target tests: production initialization targets the production app DB; live-gate initialization targets the exact validation DB; neither authority can satisfy the other.
- [ ] Add RED output hygiene tests for aggregate status only, no key/path secret/query/evidence/config plaintext, no credentials in CLI args/env/log/DB, and non-interactive fail-closed behavior.
- [ ] Require every mutating CLI command, including init and bootstrap recovery, to consume one bounded canonical review envelope from stdin. The envelope is signed by a distinct external review key, binds operation/CAS/context/target and an externally fixed implementation-plan reference HMAC, and is never accepted through argv, environment, logs, or output. The admin CLI boundary verifies only and exposes no signer; existing low-level service/test capabilities remain internal compatibility surfaces.
- [ ] Pin the review key id and opaque key-material verifier in both a committed-code registry and a runtime-key-authenticated, owner-only append-only review ledger. Record one durable nonce reservation and consumption event per envelope; reject nil/noncanonical/reused nonce, key drift, ledger drift, and settings-only key replacement. Key material is never committed or persisted.
- [ ] Create that ledger only inside fresh-init's provider-latch/advisory critical section after exact-empty DB/zero-attempt revalidation and before authority-file creation. Existing or partial authority plus a missing/unreadable/mismatched ledger is inconsistent and cannot be auto-adopted or repinned; legacy adoption requires a separately approved future migration.
- [ ] Bind bootstrap recovery to the exact inspected partial latch: target/environment, authority UUID, generation-zero envelope digest, original reviewed reference, review-key pin, and fixed plan HMAC. A review for partial authority A must not recover replacement authority B.
- [ ] Enforce operation-specific envelope schemas: init/recovery/mark/reset have no successor, rebind/supersede require an exact successor, and irrelevant acknowledgement fields are refused rather than ignored. Latch-absent status inspects authority, readiness, and history sets read-only and reports any orphan as inconsistent.
- [ ] Make rebind two separately reviewed transitions: `provider-safety-mark-rebind-required` first, then `provider-safety-rebind`; rebind refuses every state except `rebind_required`.
- [ ] Keep production/application and live-validation DB+latch targets disjoint in settings and target identity. A review for one target must not authorize the other.
- [ ] Require a supersession successor to match both the signed snapshot and a committed-code allowlist. Keep key rotation out of Task 22 and fail closed until a separately approved design exists.
- [ ] Run `uv run pytest backend/tests/test_rag_provider_safety_admin.py backend/tests/test_rag_provider_safety_postgres.py -q` and confirm RED.
- [ ] Implement commands over `RagProviderSafetyService` and durable file authority. Every command is provider-free; none grants a paid permit.
- [ ] Rerun focused tests and `uv run ruff check backend/app/admin/rag_provider_safety.py backend/app/agent_runtime/rag_provider_safety.py backend/tests/test_rag_provider_safety_admin.py backend/tests/test_rag_provider_safety_postgres.py`.
- [ ] Commit with `git commit -m "feat: administer rag provider safety"`.

### Task 23: Isolate the Exact-Six Live Release Authority

**Current status: IMPLEMENTED / NOT RELEASE-CLEAN / 1 load-bearing P1 carried.**
The implementation candidate remains `a9c729166c50b8b38f85fada45b7bbb96be9c1ce`.
The final reviewer accepted R4-B, but `case_claim` INSERTs with `before=None`
still bypass approved initial-value semantics. A complete caller-supplied HMAC
can bind forged parent permission/provider/route/tokens/metadata or child
config/policy HMACs without proving that the reviewed manifest authorized them.
Task23 cannot derive the approved initial projections because the frozen
reviewed 30-case manifest contract is introduced in Task24.

Ruling: Task23 breaker 5/5; P1 is real and release-blocking; carry into Task24
mandatory first RED slice. Task24 must define canonical approved
manifest→case_claim AgentRun/exact-two cost-child projection, validate it before
SQL/incident, re-review this carryover CLEAN before preview/authorization work.
Cost if wrong: forged runtime/provider/cost metadata could enter release ledger.
This ruling supersedes earlier pending-rereview/next-task statements below;
it does not mark Task23 COMPLETE or CLEAN or authorize release operations.

**Ruling (2026-09-13, user-approved):** the canonical `affected_rows` contract
contains only rows with a semantic before/after mutation. Same-barrier locked,
read-only peers required to validate authorization, case, runtime, or provider
safety state belong to a separately domain-separated `observation_set`. Each
observation carries only typed row-identity and row-projection HMACs; it never
exposes raw IDs or row values. Missing, changed, duplicated, overlapping, or
out-of-barrier observations fail closed, and the observation set is part of the
canonical transition HMAC. No no-op, `updated_at`-only, or clock-only write may
be introduced to satisfy an affected-set matrix.

**Round 4 implementation clarification (2026-09-13):** current provider proof is
mandatory for bootstrap and every later transition. Its digest represents the
current after-image, while the authorization's approved envelope remains
immutable. Bind the complete provider readiness set and scoped case/dispatch/
parent/exact-two-child roster through observations; validate those observations
before SQL or sealed incident application. Ordinary transitions require approved
equality and ready active families. Snapshot aborts prove drift/non-ready, sealed
component incidents prove approved-before/blocked-after, and crash/corpus aborts retain
an internally valid current snapshot. Enforce once-bound execution/cost fences,
generation admission-to-pending parent mutation, pending-to-failed terminal-cost
observations, and 30-case ordinary zero-dispatch terminal failure. Pending crash
recovery retains its projection fence and paid cost in failed/final; admission
crash remains failed/admission_only. Retain exact-six schema, no application
metadata/Alembic/public endpoint changes, and release-review separation.

Permanent round-4 tests include actual SQL append lifecycle/charge preservation,
query-parent observations, case projection, full 30/30/60 terminal roster,
provider drift/control/component-snapshot/case-null aborts, roster tampering and
digest changes, once-bound owner/fence, and pre-SQL observation rejection.
Conditional PostgreSQL tests exercise the real pinned barrier and both sealed
component incidents and later snapshot abort. Independent rereview remains the
next gate; Task24 and actual release/provider operations have not started.

**Round 5 final scoped remediation (R4-A/R4-B, 2026-09-13):** first reproduce
affected-parent permission/provider/route/token/metadata/start-time piggyback
changes, paired-child timestamp changes, and identical signatures for different
runtime completion times. Require complete typed before/after
`row_mutation_hmac` on affected AgentRun/cost entries and exact per-kind expected
images, defaulting every other column to immutable. Check literal planned images
against locked before-images before SQL or a sealed provider incident, then
recheck executed images against the same HMAC. Runtime inserts explicitly supply
every column, including physical IDs/timestamps; no implicit defaults or SQL
expressions enter the signed plan. Keep physical schema, ORM/Alembic and public
endpoints unchanged; earlier incomplete candidate history fails closed.

Retain exhaustive actual-append field-diff tests for all 25 AgentRun and 27 cost
columns, stale before/changed after/missing/forged signature and expression
negatives, and UTC completion-before-start checks. Execute all 17 registry kinds
plus both finish-failed outcomes through real SQL append. Assert before/after
rows, exact affected/observed identities, canonical bytes, independently derived
transition HMAC, gapless generations and paired invalid zero-change behavior.
Run real 30-case normal/quality/ordinary/contract-failure sequences. Preserve a
real in-process sealed Task22 incident and conditional PostgreSQL parent-audit
negatives in addition to the existing physical/lock/incident gates. Fresh focused,
direct-impact, Ruff/compile/diff/credential/status verification and a local commit
precede the final independent scoped rereview; release and Task24 remain blocked.

**Files:**

- Create: `backend/app/rag/release_schema.py`
- Create: `backend/app/rag/release_authority.py`
- Create: `backend/app/rag/release_ledger.py`
- Create: `backend/app/admin/rag_live_gate.py`
- Create: `backend/tests/test_rag_live_gate_schema.py`
- Create: `backend/tests/test_rag_release_authority.py`
- Create: `backend/tests/test_rag_release_ledger.py`
- Create: `backend/tests/test_rag_release_ledger_review_q.py`
- Modify: `backend/app/admin/rag_provider_safety.py`
- Modify: `backend/app/agent_runtime/rag_provider_safety.py`
- Modify: `backend/tests/test_rag_release_authority_postgres.py`
- Modify: `backend/tests/release_contracts.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`

**Interfaces:**

```python
RAG_RELEASE_TABLE_NAMES = frozenset({
    'rag_live_gate_ledgers',
    'rag_live_gate_authorizations',
    'rag_live_gate_cases',
    'rag_live_gate_dispatches',
    'rag_live_gate_transitions',
    'rag_live_gate_quality_reports',
})

@dataclass(frozen=True, slots=True)
class RagReleaseSnapshot:
    marker_schema_version: Literal['rag-release-ledger-marker-body:v1']
    ledger_uuid: UUID
    ledger_epoch: int
    generation: int
    last_transition_digest: str | None
    predecessor_marker_digest: str | None
    rebootstrap_reason_hmac: str | None
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    marker_envelope_hmac: str
    marker_file_digest: str
    designated_environment_id_hmac: str
    designated_host_id_hmac: str
    validation_database_identity_hmac: str

def build_rag_release_metadata() -> MetaData: ...  # separate validation-only metadata

class RagReleaseAuthority:
    def initialize(self, ...) -> RagReleaseSnapshot: ...
    def rebootstrap(self, ...) -> RagReleaseSnapshot: ...
    def disaster_initialize(self, ...) -> RagReleaseSnapshot: ...
    def inspect(self, ...) -> RagReleaseSnapshot: ...
```

```text
uv run python -m backend.app.admin.rag_live_gate release-ledger-init
uv run python -m backend.app.admin.rag_live_gate release-ledger-rebootstrap
uv run python -m backend.app.admin.rag_live_gate release-ledger-disaster-init
uv run python -m backend.app.admin.rag_live_gate status
```

- [ ] Add RED tests proving the exact six release tables exist only in a separate validation metadata/schema and are absent from application `Base.metadata`, Alembic revisions, normal `create_all`, and production DB bootstrap.
- [ ] Add RED schema tests for composite ledger UUID/epoch/generation authority, immutable authorization snapshot, case/run links, component dispatch accounting, append-only transition bytes, insert-only unique quality report, exact FKs/checks, and no raw query/evidence/model output columns.
- [ ] Add RED durable marker tests for `PARAWORKS_RELEASE_LEDGER_AUTHORITY_PATH` plus distinct stable `.lock`, provider/release four-leaf cross-alias rejection, canonical envelope/HMAC, DB identity, marker-first/DB-second crash fail-stop, and no runner repair.
- [ ] Add RED transition tests for generation 1..N gaplessness within epoch, immutable row identity across mutable state, exact semantic affected-row digest, separately sealed observation-set digest, transition kind before/after/null matrix, aggregate count/cost equality, and no-op/false affected sets. Include case-outcome unchanged authorization, component unchanged case/runtime parent, missing/changed observation, barrier TOCTOU, and clock-only negative cases.
- [ ] Add RED Task-22 provider-incident seam tests for an internally sealed, one-use plan; stable global lock order; external-first envelope change; same-transaction exact DB authority/readiness generations, state, digest, blocker evidence; and fail-stop external/DB mismatch after DB rollback. Do not broaden existing public/admin reset/rebind behavior and do not call a live provider.
- [ ] Add RED rebootstrap tests for same UUID/new epoch with predecessor/reason HMAC and generation zero, prior rows read-only, old authorization unusable, crash mismatch fail-stop, and disaster-init only for missing/corrupt authority with fresh approval.
- [ ] Run `uv run pytest backend/tests/test_rag_live_gate_schema.py backend/tests/test_rag_release_authority.py backend/tests/test_rag_release_ledger.py backend/tests/release_contracts.py -q` and confirm RED.
- [ ] Implement release authority against one explicitly designated validation PostgreSQL connection. Mutation capture must map canonical `agent_run_id` to physical `AgentRun.id`, bind every payload identity to exact DB after-images or locked peer observations, and enforce the exhaustive per-kind parent/child/dispatch/provider lifecycle. Do not import application ORM models into `release_schema.py` and do not add a third Alembic revision.
- [ ] Rerun focused tests and `uv run ruff check backend/app/rag/release_schema.py backend/app/rag/release_authority.py backend/app/rag/release_ledger.py backend/app/admin/rag_live_gate.py backend/tests/test_rag_live_gate_schema.py backend/tests/test_rag_release_authority.py backend/tests/test_rag_release_ledger.py`.
- [ ] Commit with `git commit -m "feat: add isolated rag release authority"`.

### Task 24: Freeze the 30-Case Manifest and Zero-Call Authorization Preview

**Mandatory first RED slice — carried Task23 P1 (breaker 5/5).** Before any
preview/authorization implementation, define the canonical projection from the
approved frozen manifest to each `case_claim` AgentRun and exact-two cost
children. Reproduce acceptance of forged initial permission/provider/route,
token/cost/metadata and child config/policy identities through actual append.
Implement authoritative comparison with the approved projection before any SQL
or provider incident; a HMAC over submitted values alone is insufficient.
Obtain independent CLEAN rereview of this carryover before beginning the
remaining preview/authorization work in this task. The Task23 implementation
candidate remains `a9c7291`, **IMPLEMENTED / NOT RELEASE-CLEAN / 1 load-bearing
P1 carried** until that gate is closed. No production or test change is part
of this documentation-only breaker closure.

**2026-09-13 first-slice implementation candidate:** `da79bf0` implements only
the carryover above; independent CLEAN rereview is still required. Task24 is
not complete and preview/authorization/OAuth work has not begun.
`release_review.py` freezes the versioned case-claim manifest preimage and
requires its HMAC to equal the locked, immutable authorization `manifest_hmac`.
It derives all 25 parent and both 27-column child images from reviewed case,
provider-policy and reserve inputs plus the existing `rag-run:v2` admission
defaults. Assembly v1 allocates positive unused IDs and one UTC clock sample;
the opaque projection binds the existing transition generation/process/runner
fences. It does not issue a new process identity or paid permit.
The complete literal images, exact child order, case ordinal and fresh provider
readiness are checked before SQL/incident and after SQL. Subprecision money
cannot hide behind six-place HMAC normalization. The full quality manifest
producer must integrate this complete executable preimage into its canonical
review contract; a separately self-signed projection is insufficient.
Fresh focused evidence: `327 passed, 14 skipped`; direct impact and credential
scan: `221 passed, 14 skipped` (11 existing Alembic warnings). PostgreSQL
evidence remains absent because `PARAWORKS_TEST_POSTGRES_URL` is unset.

**2026-09-13 first-slice rereview round 1 candidate: `402f889`.** Finding T24-P2-A
reproduced textual child HMACs replaced by decoded bytes reaching four/five DML
statements before SQLite constraint rollback. The fix requires exact canonical
manifest, issued-binding, case and complete runtime-image types before
normalization. Original provider-policy fields are checked before dataclass
deepcopy; native JSON types cannot alias the observation encoder's float tags.
Initial clocks must retain the assembly's UTC identity and fold, while actual
DB datetime normalization is retained. Independent actual-append oracles require
zero DML, zero sealed incidents and unchanged rows/generation/marker. The narrow
carryover still needs independent CLEAN rereview; Task24 is not complete.
Fresh focused evidence: `716 passed, 14 skipped`; direct impact: `221 passed,
14 skipped`, 11 existing Alembic warnings. Credential checks: `3 passed`;
changed-file Ruff/format, compileall and diff checks passed. The 28 PostgreSQL
skips remain unexecuted because the validation DSN is absent.

**2026-09-13 narrow carryover CLEAN closure:** Independent rereview of
`a67f76b..1de0941`, implementation `402f889`, found no actionable findings.
Reviewer-fresh projection-types/case-claim/round4/round5 evidence was
`592 passed in 164.24s`; Ruff/format/compileall/diff were green and the worktree
was clean. This supersedes the pending rereview status above for this carried
slice only. "Before SQL" here means zero mutation DML and zero sealed provider
incident before rejection; barrier/read SELECTs are expected. PostgreSQL DSN
is absent and its conditional gates remain unexecuted. Task24 overall is still
incomplete; this closure does not approve preview/authorization/OAuth work or
any actual release operation. No production or test change is made by closure.

**Files:**

- Create: `backend/app/rag/release_review.py`
- Create: `backend/tests/fixtures/rag_v2_live_gate_30.json`
- Create: `backend/tests/test_rag_live_gate_preview.py`
- Create: `backend/tests/test_rag_release_review.py`
- Create: `backend/tests/test_rag_live_gate_cli.py`
- Modify: `backend/app/admin/rag_live_gate.py`

**2026-09-13 approved split and identity clarification.** Task24-A implements
the declarative 30-case fixture, frozen snapshots, provider-free preview builder
and read-only CLI readiness surface. Task24-B retains fresh reviewer proof and
authorization construction; Task24 overall remains incomplete.

**2026-09-13 Task24-B candidate (independent review pending).** Fresh three-role
OAuth2+userinfo verification, externally signed canonical execution approval and
opaque approved-source composition are implemented. The user approved two narrow
Task23 corrections: exact structured UTF-8 environment/host registry payloads and
the shared provider key-material verifier. There is no dual acceptance, migration
or repair of legacy digests; they fail closed until separately reviewed
rebootstrap. No rebootstrap or actual authorization was executed. The signed
execution actor is separate from quality-reviewer roles. Source revalidation
threads append's existing sealed barrier through projection validation; exact-six
schema, complete runtime images/types and Option-A observations remain intact.
Production CLI remains evaluator-first refusing with no signer or secret args.
Task25, production readers and PostgreSQL verification remain outstanding.

**Round-1 remediation (independent rereview pending).** Two P1 findings exposed
adapter-exit revocation and expired-guard reuse. Authority-owned private guard
lifetimes now bind owner/connection/provider/thread and the release transaction;
adapters borrow the mandatory guard and complete teardown before callback-free
source/key/target/peer validation. Independent corpus SQL generation/key/policy
checks fence the complete approved adapter snapshot. Ledger revalidation runs
before DML, before marker publication and before commit. Marker-first crash
evidence and rollback semantics remain unchanged; no implicit repair is allowed.
The report records permanent actual-append RED reproducers and final evidence.
Remediation implementation: `3f143763dacb9a6e44c6d319265fdbb0a24cfb71`.
Fresh full release selection: **1041 passed, 14 skipped** across three disjoint
shards of all 19 files; expanded direct impact: **266 passed, 16 skipped**
(11 existing Alembic warnings); standalone credential: **3 passed**. All 11
Python files pass Ruff/format/compile and diff checks. Code/tests were frozen
before the final shard launch. PostgreSQL and independent rereview remain open.

**Round-2 callback-isolation remediation (independent rereview pending).**
Actual append reproduced committed corpus drift via reviewer-result properties
and partial case/auth commits from post-SQL reader teardown. Finish every
adapter context, oracle, reviewer/user lookup, clock and result-property call
before mutation DML. Freeze the validated complete source/provider/reviewer and
approval context, then use only locally owned SQL/file/key/Git checks and pure
complete-image validation under the same active authority barrier. No source
reacquisition or injected provider revalidation is permitted after DML. Pin
complete verified provider file/DB images before publication; refuse append
publication hooks, and commit or rollback before trusted lock cleanup even on
exceptions. Preserve guard lifetime and marker-first crash evidence. The exact
same-connection reader contract applies only to pre-DML acquisition; no production
reader is composed. Implementation `656a5c3` passes the complete 19-file release
selection: **1070 passed, 14 skipped** across three disjoint shards; direct impact
**266 passed, 16 skipped** (11 existing Alembic warnings); credential **3 passed**.
All 12 Python files pass Ruff/format/compile and diff checks. The report records
permanent RED oracles and exact final commands/results. No code/tests changed
after final suite launch, no actual operation ran, and independent rereview is open.

**Round-3 publication fencing remediation (independent rereview pending).**
Rereview of `069c034` confirmed global provider-verification regression on
init/recovery/inspect and partial commits through non-runtime SQL bindings.
Pin verified provider files/whole DB images and complete release DB state in
every authority path; use callback-free prepublication/pre-DML/precommit checks,
plus a final inspect checkpoint. Do not restore post-commit verifier callbacks.
Keep the genuine marker-first init/recovery hook before release DML, followed by
transaction/peer/release revalidation; failure preserves crash evidence, not DB
publication or automatic repair. Append continues to reject publication hooks.
The smallest internal literal-image executor v1 consumes every plan before DML
and executes only newly constructed schema-owned SQL with complete frozen row
images. Reject executable bindings, expressions, custom processors, callable
defaults, deletion and direct internal/provider plans. Recheck the transaction
after materialization and retain pure complete-image after-SQL validation.
Preserve native driver `timezone`/`ZoneInfo` instant normalization while refusing
custom `tzinfo` without callbacks; initial claim UTC identity/fold stays strict.
Five permanent RED tests reproduced both P1s; the report records the expanded
matrix and final verification. Implementation `b1ab6af` passes all 19 release
files: **1145 passed, 14 skipped**; direct impact **266 passed, 16 skipped**,
11 existing Alembic warnings; focused contracts **38 passed**; credential
**3 passed**. Four Python files pass Ruff/format/compile and diff checks, with
code/tests frozen before final runs. Existing Task24 source/reviewer/guard semantics,
exact-six schema and Option-A remain unchanged. No actual operation ran and no
new execution authority, migration, evaluator or production reader is supplied.

**Round-4 frozen-input/roster remediation (independent rereview pending).**
Review of `069c034..4a96d36` confirmed retained caller path/identity callbacks,
root-generator exhaustion, extra valid mutation plans executing before roster
refusal, and Numeric coercion. Materialize root iterables once and detach each
path protocol into canonical native strings during construction. At every public
entry require exact native DB identity fields, copy before callbacks, compare
with the independently read current identity, and retain only owned images.
Before DML/incident validate the complete frozen plan roster, typed identities,
operations, semantic order/deltas, affected rows and prospective full DB roster.
Use the existing transition checks on locked before-images plus literal deltas,
including the prospective sealed incident; independently verify SQL afterward.
Acquire the approved source once. Caller Numeric fields require exact finite
nonnegative Decimal within schema precision/scale, without signed zero or excess
scale; schema-owned defaults normalize separately. Preserve existing six-place
signed images, child order, safe statement reordering, marker-first hooks,
non-append checkpoints, callback-free post-DML validation and active guards.
The report records permanent RED and final regression/commit evidence. This is
an implementation candidate, not independent CLEAN or execution approval.
Implementation `34b2475` passes all 19 release files (**1318 passed, 14 skipped**)
and direct impact (**266 passed, 16 skipped**, 11 existing Alembic warnings).
Credential **3 passed**; six-file Ruff/format/compile and diff checks pass.
Code/tests were frozen before final launch. All 30 conditional PostgreSQL skips
remain unexecuted; independent rereview, Task25 and production readers are open.

**Round-5 ownership remediation: IMPLEMENTED / NOT RELEASE-CLEAN (breaker 5/5).**
Before any append callback, validate/materialize the whole exact native transition
tree and detach all mutation plans, primary keys and observations. Retain private
scalars/containers only; custom types, protocols and shared/cyclic containers refuse
without invoking hooks. Keep SQL metadata private too: an append-local native
Table/Column/type registry supplies execution, snapshots, runtime projections and
final roster queries. Submitted MutationSets remain input-only. Detach/authenticate
sealed incident descriptors/envelopes; preserve retry after preflight refusal by
checking/consuming only the original exact one-use slot immediately before DML,
then discard that handle. Numeric WHERE bindings require exact finite unsigned
Decimal at column-owned precision and exact fixed scale before comparing the locked
before-image. Existing SET semantics and all 17 transition kinds remain in scope.
Permanent REDs cover payload and SQL-processor partial commits, incident envelope
publication and noncanonical predicates; generic identity/field and callback matrices
verify the owned boundary. Final verification/commits are recorded in the Task24-B
report. No Task25, real execution approval, migration or production reader is added;
PostgreSQL remains open; the final independent verdict below supersedes pending review.
Implementation `3617c0e` has final green per-file release evidence **1379 passed,
14 skipped**, direct impact **266 passed, 16 skipped** (11 existing warnings),
credential **3 passed**, and independent adapted probes **5 passed**. Ruff/format/
compile/diff checks pass on four Python files. Production remained frozen; six
obsolete test expectations were updated and their complete files/shard rerun,
as distinguished in the report. Final independent review of `3617c0e` / docs
`26d8337` reproduced one load-bearing P1: sealed provider-incident SQL in
`RagProviderSafetyService._apply_release_incident` / `_commit_transition` retains
shared ORM Table/Column/Type objects outside the append-local schema. The focused
probe observed partial provider-authority generation and latch publication while
readiness, provider history and release marker remained unchanged. The reviewer
claims only that focused failure, not a rerun of the implementation suites.
Task24-B and Task24 overall remain **NOT RELEASE-CLEAN**; actual release and
authorization remain blocked. Carry this finding into Task25's mandatory first
RED slice below; no open-ended sixth Task24-B fix. Cost if wrong: persistent
provider authority/readiness/history divergence and reviewed recovery requirement.

B implementation files additionally include `backend/app/rag/release_authority.py`
(the two approved registry alignments), `backend/app/rag/release_ledger.py`
(existing sealed-guard propagation), and test-only
`backend/tests/release_ledger_fixtures.py`. New TDD suites are
`test_rag_release_reviewer.py`, `test_rag_live_gate_authorization.py` and
`test_rag_release_identity_alignment.py`. The ignored SDD Task24-B report records
the confirmed RED/GREEN sequence and exact final verification commands.

Implementation commit: `da4da6d5ab3e1a1629e4c71cf271100ab94165ab`.
Fresh final release selection: **1025 passed, 14 skipped in 882.03s**;
direct provider/cost/runtime/input/credential impact: **259 passed, 13 skipped**
(11 existing Alembic warnings); standalone credential: **3 passed**. All eight
changed Python files passed Ruff/format/compile and diff checks. The 27
PostgreSQL skips remain unexecuted. Independent review is still required.

Implementation candidate `2460bdf` is pending independent review. Verification:
release regression `799 passed, 14 skipped`; final limit/source impact `97
passed`; final CLI `12 passed`; direct provider/cost/runtime/input/credential
impact `259 passed, 13 skipped` (11 existing Alembic warnings). Final credential
scan `3 passed`; changed-file Ruff/format, compileall and diff checks passed.

The spec's root alias is authoritative:
`authorization.manifest_hmac == fixture_manifest_hmac` over exact committed
fixture path/SHA/version. Supersede the historical carryover's inner-digest
equality with a verified source binding: the full resolved execution preimage
retains `source_manifest_hmac`, selects the exact declared ordinal/case, and uses
the separately locked approved runtime/provider/corpus/baseline inputs. Its own
digest is integrity evidence, never independent authority. Preserve every
complete-image/type/pre-mutation/provider check from the CLEAN carryover.

The fixture contains stable sanitized references. Resolve all mappings exactly;
never commit key-dependent placeholder HMACs. The conservative case envelopes
sum exactly to USD `0.360000`: keyword embedding/generation split
`0.000000/0.012000`, pgvector split `0.000160/0.011840`. Token maxima remain the
existing 8,000 query, 10,000 answer-input and 512 answer-output caps; reserves
must cover their frozen costs without precision loss. This does not change
runtime pricing or authorize spend.

The preview-source capability is provenance only. The approved execution-source
verifier is fail-closed pending B. No real snapshot reader is composed in A;
successful tests use explicit locked fake snapshot readers and isolated Git
repositories. Missing committed `backend/app/rag/release_quality.py` (Task25)
must produce `evaluator_unavailable` with zero mutations/dispatches before any
real authority construction. Do not add a placeholder evaluator or OAuth flow.

**Task24-A round-1 remediation:** the locked reader's typed hard-negative
oracle must derive visible non-entailing candidates from frozen corpus/scope/
query inputs and bind the exact source/case/query/scope/corpus/backend request,
oracle-definition identity and candidate results in canonical preview. Missing
adapter, no-match and hidden-only cases refuse before provenance issuance;
fixture labels and caller booleans are insufficient. The declarative fixture
therefore permits evidence slots/support modes for negatives while its relevant
and required answer-support sets remain empty. Prior-context quotas use the
effective prepared query, not message presence. Recheck source/clean Git after
the final reader/oracle read and after lock exit, immediately before issuing
provenance. Permit null vector state only outside the reader's frozen pgvector
baseline participation roster; relevant/required/oracle pgvector members must
belong to that roster and have vector state. This adds no production adapter,
reviewer proof, authorization flow or provider dispatch.

Round-1 candidate `b2f247e` is pending independent re-review. Verification:
complete release/carryover selection `856 passed, 14 skipped`; preview impact
`131 passed`; final R1 probes `46 passed`; direct impact `259 passed, 13 skipped`
(11 existing Alembic warnings); credential `3 passed`; Ruff/format/compile/diff
clean. PostgreSQL DSN remains absent; Task24 overall remains incomplete.

**Task24-A round-2 remediation:** retain an unexposed immutable scalar snapshot
of the oracle request before invoking the reader. Validate both exposed and
returned request values against it, and serialize only independently retained
expected values. An adapter cannot change the manifest/query binding by
mutating a shared frozen-dataclass instance. Preserve zero-call, zero-DML and
zero-provenance-issuance refusal for aliased/copied mismatch, mutable-container
substitution, adapter failure and restored-input/tampered-copy cases. An exact
restored binding must produce identical canonical bytes/HMAC. No new authority,
request mutation tracker, evaluator or production adapter is added.

Round-2 implementation `a3444d8` received independent **CLEAN** rereview over
`e3efbc9..ba229aa`, with no actionable findings. Confirmed implementation RED:
`12 failed, 26 passed`; final alias probes `44 passed`; complete release/A/CLI
regression `900 passed, 14 skipped`; direct impact `259 passed, 13 skipped`
(11 existing Alembic warnings); credential `3 passed`; Ruff/format/compile/diff
clean. Independent reviewer evidence: combined selection `189 passed in 208.84s`,
credential `3 passed`, eight additional refusal probes, and clean
Ruff/format/AST/diff/status. This supersedes earlier Task24-A pending-review
statuses. All four round-1 fixes, the single fixture-manifest root and default
Task24-B refusal remain intact. PostgreSQL verification, the Task25 evaluator,
production snapshot/oracle/roster adapters and Task24-B remain incomplete;
Task24 is not complete. Actual CLI preview returns `evaluator_unavailable` with
zero dispatch and no authorization. Review closure is docs-only, not release
approval or permission for live execution.

**Interfaces:**

```text
fixture version: rag-live-quality-30:v1
distribution: ask_keyword=10, ask_pgvector=5, assistant_keyword=10, assistant_pgvector=5
assistant context: keyword prior/no-prior=5/5, pgvector prior/no-prior=3/2
```

```text
uv run python -m backend.app.admin.rag_live_gate preview
uv run python -m backend.app.admin.rag_live_gate authorization-bootstrap
```

```python
ReviewerRole = Literal['reviewer_a', 'reviewer_b', 'adjudicator_c']
LiveCaseKind = Literal['positive', 'hard_negative']
LiveCaseSurface = Literal['ask', 'assistant']

@dataclass(frozen=True, slots=True)
class ReviewerLoginChallenge:
    role: ReviewerRole
    authorization_url: SecretStr  # contains signed state; never log or persist
    redirect_uri: str
    challenge_hmac: str
    signed_state_hmac: str
    pkce_challenge_hmac: str
    issued_at_utc: datetime
    expires_at_utc: datetime

@dataclass(frozen=True, slots=True)
class AuthenticatedReviewerSubject:
    role: ReviewerRole
    auth_user_id: int
    issuer: Literal['https://accounts.google.com']
    reviewer_subject_hmac: str
    challenge_hmac: str
    authenticated_at_utc: datetime

@dataclass(frozen=True, slots=True)
class FrozenCorpusMember:
    ordinal: int
    serving_identity_hmac: str
    serving_version_fingerprint: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    effective_permission: Literal['public', 'internal', 'restricted']
    support_mode: SupportMode
    vector_index_state_hmac: str | None

@dataclass(frozen=True, slots=True)
class FrozenCorpusSnapshot:
    corpus_generation: int
    vector_index_generation: int
    embedding_model_bytes: bytes
    index_policy_version_bytes: bytes
    pgvector_cosine_policy_version: Literal['pgvector-cosine-indexable:v1']
    members: tuple[FrozenCorpusMember, ...]
    corpus_snapshot_hmac: str

@dataclass(frozen=True, slots=True)
class LiveGateLimits:
    case_claims: Literal[30]
    answer_generation_dispatches: Literal[30]
    query_embedding_dispatches: Literal[10]
    total_dispatches: Literal[40]
    case_max_cost_usd: Decimal  # exact 0.012000
    total_max_cost_usd: Decimal  # exact 0.360000

@dataclass(frozen=True, slots=True)
class FrozenLiveManifestCase:
    ordinal: int  # exact contiguous 0..29
    case_id_hmac: str
    case_kind: LiveCaseKind
    surface: LiveCaseSurface
    configured_backend: Literal['keyword', 'pgvector']
    question_fixture_id: str
    security_scope_fixture_id: str
    prior_context_fixture_id: str | None
    query_bytes_hmac: str
    expected_no_answer: bool
    allowed_support_modes: tuple[SupportMode, ...]
    allowed_slot_ids: tuple[EvidenceSlotId, ...]
    relevant_serving_identity_hmacs: tuple[str, ...]
    required_serving_identity_hmacs: tuple[str, ...]
    query_embedding_required: bool
    answer_generation_required: Literal[True]
    query_embedding_reserved_cost_usd: Decimal
    answer_generation_reserved_cost_usd: Decimal
    case_total_reserved_cost_usd: Decimal

@dataclass(frozen=True, slots=True)
class FrozenLiveManifestSnapshot:
    live_gate_contract_version: Literal['rag-live-gate:v1']
    fixture_manifest_version: Literal['rag-live-quality-30:v1']
    fixture_manifest_path: Literal['backend/tests/fixtures/rag_v2_live_gate_30.json']
    fixture_manifest_sha256: str
    fixture_manifest_hmac: str
    manifest_hmac: str  # validated exact alias of fixture_manifest_hmac
    clean_git_commit: str
    rubric_version: Literal['rag-live-quality-rubric:v1']
    cases: tuple[FrozenLiveManifestCase, ...]
    ask_keyword_count: Literal[10]
    ask_pgvector_count: Literal[5]
    assistant_keyword_count: Literal[10]
    assistant_pgvector_count: Literal[5]
    keyword_with_prior_context_count: Literal[5]
    keyword_without_prior_context_count: Literal[5]
    pgvector_with_prior_context_count: Literal[3]
    pgvector_without_prior_context_count: Literal[2]
    limits: LiveGateLimits

@dataclass(frozen=True, slots=True)
class AuthorizedRagLiveGate:
    live_gate_contract_version: Literal['rag-live-gate:v1']
    authorization_state: Literal['unused']
    ledger_uuid: UUID
    ledger_epoch: int
    approval_base_generation: int
    approval_base_release_marker_file_digest: str
    approval_id_hmac: str
    approval_hmac: str
    manifest: FrozenLiveManifestSnapshot
    corpus: FrozenCorpusSnapshot
    baseline_hmac: str
    approved_provider_safety_snapshot_hmac: str
    reviewer_roster_hmac: str
    validation_database_identity_hmac: str
    designated_environment_id_hmac: str
    designated_host_id_hmac: str
    implementation_plan_reference_hmac: str
    limits: LiveGateLimits

class FreshGoogleReviewerVerifier:
    def begin(
        self,
        *,
        role: ReviewerRole,
    ) -> ReviewerLoginChallenge: ...
    def complete(
        self,
        *,
        role: ReviewerRole,
        code: SecretStr,
        signed_state: SecretStr,
        expected_challenge_hmac: str,
    ) -> AuthenticatedReviewerSubject: ...
```

- [x] Add RED manifest tests for exact 30 unique case IDs, fixed surface/backend distribution, positive/hard-negative labels, sanitized fixture references, expected component presence/reserve split, relevant/required serving HMACs, support modes/slot allowlist, and case ceiling sum.
- [x] Add RED baseline/rubric tests for `rag-live-quality-rubric:v1`, provider-free legacy retrieval definition HMAC, evaluator/source/fixture path bytes and committed SHA, exact Git commit, frozen corpus snapshot, validation DB identity, provider-safety snapshot, release epoch, and implementation-plan reference.
- [x] Add RED preview tests computing exact 30/10/40 maximum dispatches, USD `0.012000` per case and USD `0.360000` aggregate reserve, with provider transport call count zero and no authorization/case/dispatch mutation.
- [x] Add RED reviewer tests for three pairwise-distinct subjects bound to roles `reviewer_a`, `reviewer_b`, `adjudicator_c`. For each role the verifier starts a fresh one-use Google authorization-code + PKCE challenge using the existing signed state/nonce builder, a release-specific loopback redirect, fixed Google token/userinfo endpoints and configured client ID; `complete` validates exact state/challenge/role/redirect, exchanges the no-echo code once, requires a nonblank immutable Google `sub`, and matches it to the selected current `AuthUser.external_id`. This is deliberately a fresh Google OAuth2+userinfo proof, not an ID-token OIDC flow: signed-state nonce must never be described or tested as an `id_token` nonce/JWKS/issuer claim. Existing ParaWorks session cookies are insufficient because they contain only the internal user ID/expiry. Tests inject a fake Google client and make no network call. Never put code/token/state in CLI args, env, DB, logs, or report; after verification keep only role-bound subject HMACs and reject duplicate/role swap/roster mutation. Production CLI composition remains unavailable as recorded above.
- [x] Add RED authorization tests binding single-use user confirmation to the whole preview HMAC and exact unused ledger/provider/corpus/fixture/commit/reviewer snapshot. Any change, key rotation, safety transition, or stale epoch is zero-call refusal.
- [x] Confirm RED for reviewer, authorization, source lifecycle, real-authority identity/peer and sealed-barrier integration suites; see the Task24-B report for exact expected failures.
- [x] Implement provider-free preview and verifier-only authorization construction. The `authorization-bootstrap` command surface fails closed until production readiness; construction tests use externally signed fake approval records and never treat plan approval as execution approval.
- [x] Rerun all Task23/24 release suites, direct-impact tests and eight changed-file Ruff/format/compile checks; the final counts are recorded above.
- [x] Commit with `git commit -m "feat: authorize bounded rag live gate"` (`da4da6d`).
- [ ] Independent Task24-B review, PostgreSQL verification and production reader/Task25 readiness; Task24 overall remains incomplete.

### Task 25: Add Composite Runner and Provider-Free Quality Adjudication

**Mandatory first RED slice — Task24-B breaker 5/5 carry-forward.** Before
composite runner or quality work, reproduce the sealed provider-incident shared
ORM metadata partial-publication P1 from final review of `3617c0e` / `26d8337`.
Isolate all provider-incident SQL, including
`RagProviderSafetyService._apply_release_incident` / `_commit_transition`, into
authority-owned immutable schema/type metadata. Enforce no injected callbacks
after first DML and prove atomic provider authority + exact readiness peers +
provider history + latch + release state/marker with independent database/file
oracles. Obtain an independent **CLEAN** review of this slice before proceeding.
This is not a sixth Task24-B fix round and grants no release/authorization authority;
Task24-B and Task24 overall remain NOT RELEASE-CLEAN. Passing this slice alone
does not satisfy the other open release gates.

**Implementation candidate (2026-09-14).** Commit `e94bd96` implements only this
mandatory prerequisite. Fresh private SQL metadata owns all provider-incident
writes and post-write reads; complete before/prospective/after authority,
exact-two readiness and provider-history images are checked at safe transaction
boundaries, and no caller-overridable callback runs after DML. Frozen focused
evidence is `150 passed`; broader release/provider evidence is `1010 passed,
14 skipped`; final affected evidence is `253 passed, 14 skipped`; credential is
`3 passed`. PostgreSQL is unavailable and independent CLEAN review is pending.
Do not proceed to the composite runner/evaluator or authorize a release.

**Independent-review round 1 correction (2026-09-14).** Implementation
`cb6b99f` closes the review's eight saved regressions: five post-DML provider
service dispatches and three malformed historical chains. Capability preparation
now HMAC-binds the complete physical provider image. The authority-owned,
non-virtual commit operation returns immutable latch and complete
before/prospective/actual provider images; post-DML publication uses those images
and callback-free private reads. History validation preserves physical identity
order and enforces bootstrap, block, reviewed reset, rebind and supersession state
matrices, with reverse digest authentication wherever the v1 rows retain enough
material. Release inspection refuses a provider-only incident left beside a
`started` authorization for the predecessor digest. Fresh frozen evidence is
`160 passed` focused, `8 passed` saved probes, `1594 passed, 15 skipped`
comprehensive release/provider, `221 passed, 14 skipped` direct impact and
`3 passed` credential hygiene. PostgreSQL and independent CLEAN rereview remain
open. Do not start the composite runner/evaluator or execute a release.

**Independent-review round 2 correction (2026-09-14).** Implementation
`69b4a00` closes the saved drift-abort and historical-attribution findings.
Only the owned append path can carry an exact allowlisted binding from one
predecessor provider digest to the current digest for a validated
`started -> aborted_provider_safety` authorization; ordinary inspection remains
fail-stop for that mixed state. Preparation authenticates historical blocker
actor/run attribution, signed first-blocker run/category/time, and every
reconstructible digest through supersession to bootstrap. Because v1 rebind
overwrote predecessor policy material, incident preparation for those histories
fails closed pending reviewed migration/rebootstrap; no migration is part of this
slice. Fresh frozen evidence is `182 passed` focused, `1258 passed, 15 skipped`
broad release/provider, `221 passed, 14 skipped` direct impact and `3 passed`
credential hygiene. PostgreSQL and independent CLEAN rereview remain open. Do not
start the composite runner/evaluator or execute a release.

**Independent-review round 3 correction (2026-09-14).** Implementation
`64a414b` closes the remaining false refusal for the existing case-null
`authorization_abort_corpus_drift` and `authorization_abort_execution_crash`
contracts after unrelated valid provider drift. The private binding now carries
the exact kind, terminal state/outcome, authorization identity and immutable
predecessor/current provider digests, and is derived only after all existing
kind-specific mutation, corpus/attestation, roster and provider validations.
Ordinary inspection remains fail-stop; invalid authority, wrong digest/kind and
replay refuse before DML. Fresh frozen evidence is focused `195 passed`, affected
release state/schema `80 passed`, broad release/provider `1268 passed, 15 skipped`,
direct impact `221 passed, 14 skipped`, and credential `3 passed`. PostgreSQL,
legacy-v1-rebind recovery and independent CLEAN Round-3 review remained open at
candidate handoff. Do not execute a release.

**Independent Round-3 CLEAN closeout (2026-09-14).** Independent review of
`efa72c0` found no actionable P1/P2 issue and closes every historical finding for
this prerequisite's local code-correctness slice. Fresh reviewer evidence is
`334 passed, 15 skipped`; static/range checks are green and tracked files were
unchanged. Legacy v1 rebind remains fail-closed pending separately reviewed
migration/rebootstrap, while PostgreSQL and live-release gates remain open.
Composite runner/evaluator work may now begin, but this CLEAN prerequisite does
not authorize release execution.

**Task25-A evaluator slice implemented (2026-09-14).** Commit `5f0e605`
implements only `backend/app/rag/release_quality.py` and its provider-free test
matrix. The immutable evaluator enforces the exact 30-case roster/order,
baseline/manifest/corpus/approval binding, signed A -> B -> adjudicator-only-on-
disagreement review order, complete/pairwise-distinct reviewer subjects, exact
signature HMACs, hard-negative and positive/required-slot coverage, at least 95%
faithfulness, V2 retrieval parity with the frozen legacy baseline, and zero
leaks/invalid slots. The constructor takes the exact immutable role-to-subject
map so the approved `evaluate(...)` signature remains unchanged while roster and
label signatures are recomputed independently. Raw review answer/evidence stays
ephemeral and is absent from sanitized rows and reports. Frozen evidence is `44
passed` for evaluator/credential, `43 passed` for exact authorization-refusal plus
evaluator nodes, and `408 passed` for the adjacent release matrix; static checks
are green. No runner, production reader, DB schema/persistence, CLI execution,
provider/network/paid call or release is part of this slice.

Task25-B remains actual implementation. Its mandatory first RED prerequisite is
the clean-Windows committed-source binding defect exposed by this slice:
`core.autocrlf=true` makes raw worktree bytes differ from committed blobs for
`service.py`, `pgvector_store.py` and `search_store.py`, so the current checker
returns `committed_source_changed` on a clean checkout. Fix this cross-platform
without weakening exact committed-byte baseline binding, Git-clean enforcement,
zero provider/authorization behavior or fail-closed refusal; then implement the
production readers, composite runner/live gate, schema and CLI integration.

**Task25-A independent-review Round 1 correction (2026-09-14).** Both initial
reviews of `ab10514` were NOT CLEAN. Commit `2686306` reuses the Task24 corpus
canonical validator, recomputes the snapshot HMAC and binds manifest identities
to the valid non-empty member roster under the approved
`pgvector-cosine-indexable:v1` policy. It narrows completed quality input to the
six approved product outcomes, validates exact Decimal/cost/dispatch bounds,
freezes and revalidates reviewer authority, and totalizes malformed nested input
as bounded `RagReleaseQualityError` refusals. RED evidence is saved probes `4
failed`, permanent corpus/adversarial suite `52 failed, 19 passed`, and frozen-
attribute test `1 failed`. GREEN is permanent evaluator `87 passed`, evaluator/
saved probes/credential `94 passed`, Task24 corpus/preview/reviewer `202 passed`,
and clean-commit adjacent release `454 passed in 1572.25s`; static checks are
green. No provider/network/paid call, persistence, runner/CLI or release ran.
Fresh dual independent review of `2686306` is mandatory before this slice is
CLEAN or Task25-B may begin.

**Task25-A independent-review Round 2 correction (2026-09-14).** Fresh reviews
of Round-1 HEAD `6d7353a` were NOT CLEAN. Implementation `5affa36` reuses the
Task24 `_manifest_payload` validator for the complete executable
`FrozenCaseClaimManifest`, requires the exact 30 resolved cases and order,
binds case/surface/backend plus `query_bytes_hmac == retrieval_query_hmac`,
validates provider/runtime/config/input identities, and binds the exact two
component reserves to the annotation. Decimal values now require exact native,
finite, non-negative, unsigned exponent `-6` representation. Every sanitized
result leaf is type-checked before comparison/HMAC/canonicalization. Reviewer
authority refuses deletion and bounds forced missing/malformed frozen rosters;
baseline ratio corruption reports `baseline_drift`.

RED is saved probes `11 failed, 5 passed`, permanent Round-2 matrix `26 failed,
92 passed`, and operation-order regression `2 failed, 11 passed`. GREEN is
evaluator `147 passed`, evaluator/two saved probes/credential `170 passed`,
Task24 preview/corpus/reviewer `202 passed`, and clean-commit adjacent release
`514 passed in 1574.72s`; static checks are green. No external/provider/paid
call, persistence, runner/CLI or release occurred. Fresh dual independent review
of `5affa36` is mandatory; Task25-B remains blocked and unstarted.

- [x] Complete the mandatory provider-incident prerequisite and receive
  independent local-code **CLEAN** review; PostgreSQL/live-release and legacy v1
  rebind recovery remain separate open gates.

**Files:**

- Create: `backend/app/rag/release_quality.py`
- Create: `backend/app/rag/live_gate.py`
- Create: `backend/tests/test_rag_release_quality.py`
- Create: `backend/tests/test_rag_live_gate.py`
- Create: `backend/tests/test_rag_live_gate_postgres.py`
- Modify: `backend/app/admin/rag_live_gate.py`

**Interfaces:**

```python
ReviewLabel = Literal['entailed', 'not_entailed', 'ambiguous']

LiveCaseOutcome = Literal[
    'supported', 'no_match', 'hidden_only', 'safety_filter_empty',
    'insufficient_evidence', 'evidence_unavailable', 'budget_exceeded',
    'retriever_unavailable', 'model_unavailable', 'model_provider_failed',
    'structured_output_invalid', 'citation_validation_failed',
    'persistence_failed', 'unexpected_internal_error',
    'provider_safety_unavailable', 'provider_usage_overrun',
    'provider_response_identity_invalid', 'provider_embedding_payload_invalid',
    'live_corpus_snapshot_changed', 'abandoned_unknown',
]

LiveAuthorizationTerminalState = Literal[
    'complete', 'finished_failed', 'aborted_corpus_drift',
    'aborted_execution_crash', 'aborted_overrun', 'aborted_provider_safety',
]

LiveAuthorizationTerminalOutcome = Literal[
    'quality_gate_green', 'quality_gate_failed', 'ordinary_execution_failed',
    'execution_contract_failed', 'live_corpus_snapshot_changed',
    'abandoned_unknown', 'provider_usage_overrun',
    'provider_response_identity_invalid', 'provider_embedding_payload_invalid',
    'provider_safety_unavailable',
]

@dataclass(frozen=True, slots=True)
class CompositeComponentClaim:
    authorization: AuthorizedRagLiveGate
    case: FrozenLiveManifestCase
    component: RagPaidComponent
    runtime_agent_run_id: int
    prepared_budget: PreparedPaidCallBudget
    prepared_input_hmac: str
    retrieval_query_hmac: str
    expected_release_generation: int
    current_corpus_snapshot_hmac: str
    current_provider_safety_snapshot_hmac: str
    process_instance_hmac: str
    execution_runner_fence_hmac: str

class CompositePaidCallPermit(ProviderDispatchPermit, Protocol):
    @property
    def approval_id_hmac(self) -> str: ...
    @property
    def case_id_hmac(self) -> str: ...
    @property
    def component(self) -> RagPaidComponent: ...
    @property
    def runtime_agent_run_id(self) -> int: ...
    @property
    def release_to_generation(self) -> int: ...
    @property
    def dispatch_fence_hmac(self) -> str: ...
    @property
    def provider_safety_snapshot_hmac(self) -> str: ...
    @property
    def reserved_cost_usd(self) -> Decimal: ...
    def consume_at_dispatch(self) -> None: ...

@dataclass(frozen=True, slots=True)
class SanitizedLiveBlockResult:
    block_ordinal: int
    block_result_hmac: str
    evidence_projection_hmac: str

@dataclass(frozen=True, slots=True)
class SanitizedLiveCaseResult:
    ordinal: int
    case_id_hmac: str
    case_kind: LiveCaseKind
    surface: LiveCaseSurface
    configured_backend: Literal['keyword', 'pgvector']
    state: Literal['complete', 'failed']
    outcome: LiveCaseOutcome
    runtime_agent_run_id_hmac: str
    case_projection_hmac: str | None
    current_corpus_snapshot_hmac: str
    provider_safety_snapshot_hmac: str
    query_embedding_dispatch_count: Literal[0, 1]
    answer_generation_dispatch_count: Literal[0, 1]
    reserved_cost_usd: Decimal
    charged_cost_usd: Decimal
    legacy_precision_numerator: int
    legacy_precision_denominator: int
    legacy_recall_numerator: int
    legacy_recall_denominator: int
    v2_precision_numerator: int
    v2_precision_denominator: int
    v2_recall_numerator: int
    v2_recall_denominator: int
    expected_no_answer: bool
    hard_negative_correct: bool
    positive_answered: bool
    required_slot_covered: bool
    blocks: tuple[SanitizedLiveBlockResult, ...]

@dataclass(frozen=True, slots=True)
class FrozenLegacyBaselineMetrics:
    baseline_hmac: str
    baseline_policy_snapshot_hmac: str
    evaluator_code_hmac: str
    retriever_source_bundle_hmac: str
    evaluator_source_commit: str
    fixture_manifest_hmac: str
    corpus_snapshot_hmac: str
    precision_numerator: int
    precision_denominator: int
    recall_numerator: int
    recall_denominator: int

@dataclass(frozen=True, slots=True)
class QualityBlockAdjudication:
    block_ordinal: int
    block_result_hmac: str
    evidence_projection_hmac: str
    reviewer_a_label: ReviewLabel
    reviewer_a_signature_hmac: str
    reviewer_b_label: ReviewLabel
    reviewer_b_signature_hmac: str
    adjudicator_label: ReviewLabel | None
    adjudicator_signature_hmac: str | None
    decided_label: ReviewLabel

@dataclass(frozen=True, slots=True)
class QualityCaseAdjudication:
    case_id_hmac: str
    case_projection_hmac: str
    case_kind: LiveCaseKind
    required_slot_covered: bool
    block_labels: tuple[QualityBlockAdjudication, ...]

QualityFailureReason = Literal[
    'hard_negative_accuracy', 'positive_answer_coverage', 'faithfulness',
    'retrieval_precision', 'retrieval_recall',
]

@dataclass(frozen=True, slots=True)
class RagQualityReport:
    approval_hmac: str
    approval_id_hmac: str
    approved_corpus_snapshot_hmac: str
    current_corpus_snapshot_hmac: str
    baseline_hmac: str
    manifest_hmac: str
    reviewer_roster_hmac: str
    rubric_version: Literal['rag-live-quality-rubric:v1']
    case_adjudications: tuple[QualityCaseAdjudication, ...]
    case_count: Literal[30]
    failure_reasons: tuple[QualityFailureReason, ...]
    gate_outcome: Literal['green', 'quality_gate_failed']
    faithfulness_entailed_blocks: int
    faithfulness_total_blocks: int
    hard_negative_case_count: int
    hard_negative_correct_count: int
    positive_case_count: int
    positive_answered_count: int
    legacy_precision_numerator: int
    legacy_precision_denominator: int
    legacy_recall_numerator: int
    legacy_recall_denominator: int
    v2_precision_numerator: int
    v2_precision_denominator: int
    v2_recall_numerator: int
    v2_recall_denominator: int
    payload_canonical_bytes: bytes
    quality_report_hmac: str

@dataclass(frozen=True, slots=True)
class RagLiveGateResult:
    ledger_uuid: UUID
    ledger_epoch: int
    approval_id_hmac: str
    approval_hmac: str
    terminal_authorization_state: LiveAuthorizationTerminalState
    terminal_outcome: LiveAuthorizationTerminalOutcome
    release_generation: int
    last_transition_digest: str
    terminal_cases: tuple[SanitizedLiveCaseResult, ...]
    case_claim_count: int
    query_embedding_dispatch_count: int
    answer_generation_dispatch_count: int
    total_dispatch_count: int
    provider_attempt_count: int
    committed_reserved_cost_usd: Decimal
    charged_cost_usd: Decimal
    quality_report: RagQualityReport | None

@dataclass(frozen=True, slots=True)
class EphemeralReviewBlock:
    case_id_hmac: str
    block_ordinal: int
    block_result_hmac: str
    answer_text: str
    evidence_projection: V1EvidenceProjection
    evidence_projection_hmac: str

@dataclass(frozen=True, slots=True)
class SignedReviewLabel:
    case_id_hmac: str
    block_ordinal: int
    reviewer_role: ReviewerRole
    reviewer_subject_hmac: str
    label: ReviewLabel
    signature_hmac: str

class AuthenticatedLiveReviewSession:
    def review(self, block: EphemeralReviewBlock) -> SignedReviewLabel: ...

class CompositePaidCallAdmissionCoordinator:
    def claim_component(self, connection: Connection, request: CompositeComponentClaim) -> CompositePaidCallPermit: ...

class RagLiveGateRunner:
    def run(self, authorization: AuthorizedRagLiveGate) -> RagLiveGateResult: ...

class RagReleaseQualityEvaluator:
    def evaluate(
        self,
        *,
        terminal_cases: Sequence[SanitizedLiveCaseResult],
        signed_labels: Sequence[SignedReviewLabel],
        baseline_metrics: FrozenLegacyBaselineMetrics,
        manifest: FrozenLiveManifestSnapshot,
        corpus: FrozenCorpusSnapshot,
        approval: AuthorizedRagLiveGate,
    ) -> RagQualityReport: ...
```

```text
uv run python -m backend.app.admin.rag_live_gate run
uv run python -m backend.app.admin.rag_live_gate authorization-abort-execution-crash
```

- [ ] Add RED fake-provider happy-path tests for exactly 30 terminal cases, 30 generation, 10 embedding, 40 total dispatches, zero retries, runtime/release charged equality after every transition, one execution-process/session lock, and one insert-only quality report.
- [ ] Add RED bound tests blocking case/generation 31, embedding 11, dispatch 41, duplicate claim, second runner, distinct concurrent case, reviewer takeover, and same authorization resume/rerun.
- [ ] Add RED one-physical-connection tests: release/runtime/safety/case/AgentRun claims share the exact validation PostgreSQL connection/transaction; separate engine/DSN or database identity mismatch performs zero marker mutation/call.
- [ ] Add RED crash/failure matrix for pre-dispatch local refusal, response-less embedding/generation failure, claim-before-call, call-before-outcome, component/finalization gap, post-30 pre-report/scoring, process/supervisor/fence mismatch, corpus drift at every barrier, safety/control change, overrun, malformed provider metadata, and marker/DB divergence. No automatic resume or missing downstream call.
- [ ] Add RED transition precedence: safety envelope block persists before release abort for overrun/remediation; corpus/safety abort outranks ordinary finished-failed; all 30 terminal ordinary failures may finish failed with lower dispatch counts.
- [x] Add RED quality tests with no LLM judge: hard-negative accuracy 100%, positive coverage 100%, required-slot coverage, claim-to-evidence faithfulness >=95%, retrieval precision/recall not below legacy, zero leaks/invalid slots, exact reviewer adjudication order/signature, and red report on any failed metric. The evaluator rejects missing/mismatched signed labels, baseline/manifest/corpus/approval drift, and any attempt to infer faithfulness from sanitized terminal rows alone. This closes Task25-A only; runner, PostgreSQL and combined gate checks remain open.
- [ ] Run `uv run pytest backend/tests/test_rag_release_quality.py backend/tests/test_rag_live_gate.py backend/tests/test_rag_live_gate_postgres.py -q` using fake transports only and confirm RED.
- [ ] Implement the live-only composite one-use permit, runner, and authenticated review session. Raw answer/evidence review blocks exist in process memory only, are shown only to the authenticated role in rubric order, and are destroyed after signing; DB/report rows keep HMACs/labels only. The CLI `run` is the only **release-harness/live-gate** paid path; production V2 enforce remains possible only through the ordinary runtime cost claim plus provider-safety admission. Unit/integration tests inject fake dispatch/reviewer boundaries and assert public network is unavailable.
- [ ] Rerun focused tests and `uv run ruff check backend/app/rag/release_quality.py backend/app/rag/live_gate.py backend/app/admin/rag_live_gate.py backend/tests/test_rag_release_quality.py backend/tests/test_rag_live_gate.py backend/tests/test_rag_live_gate_postgres.py`.
- [ ] Commit with `git commit -m "feat: add auditable rag live quality gate"`.

### Task 26: Extend the Existing Pytest Release Matrix for D Core

**Files:**

- Modify: `scripts/backend_release_matrix.py`
- Modify: `backend/tests/release_contracts.py`
- Modify: `backend/tests/test_release_contracts.py`
- Modify: `backend/tests/test_backend_release_matrix.py`

- [ ] Add a RED release-contract test that preserves the existing `settings-diagnostic`, `postgres`, `compatibility`, `non-slack`, and `full` profiles and adds `rag-v2-provider-free`; no profile may invoke `rag_live_gate run`, live embeddings/LLM/connectors, or silently hide the exact ten deferred Slack failures.
- [ ] Add RED manifest tests requiring `rag-v2-provider-free` to contain explicit pytest module/node selectors covering every backend test owned by Tasks 1–25, including `backend/tests/test_rag_v2_provider_free_golden.py` with a collected golden case count of at least 60. Extend the existing `postgres` profile with the exact D migration/pgvector/provider-safety/live-runner fake-transport modules rather than replacing its current C.5 children.
- [ ] Run `uv run pytest backend/tests/test_release_contracts.py backend/tests/test_backend_release_matrix.py -q` and confirm RED.
- [ ] Implement only pytest `ReleaseChild` entries because the current controller's evidence sidecars and coverage reconciliation are pytest-specific. Keep Ruff, frontend, secret, and clean-tree commands outside the matrix as explicit Task 27 gates; do not introduce a weak generic subprocess child merely to make the matrix run them.
- [ ] Rerun `uv run pytest backend/tests/test_release_contracts.py backend/tests/test_backend_release_matrix.py -q`, then `uv run ruff check scripts/backend_release_matrix.py backend/tests/release_contracts.py backend/tests/test_release_contracts.py backend/tests/test_backend_release_matrix.py`.
- [ ] Run `uv run --no-cache --locked python scripts/backend_release_matrix.py --profile rag-v2-provider-free`, the augmented `--profile postgres` against its disposable PostgreSQL+pgvector database, and `--profile non-slack`; require zero unexpected skips/xfails/errors, no leaked leases/residue, and exactly the known ten Slack failures only in the documented baseline.
- [ ] Run `git diff --check`, stage only these four release-matrix files, and commit with `git commit -m "test: add rag v2 provider-free release matrix"`.

### Task 27: Record Exact Provider-Free Evidence and Stop Before Paid Execution

**Files:**

- Modify: `plan.md`
- Modify: `docs/portfolio-log.md`
- Modify: `docs/superpowers/runbooks/session-handoff.md`

- [ ] Start from the exact clean Task 26 commit. Run `git status --porcelain=v1` and require empty output before collecting evidence.
- [ ] Run `uv run --no-cache --locked python scripts/backend_release_matrix.py --profile rag-v2-provider-free` and require the manifest-enforced explicit selectors, provider-free golden count `>=60`, migration/head checks, SQLite smoke, release-schema isolation, and all focused D Core suites green.
- [ ] Run `uv run --no-cache --locked ruff check backend/app backend/tests scripts`.
- [ ] Run `uv run --no-cache --locked python scripts/backend_release_matrix.py --profile postgres` against a disposable PostgreSQL+pgvector database and require zero residue.
- [ ] Run `uv run --no-cache --locked python scripts/backend_release_matrix.py --profile non-slack`; require all non-Slack tests green and exactly the documented ten deferred Slack failures visible, with no new skip/deselect/xfail.
- [ ] Run `uv run --no-cache --locked pytest backend/tests/test_secret_hygiene.py -q`; its scanner may report only path/line/detector kind and must not echo suspected bytes.
- [ ] Run frontend gates: `Push-Location frontend; npx tsc --noEmit; npm run lint; npm run build; $env:PLAYWRIGHT_MANAGED_SERVER='1'; npm run test:visual -- e2e/rag-v2-assistant.spec.ts e2e/assistant-memory.spec.ts e2e/visual-smoke.spec.ts --project=chromium-desktop; npm run test:visual -- e2e/rag-v2-assistant.spec.ts --project=chromium-mobile; Pop-Location`.
- [ ] Update the three project/handoff documents with implemented architecture, exact commands/counts, migration heads, rollout still disabled, paid call count zero, and remaining live authorization boundary.
- [ ] Run `git diff --check`, `git status --short`, and `git diff --name-only --cached`; stage only the three documentation files and commit with `git commit -m "docs: record deliverable d implementation evidence"`.
- [ ] On that exact clean documentation commit, rerun the three backend release profiles, Ruff, secret hygiene, and the full frontend command above. Require `git status --porcelain=v1` empty afterward so the final evidence and preview bind the same commit.
- [ ] Run the provider-free `status` and `preview` commands only. Verify fixture SHA, commit SHA, validation DB, provider/release authorities, reviewer roster, corpus snapshot, 30/10/40 limits, and USD `0.360000`; verify provider dispatch count remains zero.
- [ ] Stop and report the exact zero-call preview. Do not run `authorization-bootstrap` or `run` until the user separately confirms this exact tuple. The available USD 100 balance does not enlarge the first gate or authorize rerun/recovery/production/reindex/D.1/E spend.

---

## Approved Spec Coverage Map

| Design spec section | Implementation tasks | Executable proof |
|---|---|---|
| 1: decision summary | 1, 14, 21, 27 | frozen identities, real library boundaries, staged rollout and final stop rule |
| 2: problem and current gap | 1, 6, 7, 14–20 | hard-coded route removal, Runnable/StateGraph inspection and V1 UX regression tests |
| 3: scope | 1, 21, 27 | explicit D Core inclusions/exclusions and provider-free release assertions |
| 4: execution order and independent green deliverables | 1–27 | Tasks 1–26 use per-task RED/GREEN/lint/commit checkpoints; Task 27 records clean-tree provider-free evidence and preview |
| 5: target architecture | 1, 4–18, 21–25 | registry, facade, canonical evidence, finalization, Assistant and release boundaries |
| 6: graph and registry contract | 1, 14 | separate exact-version registry and exact compiled conditional topology tests |
| 7: input, state, output and runtime context | 2, 13, 14 | immutable ingress, request-local concurrency, explicit runtime dependencies and no checkpoint |
| 8: retriever port | 2, 6, 7 | common real `Runnable[RetrievalRequest, RetrievalResult]`, scope and one-embedding tests |
| 9: retrieval policy and bounds | 5–8, 14 | scorer parity, 50-window, slot limits, readiness fences and Assistant-only term preflight |
| 10: evidence trust policy | 4, 5, 8, 13, 17 | raw/trusted eligibility, tier order, fresh projection and whole-message redaction |
| 11: canonical serving identity | 4, 8, 13, 17 | exact internal/public mapping, typed version envelopes and HMAC/revalidation goldens |
| 12: evidence slot and answer contract | 8, 10, 13, 14 | E1–E8, strict blocks/schema, selected-vs-influence and final projection tests |
| 13: model, token and cost policy | 9–12, 22–25 | router, strict usage, exact-two ledger, safety authority and bounded fake runner |
| 14: fallback and error contract | 7, 12–18 | exact fallback union, status/outcome matrices, terminal costs and no-retry recovery |
| 15: V1 API and UX compatibility | 15, 16, 18–20 | exact keys/status/nullability, capability, safe links and same-screen reconciliation |
| 16: persistence, privacy and keyed identity | 2, 8, 11–13, 17, 18 | exact-byte writes, two-phase commit, dependency HMACs and privacy/redaction tests |
| 17: shadow parity and rollout | 14, 21 | exhaustive route policy, exact five deltas and retrieval-only shadow tests |
| 18: testing and quality gate | 1–27, especially 21, 25–27 | >=60 fake goldens, 30-case fake live runner, PostgreSQL/frontend/secret/clean-tree gates |
| 19: D.1 answer-cache handoff | explicit handoff only | D Core no-cache assertions; separate post-D-Core design/plan/approval required |
| 20: Deliverable E handoff | explicit handoff only | no Neo4j dependency; separate post-D.1 design/plan/approval required |
| 21: rollback | 14, 21, 22, 27 | deployment-static rollback, retained breaker health and unchanged legacy path tests |
| 22: success criteria | 21–27 | library, leak, parity, cost, quality, release and paid-zero acceptance evidence |
| 23: written-spec approval gate | 24–27 | clean-commit zero-call preview and mandatory fresh paid-execution confirmation |

---

## Final Implementation Review Checklist

- [ ] Every approved spec section 1–23 maps to at least one task/test above; no contract is deferred to D.1 or E except answer caching and Neo4j explicitly excluded by the spec.
- [ ] All new retrieval adapters are real LangChain `Runnable` implementations and the answer path is a real LangGraph `StateGraph` with conditional edges.
- [ ] The Review registry/checkpointer is unchanged and RAG state is request-local with zero checkpoint rows.
- [ ] Permission, source scope, evidence trust, citation authority, revalidation, and Assistant whole-message redaction are fail closed.
- [ ] Every future V2 run has exact-two cost children and no provider call can precede a committed one-use claim.
- [ ] V1 keys/status/nullability and same-screen interaction are preserved; V2 model text cannot create a link.
- [ ] D Core performs no answer reuse and adds no Redis/Neo4j/Slack/CDC dependency.
- [ ] Release tables are isolated from application metadata/Alembic and the live runner is unreachable from automated/provider-free profiles.
- [ ] Provider-free gates are green, worktree is clean, paid calls are still zero, and the final zero-call preview is shown before any live execution approval.

## Implementation Handoff

After this plan receives separate approval, choose one execution mode:

1. **Subagent-Driven (recommended):** use `superpowers:subagent-driven-development`, one task/commit at a time, with a specification review and code-quality review after every task.
2. **Inline Execution:** use `superpowers:executing-plans` in the current task, honoring the same RED/GREEN/commit checkpoints and stopping at every human decision boundary.

The next activity after plan approval is **actual implementation**, beginning with Task 1. No paid model is required for Tasks 1–27; the only possible paid action is the separately approved `rag_live_gate run` after the exact clean-commit zero-call preview.
