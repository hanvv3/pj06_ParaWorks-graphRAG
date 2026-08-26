# LangChain·LangGraph Dependency Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade ParaWorks to the approved LangChain, LangGraph, provider, and PostgreSQL checkpointer minor lines while proving that every existing AI integration surface remains importable and no-network testable.

**Architecture:** This is Deliverable A only. A focused compatibility test fixes the supported package ranges and exercises the public APIs that later deliverables depend on; `pyproject.toml` and `uv.lock` then move together to the approved resolution. No API route, graph topology, checkpoint lifecycle, Review Queue behavior, or RAG behavior changes in this deliverable.

**Tech Stack:** Python 3.12, uv, pytest, LangChain 1.3.x, LangGraph 1.2.x, `langchain-openai`, `langchain-google-genai`, `langgraph-checkpoint-postgres`, psycopg 3, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-langchain-langgraph-runtime-foundation-design.md`

## Global Constraints

- Use `langchain>=1.3.17,<1.4.0`.
- Use `langgraph>=1.2.11,<1.3.0`.
- Use `langchain-openai>=1.6.0,<1.7.0`.
- Use `langchain-google-genai>=4.3.5,<4.4.0`.
- Use `langgraph-checkpoint-postgres>=3.1.2,<3.2.0`.
- Use `psycopg[binary,pool]>=3.3.2`.
- Keep the exact reproducible resolution in `uv.lock` and restrict the refresh to the approved AI/checkpointer dependency closure.
- Do not add a compatibility shim, alternate orchestration wrapper, route change, graph node, feature flag, database migration, or checkpointer initialization.
- Do not call OpenAI, Gemini, PostgreSQL, Slack, Gmail, Drive, Calendar, OAuth, or embedding services from tests.
- Preserve current provider ordering, structured-output schemas, token/cost parsing, permission checks, Review Queue behavior, and SQLite smoke behavior.
- A resolver failure, constructor failure, or public API incompatibility is a stop condition for Deliverable A. Preserve the error evidence and return to the approved design instead of silently lowering a version or changing production behavior.
- Run all post-lock Python commands with `--locked` so verification cannot mutate the accepted resolution.

---

## File Structure

- Modify: `pyproject.toml`
  - Declares the six approved direct dependency ranges and required psycopg extras.
- Modify: `uv.lock`
  - Stores the exact target and transitive resolution produced by a targeted `uv lock` refresh.
- Create: `backend/tests/test_langchain_langgraph_dependency_compat.py`
  - Owns resolved-version, import, runtime-context, interrupt/resume, all application provider-constructor, structured-output, and `create_agent` compatibility checks without external calls.
- Modify: `plan.md`
  - Marks Deliverable A green and makes Deliverable B planning the next runtime priority only after all gates pass.
- Modify: `docs/portfolio-log.md`
  - Records the exact direct versions and verification story.
- Modify: `docs/superpowers/runbooks/session-handoff.md`
  - Gives the next worker the final locked versions and the boundary between import readiness and runtime integration.

Production modules are deliberately read-only in this deliverable. The compatibility tests exercise these existing surfaces:

- `backend/app/agent_runtime/orchestration.py`
  - `StateGraph`, `START`, `END`, `compile()`, `invoke()`, Mermaid rendering.
- `backend/app/agent_runtime/project_routing.py`
  - `langchain.agents.create_agent`, `@tool`, `response_format`, `structured_response`.
- `backend/app/agents/memory_extraction_agent/langchain_adapter.py`
  - `with_structured_output(StructuredMemoryExtractionOutput)`.
- `backend/app/agents/slack_agent/llm.py`
  - `ChatOpenAI` and `ChatGoogleGenerativeAI` constructor arguments.
- `backend/app/agents/mail_document_agent/llm.py`
  - Shared OpenAI/Gemini constructor and response metadata assumptions.
- `backend/app/agents/rag_orchestrator_agent/llm.py`
  - Provider ordering, fallback, content, and usage parsing.
- `backend/app/assistant/email_agent.py`
  - `ChatOpenAI(..., max_tokens=...)` for intent and draft builders.

---

### Task 1: Establish the RED Dependency Contract and Upgrade the Lock

**Files:**
- Create: `backend/tests/test_langchain_langgraph_dependency_compat.py`
- Modify: `pyproject.toml:8-18`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: installed distribution metadata; `StateGraph`, `Runtime`, `interrupt`, `Command`, `InMemorySaver`, `PostgresSaver`, `ConnectionPool`.
- Produces: an approved resolved-version contract plus proven LangGraph 1.2 runtime primitive imports for Deliverables B and C.

- [ ] **Step 1: Record the execution workspace state and capture the focused baseline**

Run these commands separately:

```powershell
git status --short
```

Expected: either no output or only understood pre-existing changes. Record every
pre-existing path before editing, preserve unrelated user or teammate work, and
stage only explicit paths from this plan. If a pre-existing change overlaps a
plan-owned file and cannot be safely separated, stop and ask the user before
continuing.

```powershell
uv run --frozen pytest backend/tests/test_agent_orchestration.py backend/tests/test_memory_extraction_langchain_adapter.py backend/tests/test_agent_runtime_project_routing.py backend/tests/test_agent_slack_project_routing.py backend/tests/test_slack_agent.py backend/tests/test_mail_document_agent.py backend/tests/test_rag_orchestrator_llm.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_assistant_email_agent.py -q
```

Expected: all selected baseline tests pass without a live provider call. Record the observed pass count in the session notes for comparison after the lock refresh.

- [ ] **Step 2: Write the compatibility test before changing dependencies**

Create `backend/tests/test_langchain_langgraph_dependency_compat.py` with this content:

```python
from importlib.metadata import version

import pytest


APPROVED_MINOR_LINES = (
    ('langchain', (1, 3, 17), (1, 4, 0)),
    ('langgraph', (1, 2, 11), (1, 3, 0)),
    ('langchain-openai', (1, 6, 0), (1, 7, 0)),
    ('langchain-google-genai', (4, 3, 5), (4, 4, 0)),
    ('langgraph-checkpoint-postgres', (3, 1, 2), (3, 2, 0)),
)


def _release_tuple(distribution: str) -> tuple[int, int, int]:
    release = version(distribution).split('+', maxsplit=1)[0].split('-', maxsplit=1)[0]
    return tuple(int(part) for part in release.split('.')[:3])


@pytest.mark.parametrize(('distribution', 'minimum', 'upper_bound'), APPROVED_MINOR_LINES)
def test_resolved_ai_package_is_in_approved_minor_line(
    distribution: str,
    minimum: tuple[int, int, int],
    upper_bound: tuple[int, int, int],
) -> None:
    resolved = _release_tuple(distribution)

    assert minimum <= resolved < upper_bound


def test_resolved_psycopg_meets_floor() -> None:
    assert _release_tuple('psycopg') >= (3, 3, 2)


def test_checkpoint_and_pool_symbols_import_without_database_access() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.types import Command, interrupt
    from psycopg_pool import ConnectionPool

    assert callable(interrupt)
    assert callable(Command)
    assert callable(InMemorySaver)
    assert callable(PostgresSaver.from_conn_string)
    assert callable(ConnectionPool)


def test_state_graph_accepts_typed_runtime_context() -> None:
    from langgraph.graph import END, START, StateGraph
    from langgraph.runtime import Runtime
    from typing_extensions import TypedDict

    class State(TypedDict):
        text: str

    class Context(TypedDict):
        suffix: str

    def append_context(state: State, runtime: Runtime[Context]) -> dict[str, str]:
        return {'text': f"{state['text']}{runtime.context['suffix']}"}

    builder = StateGraph(State, context_schema=Context)
    builder.add_node('append_context', append_context)
    builder.add_edge(START, 'append_context')
    builder.add_edge('append_context', END)

    result = builder.compile().invoke({'text': 'runtime'}, context={'suffix': '-ok'})

    assert result == {'text': 'runtime-ok'}


def test_state_graph_routes_with_conditional_edge() -> None:
    from typing import Literal

    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict):
        route: str
        visited: str

    def route_node(_state: State) -> dict[str, str]:
        return {}

    def choose_route(state: State) -> Literal['left', 'right']:
        return 'left' if state['route'] == 'left' else 'right'

    def visit_left(_state: State) -> dict[str, str]:
        return {'visited': 'left'}

    def visit_right(_state: State) -> dict[str, str]:
        return {'visited': 'right'}

    builder = StateGraph(State)
    builder.add_node('route', route_node)
    builder.add_node('left', visit_left)
    builder.add_node('right', visit_right)
    builder.add_edge(START, 'route')
    builder.add_conditional_edges(
        'route',
        choose_route,
        {'left': 'left', 'right': 'right'},
    )
    builder.add_edge('left', END)
    builder.add_edge('right', END)

    result = builder.compile().invoke({'route': 'right', 'visited': ''})

    assert result['visited'] == 'right'


def test_interrupt_resumes_same_in_memory_thread_with_command() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
    from typing_extensions import TypedDict

    class State(TypedDict):
        resolution: str | None

    def await_review(_state: State) -> dict[str, str]:
        return {'resolution': interrupt({'kind': 'review_resolution'})}

    builder = StateGraph(State)
    builder.add_node('await_review', await_review)
    builder.add_edge(START, 'await_review')
    builder.add_edge('await_review', END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'dependency-compatibility-smoke'}}

    paused = graph.invoke({'resolution': None}, config, durability='sync')
    resumed = graph.invoke(Command(resume='approved'), config, durability='sync')

    assert paused['__interrupt__']
    assert resumed['resolution'] == 'approved'
```

This file contains ten pytest cases because the approved package contract is parameterized into five cases.

- [ ] **Step 3: Run the test against the old lock and verify RED**

```powershell
uv run --frozen pytest backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

Expected: RED. The existing `langchain 1.2.17`, `langgraph 1.1.10`, provider packages, and missing `langgraph-checkpoint-postgres` do not satisfy the approved contract. The failure must be about version/import readiness, not a live network or database attempt.

- [ ] **Step 4: Replace the direct dependency declarations with the approved ranges**

Change only the AI/checkpointer portion of `pyproject.toml` to:

```toml
    "langchain>=1.3.17,<1.4.0",
    "langchain-google-genai>=4.3.5,<4.4.0",
    "langchain-openai>=1.6.0,<1.7.0",
    "langgraph>=1.2.11,<1.3.0",
    "langgraph-checkpoint-postgres>=3.1.2,<3.2.0",
    "pgvector>=0.4.2",
    "psycopg[binary,pool]>=3.3.2",
```

Do not change unrelated direct dependencies.

- [ ] **Step 5: Refresh only the approved dependency closure**

Run:

```powershell
uv lock --upgrade-package langchain --upgrade-package langgraph --upgrade-package langchain-openai --upgrade-package langchain-google-genai --upgrade-package langgraph-checkpoint-postgres --upgrade-package psycopg
```

Then run:

```powershell
uv lock --check
```

Expected: resolution succeeds and the lock is current.

- [ ] **Step 6: Audit the lock diff before installing it**

Run:

```powershell
git diff -- pyproject.toml uv.lock
```

Accept changes only for:

- the six direct dependency declarations above;
- `langchain`, `langchain-core`, `langchain-openai`, `langchain-google-genai`;
- `langgraph`, `langgraph-checkpoint`, `langgraph-prebuilt`, `langgraph-sdk`;
- `langgraph-checkpoint-postgres` and its required checkpoint/serialization closure;
- `psycopg`, `psycopg-binary`, `psycopg-pool`;
- transitive packages whose changed constraints are directly required by that closure.

An unrelated FastAPI, SQLAlchemy, Celery, frontend, connector, or test-tool upgrade is a stop condition. Do not accept a broad lock refresh.

- [ ] **Step 7: Sync the locked environment and verify GREEN**

Run these commands separately:

```powershell
uv sync --all-groups --locked
```

```powershell
uv run --locked pytest backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

Expected:

```text
10 passed
```

No database or provider endpoint is contacted.

- [ ] **Step 8: Lint, review, and commit the dependency contract**

Run:

```powershell
uv run --locked ruff check backend/tests/test_langchain_langgraph_dependency_compat.py
```

Expected:

```text
All checks passed!
```

Run:

```powershell
git diff --check
```

Expected: no output.

Commit only the dependency contract:

```powershell
git add pyproject.toml uv.lock backend/tests/test_langchain_langgraph_dependency_compat.py
git commit -m "chore: refresh langchain langgraph dependency line"
```

---

### Task 2: Characterize ParaWorks Provider, Structured-Output, and Agent Surfaces

**Files:**
- Modify: `backend/tests/test_langchain_langgraph_dependency_compat.py`

**Interfaces:**
- Consumes: Slack, Mail/Document, and RAG provider settings/builders; `StructuredMemoryExtractionOutput`; `ProjectOption`; `LangChainProjectRouterModel`; assistant email builders; and the newly locked provider packages.
- Produces: a no-network compatibility gate for the exact constructor kwargs and LangChain agent APIs already used by production code.

This task adds characterization coverage for existing behavior; it does not introduce a new product behavior. Therefore the first run is expected to pass on the approved lock. A failure is an incompatibility stop signal, not authorization to edit production adapters in Deliverable A.

- [ ] **Step 1: Add a no-network application-surface test**

Append this test to `backend/tests/test_langchain_langgraph_dependency_compat.py`:

```python
def test_application_provider_and_agent_surfaces_bind_without_network_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime.project_routing import (
        LangChainProjectRouterModel,
        ProjectOption,
    )
    from backend.app.agents.mail_document_agent.llm import (
        MailDocumentLlmSettings,
        build_langchain_mail_document_agent_model,
    )
    from backend.app.agents.memory_extraction_agent.langchain_adapter import (
        StructuredMemoryExtractionOutput,
    )
    from backend.app.agents.rag_orchestrator_agent.llm import (
        RagLlmSettings,
        build_langchain_rag_orchestrator_model,
    )
    from backend.app.agents.slack_agent.llm import (
        SlackLlmSettings,
        build_langchain_slack_agent_model,
    )
    from backend.app.assistant.email_agent import (
        EmailDraftComposer,
        EmailIntentGate,
        build_email_draft_composer,
        build_email_intent_gate,
    )
    from backend.app.core.config import Settings

    monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)
    monkeypatch.delenv('GOOGLE_CLOUD_PROJECT', raising=False)
    monkeypatch.delenv('GOOGLE_CLOUD_LOCATION', raising=False)
    monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)

    slack_model = build_langchain_slack_agent_model(
        SlackLlmSettings(
            enabled=True,
            provider_order=('openai', 'gemini'),
            openai_api_key='compat-openai-key',
            gemini_api_key='compat-gemini-key',
        )
    )
    mail_model = build_langchain_mail_document_agent_model(
        MailDocumentLlmSettings(
            enabled=True,
            provider_order=('openai', 'gemini'),
            openai_api_key='compat-openai-key',
            gemini_api_key='compat-gemini-key',
        )
    )
    rag_model = build_langchain_rag_orchestrator_model(
        RagLlmSettings(
            enabled=True,
            provider_order=('openai', 'gemini'),
            openai_api_key='compat-openai-key',
            gemini_api_key='compat-gemini-key',
        )
    )
    openai_chat = slack_model.providers[0].chat_model
    gemini_chat = slack_model.providers[1].chat_model

    structured_model = openai_chat.with_structured_output(StructuredMemoryExtractionOutput)
    router = LangChainProjectRouterModel(
        chat_model=openai_chat,
        projects=[
            ProjectOption(
                project_key='project-compat',
                name='Compatibility Project',
                summary='No-network LangChain agent construction smoke',
            )
        ],
        model_name='gpt-5.4-mini',
    )
    email_settings = Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        assistant_email_agent_enabled=True,
        openai_api_key='compat-openai-key',
    )

    assert [provider.provider for provider in slack_model.providers] == ['openai', 'gemini']
    assert [provider.provider for provider in mail_model.providers] == ['openai', 'gemini']
    assert [provider.provider for provider in rag_model.providers] == ['openai', 'gemini']
    assert openai_chat.__class__.__name__ == 'ChatOpenAI'
    assert gemini_chat.__class__.__name__ == 'ChatGoogleGenerativeAI'
    assert callable(structured_model.invoke)
    assert router.model_name == 'gpt-5.4-mini'
    assert isinstance(build_email_intent_gate(email_settings), EmailIntentGate)
    assert isinstance(build_email_draft_composer(email_settings), EmailDraftComposer)
```

The test constructs clients and a LangChain agent graph but never calls `invoke()` on a provider-backed model.

- [ ] **Step 2: Add an actual `create_agent` fake structured-response test**

Append this test to the same file:

```python
def test_create_agent_returns_fake_structured_project_routing_response() -> None:
    from typing import Any

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from backend.app.agent_runtime.project_routing import (
        LangChainProjectRouterModel,
        ProjectOption,
        ProjectRoutingResult,
    )

    class FakeStructuredChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return 'fake-structured-tool-model'

        def bind_tools(
            self,
            tools: Any,
            *,
            tool_choice: str | None = None,
            **kwargs: Any,
        ) -> BaseChatModel:
            return self

        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content='',
                            tool_calls=[
                                {
                                    'name': 'ProjectRoutingResult',
                                    'args': {
                                        'decisions': [
                                            {
                                                'source_id': 'source-1',
                                                'item_index': 0,
                                                'project_key': 'project-compat',
                                                'project_name': 'Compatibility Project',
                                                'confidence_score': 0.91,
                                                'assignment_summary': 'Compatibility routing result.',
                                                'assignment_reason': 'The fake model selected the registered project.',
                                                'alternatives': [],
                                                'needs_user_selection': False,
                                            }
                                        ],
                                        'input_tokens': 7,
                                        'output_tokens': 3,
                                        'model_name': 'fake-structured-tool-model',
                                    },
                                    'id': 'structured-1',
                                    'type': 'tool_call',
                                }
                            ],
                        )
                    )
                ]
            )

    fake_model = FakeStructuredChatModel()
    direct_structured = fake_model.with_structured_output(ProjectRoutingResult).invoke(
        'Route source-1 to the registered project.'
    )
    router = LangChainProjectRouterModel(
        chat_model=fake_model,
        projects=[
            ProjectOption(
                project_key='project-compat',
                name='Compatibility Project',
                summary='Fake structured response smoke',
            )
        ],
        model_name='fake-structured-tool-model',
    )

    result = router.invoke({'candidate_items': []})

    assert isinstance(direct_structured, ProjectRoutingResult)
    assert direct_structured.decisions[0].project_key == 'project-compat'
    assert result['decisions'][0]['project_key'] == 'project-compat'
    assert result['model_name'] == 'fake-structured-tool-model'
```

This test invokes the real LangChain `create_agent` graph with a hermetic fake chat model. The same fake model first proves the direct `with_structured_output()` path, so an agent failure isolates the `create_agent` compatibility boundary. It does not instantiate or call an external provider.

- [ ] **Step 3: Run the two new characterization tests**

```powershell
uv run --locked pytest backend/tests/test_langchain_langgraph_dependency_compat.py::test_application_provider_and_agent_surfaces_bind_without_network_calls backend/tests/test_langchain_langgraph_dependency_compat.py::test_create_agent_returns_fake_structured_project_routing_response -q
```

Expected:

```text
2 passed
```

- [ ] **Step 4: Run the complete compatibility file**

```powershell
uv run --locked pytest backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

Expected:

```text
12 passed
```

- [ ] **Step 5: Run the existing high-signal provider and orchestration regressions**

```powershell
uv run --locked pytest backend/tests/test_agent_orchestration.py backend/tests/test_memory_extraction_langchain_adapter.py backend/tests/test_agent_runtime_project_routing.py backend/tests/test_agent_slack_project_routing.py backend/tests/test_slack_agent.py backend/tests/test_mail_document_agent.py backend/tests/test_rag_orchestrator_llm.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_assistant_email_agent.py -q
```

Expected: all selected tests pass with the same count as the baseline captured in Task 1. Provider response parsing, fallback order, cost metadata, project tools, structured output, and existing linear LangGraph behavior remain unchanged.

- [ ] **Step 6: Lint and commit the characterization gate**

Run:

```powershell
uv run --locked ruff check backend/tests/test_langchain_langgraph_dependency_compat.py
```

Expected:

```text
All checks passed!
```

Commit:

```powershell
git add backend/tests/test_langchain_langgraph_dependency_compat.py
git commit -m "test: cover langchain langgraph compatibility"
```

---

### Task 3: Run the Deliverable A Gate and Record the Green Baseline

**Files:**
- Modify: `plan.md:352-386`
- Modify: `docs/portfolio-log.md:9`
- Modify: `docs/superpowers/runbooks/session-handoff.md:5`

**Interfaces:**
- Consumes: the locked dependency set and compatibility tests from Tasks 1 and 2.
- Produces: a documented green checkpoint authorizing a separate Deliverable B implementation plan; it does not authorize Deliverable B code in this branch.

- [ ] **Step 1: Run the complete backend regression with the accepted lock**

```powershell
uv run --locked pytest backend/tests -q
```

Expected: the full backend suite passes with zero new failures. Do not relabel a new failure as pre-existing; compare against the focused baseline and inspect the failing ownership area.

- [ ] **Step 2: Re-run the lock and source integrity checks**

Run these commands separately:

```powershell
uv lock --check
```

```powershell
uv run --locked ruff check backend/tests/test_langchain_langgraph_dependency_compat.py
```

```powershell
git diff --check
```

Expected: lock check and Ruff pass; diff check has no output.

Frontend lint, build, and Playwright are not part of this dependency-only gate because no frontend file or API response changes.

- [ ] **Step 3: Update the product plan only after the gate is green**

Perform this step only when Tasks 1 and 2 plus Task 3 Steps 1 and 2 have all
passed with fresh output in the current execution session. If any gate failed,
leave `plan.md`, the portfolio log, and the handoff status unchanged and report
the preserved failure evidence instead.

Under `plan.md` Milestone 4, replace the current Deliverable A priority wording with:

```markdown
- Deliverable A dependency compatibility: Done.
  - Locked LangChain 1.3.17, LangGraph 1.2.11, langchain-openai 1.6.0,
    langchain-google-genai 4.3.5, and langgraph-checkpoint-postgres 3.1.2.
  - Added no-network version, provider constructor, structured-output,
    `create_agent`, runtime-context, interrupt/resume, and checkpointer import
    compatibility tests.
- Next: write and review the separate Deliverable B runtime/checkpoint
  primitives implementation plan. Do not begin Review HITL V2 in this slice.
```

- [ ] **Step 4: Record the portfolio verification story**

Add this entry near the top of `docs/portfolio-log.md`:

```markdown
## 2026-08-26 LangChain·LangGraph dependency compatibility

- Upgraded and locked the approved dependency lines: LangChain 1.3.17,
  LangGraph 1.2.11, langchain-openai 1.6.0,
  langchain-google-genai 4.3.5, and
  langgraph-checkpoint-postgres 3.1.2.
- Added no-network compatibility coverage for the current OpenAI/Gemini
  constructors, LangChain structured output and `create_agent`, typed LangGraph
  runtime context, `interrupt()` / `Command(resume=...)`, and PostgreSQL saver
  imports.
- Verified the focused AI integration suite and complete backend suite against
  the locked environment. This deliverable changes no production route, graph
  topology, Review Queue behavior, or RAG behavior.
```

- [ ] **Step 5: Update the handoff boundary for Deliverable B**

Add this entry near the top of `docs/superpowers/runbooks/session-handoff.md`:

```markdown
## 2026-08-26 LangChain·LangGraph dependency compatibility

- Deliverable A is green with LangChain 1.3.17, LangGraph 1.2.11,
  langchain-openai 1.6.0, langchain-google-genai 4.3.5, and
  langgraph-checkpoint-postgres 3.1.2 locked.
- `backend/tests/test_langchain_langgraph_dependency_compat.py` proves the
  supported version ranges and no-network API surfaces.
- `PostgresSaver` is import-ready only. No saver lifecycle, `.setup()`, schema,
  feature flag, runtime context contract, or public V2 route was implemented in
  Deliverable A.
- The next worker must write the Deliverable B runtime/checkpoint primitives
  plan from the approved foundation spec before changing production code.
```

- [ ] **Step 6: Review the final Deliverable A diff and commit documentation**

Run:

```powershell
git status --short
```

Expected: only `plan.md`, `docs/portfolio-log.md`, and
`docs/superpowers/runbooks/session-handoff.md` are newly modified after the two
implementation commits. Any unrelated pre-existing paths recorded in Task 1
remain unchanged and must not be staged.

Run:

```powershell
git diff --check
```

Expected: no new uncommitted paths from Deliverable A. If Task 1 recorded
pre-existing changes, the output must match that preserved baseline instead of
being forced clean.

Commit:

```powershell
git add plan.md docs/portfolio-log.md docs/superpowers/runbooks/session-handoff.md
git commit -m "docs: record langgraph dependency compatibility"
```

- [ ] **Step 7: Confirm the final handoff state**

Run these commands separately:

```powershell
git status --short
```

Expected: no output.

```powershell
git log -3 --oneline
```

Expected commits, newest first:

```text
docs: record langgraph dependency compatibility
test: cover langchain langgraph compatibility
chore: refresh langchain langgraph dependency line
```

Deliverable A is complete only after all three commits and all verification gates are green. The next action is planning Deliverable B, not implementing Review HITL V2 or GraphRAG.
