# Task 2 Report: Provider and Agent Surface Characterization

## Status

Completed. Added the no-network provider-constructor and structured-output compatibility gate against the approved dependency lock.

## Characterization coverage

- Added `test_application_provider_and_agent_surfaces_bind_without_network_calls`.
  It constructs the Slack, Mail/Document, and RAG OpenAI/Gemini provider models
  with fake keys, verifies provider ordering and concrete client classes, binds
  `StructuredMemoryExtractionOutput`, constructs the project router, and builds
  both assistant email components without invoking a provider-backed model.
- Added `test_create_agent_returns_fake_structured_project_routing_response`.
  It verifies direct `with_structured_output(ProjectRoutingResult)` and then
  invokes the production `LangChainProjectRouterModel`/LangChain `create_agent`
  path using a hermetic `BaseChatModel` fake that returns a structured tool call.
- Production adapters were not modified; no external provider or network call
  was made.

## Verification

- Focused characterization cases: `2 passed in 2.29s`.
- Complete compatibility file: `12 passed in 0.76s`.
- High-signal provider/orchestration regressions: `56 passed in 0.52s`.
- `uv run --locked ruff check backend/tests/test_langchain_langgraph_dependency_compat.py`: `All checks passed!`.

## Concerns

- None. The approved lock supports the characterized provider constructors,
  direct structured output, and real `create_agent` path without network access.
