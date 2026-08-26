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
