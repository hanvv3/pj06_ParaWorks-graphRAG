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
