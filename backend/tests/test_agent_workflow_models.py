from collections.abc import Callable

import pytest
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.models import (
    AgentRun,
    AgentRuntimeSchemaVersion,
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    DecisionRecord,
    HistoryEvent,
    ReviewItem,
    TimelineEvent,
    Todo,
)


@pytest.fixture
def engine() -> Engine:
    target = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(bind=target)
    return target


def _thread(sequence: int, *, client_request_id: str | None) -> AgentWorkflowThread:
    return AgentWorkflowThread(
        thread_id=f'thread-{sequence}',
        workflow_name='company_memory_review_v2',
        graph_version='review-v2.0',
        checkpoint_thread_id=f'checkpoint-{sequence}',
        checkpoint_store='memory',
        owner_subject_id='user-1',
        security_scope_id='workspace-1',
        client_request_id=client_request_id,
        input_hash='a' * 64,
        evidence_version_hash='b' * 64,
    )


def _evidence_ref(sequence: int, *, ordinal: int) -> AgentWorkflowEvidenceRef:
    return AgentWorkflowEvidenceRef(
        workflow_thread_id='thread-1',
        ordinal=ordinal,
        canonical_source_type='document_version',
        canonical_table='document_versions',
        canonical_row_id=sequence,
        document_version_id=sequence,
        external_revision=f'revision-{sequence}',
        content_signature=f'signature-{sequence}',
        permission_level_snapshot='internal',
        content_fingerprint=f'{sequence:064d}',
    )


def _review_item(sequence: int, *, candidate_key: str | None) -> ReviewItem:
    return ReviewItem(
        item_type='timeline_event',
        payload={'title': f'Candidate {sequence}'},
        source_links=[f'https://example.test/{sequence}'],
        source_snippets=[f'Evidence {sequence}'],
        confidence_score=0.9,
        permission_level='internal',
        workflow_thread_id='thread-1' if candidate_key is not None else None,
        candidate_key=candidate_key,
    )


def _agent_run(sequence: int, *, effect_key: str | None) -> AgentRun:
    return AgentRun(
        agent_name='mail_document_agent',
        prompt_version='v1',
        source_window=f'window-{sequence}',
        cache_key=f'cache-{sequence}',
        model_name='deterministic-test-model',
        permission_level='internal',
        workflow_thread_id='thread-1' if effect_key is not None else None,
        effect_key=effect_key,
    )


def _decision(sequence: int, source_review_item_id: int | None) -> DecisionRecord:
    return DecisionRecord(
        title=f'Decision {sequence}',
        decision_summary='Approved decision',
        source_links=['https://example.test/decision'],
        source_snippets=['Decision evidence'],
        confidence_score=0.9,
        permission_level='internal',
        source_review_item_id=source_review_item_id,
    )


def _history(sequence: int, source_review_item_id: int | None) -> HistoryEvent:
    return HistoryEvent(
        title=f'History {sequence}',
        reason='Approved history',
        source_links=['https://example.test/history'],
        source_snippets=['History evidence'],
        confidence_score=0.9,
        permission_level='internal',
        source_review_item_id=source_review_item_id,
    )


def _timeline(sequence: int, source_review_item_id: int | None) -> TimelineEvent:
    return TimelineEvent(
        title=f'Timeline {sequence}',
        result_summary='Approved timeline event',
        source_links=['https://example.test/timeline'],
        source_snippets=['Timeline evidence'],
        confidence_score=0.9,
        permission_level='internal',
        source_review_item_id=source_review_item_id,
    )


def _todo(sequence: int, source_review_item_id: int | None) -> Todo:
    return Todo(
        title=f'Todo {sequence}',
        priority='medium',
        priority_reason='Approved todo',
        source_links=['https://example.test/todo'],
        source_snippets=['Todo evidence'],
        confidence_score=0.9,
        permission_level='internal',
        source_review_item_id=source_review_item_id,
    )


def _assert_duplicate_rejected(engine: Engine, first: object, second: object) -> None:
    with Session(engine) as session:
        session.add_all([first, second])
        with pytest.raises(IntegrityError):
            session.commit()


def _assert_rows_accepted(engine: Engine, first: object, second: object) -> None:
    with Session(engine) as session:
        session.add_all([first, second])
        session.commit()


def test_workflow_models_create_expected_constraints_and_defaults(engine: Engine) -> None:
    inspector = inspect(engine)
    assert inspector.get_unique_constraints('agent_workflow_evidence_refs')
    assert {
        index['name'] for index in inspector.get_indexes('agent_workflow_threads')
    } >= {'uq_agent_workflow_thread_client_request'}

    thread = _thread(1, client_request_id='request-1')
    request = AgentWorkflowRequest(
        workflow_thread_id='thread-1',
        input_schema_version='review-request-v1',
        agent_names=['mail_document_agent'],
        selection_policy_version='selection-v1',
        input_hash='a' * 64,
        fingerprint_key_version='key-v1',
    )
    schema_version = AgentRuntimeSchemaVersion(
        component='langgraph_checkpoint',
        package_name='langgraph-checkpoint-postgres',
        package_version='3.1.2',
        schema_revision=1,
    )
    with Session(engine) as session:
        session.add_all([thread, request, schema_version])
        session.flush()
        assert thread.status == 'created'
        assert thread.state_version == 0
        assert request.request_kind == 'review_source_versions'


def test_thread_client_request_key_rejects_duplicate_non_null_values(
    engine: Engine,
) -> None:
    _assert_duplicate_rejected(
        engine,
        _thread(1, client_request_id='request-1'),
        _thread(2, client_request_id='request-1'),
    )


def test_thread_client_request_key_accepts_multiple_null_values(engine: Engine) -> None:
    _assert_rows_accepted(
        engine,
        _thread(1, client_request_id=None),
        _thread(2, client_request_id=None),
    )


def test_evidence_ordinal_rejects_duplicate_values_within_thread(engine: Engine) -> None:
    _assert_duplicate_rejected(
        engine,
        _evidence_ref(1, ordinal=0),
        _evidence_ref(2, ordinal=0),
    )


@pytest.mark.parametrize(
    ('factory', 'key'),
    [
        (_review_item, 'candidate-1'),
        (_agent_run, 'effect-1'),
    ],
)
def test_thread_bound_effect_keys_reject_duplicate_non_null_values(
    engine: Engine,
    factory: Callable,
    key: str,
) -> None:
    _assert_duplicate_rejected(engine, factory(1, **_key_argument(factory, key)), factory(2, **_key_argument(factory, key)))


@pytest.mark.parametrize('factory', [_review_item, _agent_run])
def test_thread_bound_effect_keys_accept_multiple_null_values(
    engine: Engine,
    factory: Callable,
) -> None:
    _assert_rows_accepted(engine, factory(1, **_key_argument(factory, None)), factory(2, **_key_argument(factory, None)))


def _key_argument(factory: Callable, value: str | None) -> dict[str, str | None]:
    if factory is _review_item:
        return {'candidate_key': value}
    return {'effect_key': value}


@pytest.mark.parametrize('factory', [_decision, _history, _timeline, _todo])
def test_knowledge_provenance_rejects_duplicate_non_null_review_item_ids(
    engine: Engine,
    factory: Callable[[int, int | None], object],
) -> None:
    _assert_duplicate_rejected(engine, factory(1, 42), factory(2, 42))


@pytest.mark.parametrize('factory', [_decision, _history, _timeline, _todo])
def test_knowledge_provenance_accepts_multiple_null_review_item_ids(
    engine: Engine,
    factory: Callable[[int, int | None], object],
) -> None:
    _assert_rows_accepted(engine, factory(1, None), factory(2, None))
