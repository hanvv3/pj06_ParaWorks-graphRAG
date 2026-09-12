from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import (
    AssistantConversation,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
)


def test_assistant_conversation_and_messages_persist(db_session: Session) -> None:
    conversation = AssistantConversation(
        user_id='employee-mina',
        title='Redis 회의 준비',
        summary='Redis 관련 이전 대화 요약',
    )
    db_session.add(conversation)
    db_session.flush()

    message = AssistantMessage(
        conversation_id=conversation.id,
        role='assistant',
        content='Redis는 일시적인 작업 상태 공유에 사용됩니다.',
        citations=[
            {
                'source_id': 'gmail-redis',
                'source_url': 'https://gmail.mock/redis',
                'permission_level': 'internal',
            }
        ],
        source_ids=['gmail-redis'],
        source_links=['https://gmail.mock/redis'],
        source_snippets=['Redis 작업 상태 근거'],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=7,
        metadata_={'retrieval_backend': 'deterministic_lexical'},
    )
    db_session.add(message)
    db_session.commit()

    stored = db_session.scalar(
        select(AssistantConversation).where(AssistantConversation.user_id == 'employee-mina')
    )

    assert stored is not None
    assert stored.title == 'Redis 회의 준비'
    assert stored.summary == 'Redis 관련 이전 대화 요약'
    assert len(stored.messages) == 1
    assert stored.messages[0].content == 'Redis는 일시적인 작업 상태 공유에 사용됩니다.'
    assert stored.messages[0].citations[0]['source_id'] == 'gmail-redis'
    assert stored.messages[0].agent_run_id == 7


def test_assistant_message_json_fields_track_mutation_and_keep_independent_defaults(
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-jun')
    first_message = AssistantMessage(
        conversation=conversation,
        role='assistant',
        content='첫 번째 답변',
    )
    second_message = AssistantMessage(
        conversation=conversation,
        role='assistant',
        content='두 번째 답변',
    )
    db_session.add(conversation)
    db_session.commit()

    first_message.citations.append({'source_id': 'source-1'})
    first_message.source_ids.append('source-1')
    first_message.source_links.append('https://docs.mock/source-1')
    first_message.source_snippets.append('첫 번째 근거')
    first_message.metadata_['retrieval_backend'] = 'deterministic_lexical'
    db_session.commit()
    db_session.expire_all()

    stored_messages = db_session.scalars(
        select(AssistantMessage).order_by(AssistantMessage.id)
    ).all()

    assert stored_messages[0].citations == [{'source_id': 'source-1'}]
    assert stored_messages[0].source_ids == ['source-1']
    assert stored_messages[0].source_links == ['https://docs.mock/source-1']
    assert stored_messages[0].source_snippets == ['첫 번째 근거']
    assert stored_messages[0].metadata_ == {'retrieval_backend': 'deterministic_lexical'}
    assert stored_messages[1].id == second_message.id
    assert stored_messages[1].citations == []
    assert stored_messages[1].source_ids == []
    assert stored_messages[1].source_links == []
    assert stored_messages[1].source_snippets == []
    assert stored_messages[1].metadata_ == {}


def test_assistant_conversation_messages_order_has_id_tie_breaker() -> None:
    order_by = AssistantConversation.messages.property.order_by

    assert AssistantMessage.created_at in order_by
    assert AssistantMessage.id in order_by


def test_assistant_evidence_contract_columns_are_legacy_nullable() -> None:
    assert AssistantMessage.__table__.c.evidence_contract_version.nullable is True
    assert AssistantMessage.__table__.c.serving_dependency_count.nullable is True
    checks = {
        constraint.name
        for constraint in AssistantMessage.__table__.constraints
        if constraint.name is not None
    }
    assert 'ck_assistant_messages_evidence_contract' in checks
    assert 'ck_assistant_messages_serving_dependency_count' in checks


def test_historical_null_content_marker_roundtrips_without_inferred_v2_authority(db_session):
    from backend.app.assistant.service import serialize_message
    from backend.app.core.demo_auth import USERS
    conversation = AssistantConversation(user_id=USERS['viewer'].id)
    message = AssistantMessage(conversation=conversation, role='assistant', content='Historical operational answer',
        evidence_contract_version='none-v1', serving_dependency_count=0,
        metadata_={'agent_name': 'rag_orchestrator_agent', 'prompt_version': 'rag-answer:v1'})
    db_session.add(conversation)
    db_session.commit()
    projected = serialize_message(message, db=db_session, user=USERS['viewer'])
    assert projected['content'] == 'Historical operational answer'
    assert message.content_write_mode is None
    assert message.assistant_message_content_hmac is None
    assert message.dependency_set_hmac_schema_version is None


def test_assistant_evidence_dependencies_bind_exact_parent_and_effect() -> None:
    parent_constraints = {
        constraint.name
        for constraint in AssistantMessageEvidenceDependency.__table__.constraints
    }
    child_constraints = {
        constraint.name
        for constraint in AssistantMessageKnowledgeEvidenceRef.__table__.constraints
    }
    assert {
        'uq_assistant_message_dependency_ordinal',
        'uq_assistant_message_dependency_serving_document',
        'fk_assistant_message_dependency_raw_chunk_identity',
        'fk_assistant_message_dependency_approval_target',
    } <= parent_constraints
    assert {
        'uq_assistant_message_knowledge_evidence_ref',
        'fk_assistant_knowledge_evidence_ref_same_dependency_effect',
        'fk_assistant_knowledge_evidence_ref_same_approval_effect',
    } <= child_constraints
