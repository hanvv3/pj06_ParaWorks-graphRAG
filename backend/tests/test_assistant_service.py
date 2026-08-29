import pytest
from sqlalchemy.orm import Session

import backend.app.assistant.service as assistant_service
from backend.app.agents.rag_orchestrator_agent import answer_question_with_rag
from backend.app.agents.rag_orchestrator_agent.service import (
    RagEvidenceCandidate,
    build_serving_dependency_snapshot,
    retrieve_matching_knowledge_candidates,
)
from backend.app.assistant.service import (
    DEFAULT_CONVERSATION_TITLE,
    RECENT_CONTEXT_MESSAGE_LIMIT,
    append_assistant_message,
    append_user_message,
    assistant_message_evidence_is_live,
    build_contextual_question,
    create_conversation,
    eligible_context_messages,
    find_reusable_empty_conversation,
    get_owned_conversation,
    list_conversations,
    list_messages,
    serialize_conversation,
    serialize_message,
    summarize_conversation_title,
)
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AgentRun,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
    DecisionRecord,
    ReviewItem,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)
from backend.tests.test_auto_review_source_reconciliation import (
    _seed_explicit_history,
)


def test_conversations_are_scoped_to_user(db_session: Session) -> None:
    viewer = USERS['viewer']
    employee = USERS['hanvv-employee']
    create_conversation(db_session, viewer, title='Viewer private thread')
    create_conversation(db_session, employee, title='Employee private thread')

    viewer_conversations = list_conversations(db_session, viewer)

    assert [conversation.title for conversation in viewer_conversations] == ['Viewer private thread']


def test_get_owned_conversation_rejects_other_user(db_session: Session) -> None:
    viewer = USERS['viewer']
    employee = USERS['hanvv-employee']
    conversation = create_conversation(db_session, viewer, title='Viewer private thread')

    with pytest.raises(ValueError, match='assistant conversation not found'):
        get_owned_conversation(db_session, employee, conversation.id)


def test_append_messages_and_context_window(db_session: Session) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='Redis')
    for index in range(8):
        append_user_message(db_session, viewer, conversation, f'사용자 질문 {index}')
        append_assistant_message(
            db_session,
            viewer,
            conversation,
            content=f'비서 답변 {index}',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={'turn': index},
        )
    conversation.summary = '이전 대화는 Redis 작업 상태에 관한 내용입니다.'
    db_session.commit()

    messages = list_messages(db_session, viewer, conversation.id)
    contextual_question = build_contextual_question(
        conversation=conversation,
        messages=messages,
        new_message='그 다음 할 일은?',
    )

    assert len(messages) == 16
    assert '대화 요약: 이전 대화는 Redis 작업 상태에 관한 내용입니다.' in contextual_question
    assert '현재 질문: 그 다음 할 일은?' in contextual_question
    assert '사용자 질문 0' not in contextual_question
    assert '비서 답변 7' in contextual_question
    assert contextual_question.count('assistant:') <= RECENT_CONTEXT_MESSAGE_LIMIT // 2 + 1


def test_append_user_message_rejects_other_user(db_session: Session) -> None:
    viewer = USERS['viewer']
    employee = USERS['hanvv-employee']
    conversation = create_conversation(db_session, viewer, title='Viewer private thread')

    with pytest.raises(ValueError, match='assistant conversation not found'):
        append_user_message(db_session, employee, conversation, '권한 없는 질문')


def test_append_assistant_message_rejects_other_user(db_session: Session) -> None:
    viewer = USERS['viewer']
    employee = USERS['hanvv-employee']
    conversation = create_conversation(db_session, viewer, title='Viewer private thread')

    with pytest.raises(ValueError, match='assistant conversation not found'):
        append_assistant_message(
            db_session,
            employee,
            conversation,
            content='권한 없는 답변',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={},
        )


def test_append_user_message_rejects_blank_content(db_session: Session) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='Redis')

    with pytest.raises(ValueError, match='assistant message content is required'):
        append_user_message(db_session, viewer, conversation, '   ')

    assert list_messages(db_session, viewer, conversation.id) == []


def test_append_assistant_message_rejects_blank_content(db_session: Session) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='Redis')

    with pytest.raises(ValueError, match='assistant message content is required'):
        append_assistant_message(
            db_session,
            viewer,
            conversation,
            content='   ',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={},
        )

    assert list_messages(db_session, viewer, conversation.id) == []


def test_append_assistant_message_strips_content(db_session: Session) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='Redis')

    message = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='  Redis 작업은 진행 중입니다.  ',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
    )

    assert message.content == 'Redis 작업은 진행 중입니다.'


def test_contextual_question_excludes_matching_current_user_message(db_session: Session) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='Redis')
    append_user_message(db_session, viewer, conversation, 'Redis 상태 알려줘')
    append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Redis 작업은 진행 중입니다.',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
    )
    append_user_message(db_session, viewer, conversation, '그 다음 할 일은?')

    messages = list_messages(db_session, viewer, conversation.id)
    contextual_question = build_contextual_question(
        conversation=conversation,
        messages=messages,
        new_message='그 다음 할 일은?',
    )

    assert contextual_question.count('그 다음 할 일은?') == 1
    assert '현재 질문: 그 다음 할 일은?' in contextual_question


def test_finds_only_one_reusable_empty_conversation(db_session: Session) -> None:
    viewer = USERS['viewer']
    reusable = create_conversation(db_session, viewer, title=DEFAULT_CONVERSATION_TITLE)
    filled = create_conversation(db_session, viewer, title=DEFAULT_CONVERSATION_TITLE)
    append_user_message(db_session, viewer, filled, '이미 사용한 대화입니다')

    empty_conversation = find_reusable_empty_conversation(db_session, viewer)

    assert empty_conversation is not None
    assert empty_conversation.id == reusable.id


def test_first_user_message_sets_short_chat_history_title(db_session: Session) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title=DEFAULT_CONVERSATION_TITLE)
    question = '이번 주 목요일 오전 회의 일정과 준비할 문서를 기획팀 관점에서 정리해줘'

    append_user_message(db_session, viewer, conversation, question)

    db_session.refresh(conversation)
    assert conversation.title == summarize_conversation_title(question)
    assert len(conversation.title) <= 32
    assert conversation.title.endswith('…')


def test_pre_c5_unbound_evidence_answer_is_audit_only_but_non_evidence_operational_message_survives(
    db_session: Session,
) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='Legacy')
    legacy_answer = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Legacy answer bytes',
        citations=[{'source_id': 'legacy'}],
        source_ids=['legacy'],
        source_links=['https://legacy.invalid/evidence'],
        source_snippets=['legacy evidence'],
        permission_level='internal',
        hidden_match_count=1,
        permission_notice='legacy',
        agent_run_id=None,
        metadata={},
    )
    operational = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Recipient clarification required',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={'action_type': 'email_clarification'},
    )

    redacted = serialize_message(legacy_answer, db=db_session, user=viewer)
    visible = serialize_message(operational, db=db_session, user=viewer)

    assert redacted['metadata']['status'] == 'evidence_unavailable'
    assert redacted['citations'] == []
    assert redacted['source_ids'] == []
    assert redacted['source_links'] == []
    assert redacted['source_snippets'] == []
    assert redacted['hidden_match_count'] == 0
    assert visible['content'] == 'Recipient clarification required'


def test_evidence_derived_email_draft_without_exact_dependencies_fails_closed(
    db_session: Session,
) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='RAG email')

    with pytest.raises(ValueError, match='complete serving dependencies'):
        append_assistant_message(
            db_session,
            viewer,
            conversation,
            content='Evidence-derived email bytes',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level=None,
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={'action_type': 'email_draft'},
            evidence_derived=True,
        )

    assert db_session.query(AssistantMessage).count() == 0


def test_rag_answer_persists_complete_exact_dependencies_with_message_atomically(
    db_session: Session,
) -> None:
    viewer = USERS['viewer']
    history, _, _, approval = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    evidence_ids = tuple(
        row.id
        for row in db_session.query(TrustedKnowledgeEvidenceLink)
        .filter_by(approval_link_id=approval.id)
        .all()
    )
    conversation = create_conversation(db_session, viewer, title='Bound')
    candidate = RagEvidenceCandidate(
        source_id=f'history_event:{history.id}',
        source_url=history.source_links[0],
        text=f'{history.title}\n{history.reason}',
        source_snippet=history.source_snippets[0],
        author=None,
        timestamp=history.created_at.isoformat(),
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    dependency = build_serving_dependency_snapshot(db_session, candidate)
    assert dependency is not None

    message = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Bound answer',
        citations=[{'source_id': f'history_event:{history.id}'}],
        source_ids=[f'history_event:{history.id}'],
        source_links=history.source_links,
        source_snippets=history.source_snippets,
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
        serving_dependencies=(dependency,),
    )

    dependency = db_session.query(AssistantMessageEvidenceDependency).one()
    refs = db_session.query(AssistantMessageKnowledgeEvidenceRef).all()
    assert message.evidence_contract_version == 'assistant-evidence:v1'
    assert message.serving_dependency_count == 1
    assert dependency.serving_document_id == f'history_event:{history.id}'
    assert {ref.trusted_knowledge_evidence_link_id for ref in refs} == set(
        evidence_ids
    )


@pytest.mark.parametrize('stored_knowledge_type', ['decision', 'decision_record'])
def test_canonical_decision_candidate_persists_exact_stored_link_dependency(
    db_session: Session,
    stored_knowledge_type: str,
) -> None:
    viewer = USERS['viewer']
    history, item, _, approval = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    decision = DecisionRecord(
        project_key='project-a',
        title='Canonical decision dependency',
        decision_summary='Use the exact stored approval link identity.',
        source_links=list(item.source_links),
        source_snippets=list(item.source_snippets),
        confidence_score=0.99,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=item.id,
    )
    db_session.add(decision)
    db_session.flush([decision])
    item.item_type = 'decision_record'
    approval.knowledge_type = stored_knowledge_type
    approval.knowledge_id = decision.id
    db_session.commit()

    candidate = next(
        candidate
        for candidate in retrieve_matching_knowledge_candidates(
            db=db_session,
            question='exact stored approval link identity',
        )
        if candidate.source_id == f'decision_record:{decision.id}'
    )
    dependency = build_serving_dependency_snapshot(db_session, candidate)

    assert dependency is not None
    assert dependency.serving_document_id == f'decision_record:{decision.id}'
    assert dependency.knowledge_type == stored_knowledge_type
    assert dependency.approval_link_id == approval.id

    answer = answer_question_with_rag(
        db=db_session,
        user=viewer,
        question='exact stored approval link identity',
    )
    assert answer.source_ids == [f'decision_record:{decision.id}']
    assert answer.serving_dependencies == (dependency,)

    conversation = create_conversation(
        db_session, viewer, title='Canonical decision dependency'
    )
    message = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Bound canonical decision answer',
        citations=[{'source_id': candidate.source_id}],
        source_ids=[candidate.source_id],
        source_links=[candidate.source_url],
        source_snippets=[candidate.source_snippet],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
        serving_dependencies=(dependency,),
    )

    stored_dependency = db_session.query(
        AssistantMessageEvidenceDependency
    ).one()
    evidence_refs = db_session.query(
        AssistantMessageKnowledgeEvidenceRef
    ).all()
    assert stored_dependency.serving_document_id == (
        f'decision_record:{decision.id}'
    )
    assert stored_dependency.knowledge_type == stored_knowledge_type
    assert stored_dependency.approval_link_id == approval.id
    assert {ref.approval_link_id for ref in evidence_refs} == {approval.id}
    assert assistant_message_evidence_is_live(
        db_session, user=viewer, message=message
    )

    approval.active = False
    db_session.commit()

    assert not assistant_message_evidence_is_live(
        db_session, user=viewer, message=message
    )
    assert serialize_message(
        message, db=db_session, user=viewer
    )['metadata']['status'] == 'evidence_unavailable'


def test_assistant_dependency_source_drift_permission_narrowing_or_lookup_failure_fails_closed(
    db_session: Session,
) -> None:
    viewer = USERS['viewer']
    history, _, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    candidate = RagEvidenceCandidate(
        source_id=f'history_event:{history.id}',
        source_url=history.source_links[0],
        text=f'{history.title}\n{history.reason}',
        source_snippet=history.source_snippets[0],
        author=None,
        timestamp=history.created_at.isoformat(),
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    dependency = build_serving_dependency_snapshot(db_session, candidate)
    assert dependency is not None
    conversation = create_conversation(db_session, viewer, title='Content drift')
    message = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Bound answer',
        citations=[{'source_id': candidate.source_id}],
        source_ids=[candidate.source_id],
        source_links=[candidate.source_url],
        source_snippets=[candidate.source_snippet],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
        serving_dependencies=(dependency,),
    )
    assert serialize_message(message, db=db_session, user=viewer)['content'] == (
        'Bound answer'
    )

    history.reason = 'mutated after the answer was stored'
    db_session.commit()

    projected = serialize_message(message, db=db_session, user=viewer)
    assert projected['metadata']['status'] == 'evidence_unavailable'
    assert projected['source_ids'] == []


def test_assistant_dependency_rechecks_the_selected_effect_even_when_shared_provenance_remains(
    db_session: Session,
) -> None:
    viewer = USERS['viewer']
    history, auto_item, source, auto_link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    candidate = RagEvidenceCandidate(
        source_id=f'history_event:{history.id}',
        source_url=history.source_links[0],
        text=f'{history.title}\n{history.reason}',
        source_snippet=history.source_snippets[0],
        author=None,
        timestamp=history.created_at.isoformat(),
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    dependency = build_serving_dependency_snapshot(db_session, candidate)
    assert dependency is not None
    assert dependency.approval_link_id == auto_link.id
    conversation = create_conversation(db_session, viewer, title='Exact effect')
    message = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Bound answer',
        citations=[{'source_id': candidate.source_id}],
        source_ids=[candidate.source_id],
        source_links=[candidate.source_url],
        source_snippets=[candidate.source_snippet],
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
        serving_dependencies=(dependency,),
    )

    human_item = ReviewItem(
        item_type='history_event',
        payload=dict(auto_item.payload),
        source_links=list(auto_item.source_links),
        source_snippets=list(auto_item.source_snippets),
        confidence_score=auto_item.confidence_score,
        permission_level='internal',
        status='approved',
        candidate_contract_version='c5-v1',
        resolution_source='human',
    )
    db_session.add(human_item)
    db_session.flush([human_item])
    human_link = TrustedKnowledgeApprovalLink(
        knowledge_type='history_event',
        knowledge_id=history.id,
        review_item_id=human_item.id,
        security_scope_id=auto_link.security_scope_id,
        promotion_effect_kind='reaffirmation',
        resolution_source='human',
        claim_fingerprint='f' * 64,
        permission_level='internal',
        fingerprint_key_version=auto_link.fingerprint_key_version,
        fingerprint_key_material_verifier=(
            auto_link.fingerprint_key_material_verifier
        ),
        active=True,
    )
    db_session.add(human_link)
    db_session.flush([human_link])
    db_session.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=human_link.id,
            canonical_source_kind=source.source_type,
            canonical_source_id=str(source.id),
            canonical_version_or_signature=source.server_content_signature,
            evidence_hash='f' * 64,
            fingerprint_key_version=auto_link.fingerprint_key_version,
            fingerprint_key_material_verifier=(
                auto_link.fingerprint_key_material_verifier
            ),
        )
    )
    auto_item.auto_validation_id = None
    db_session.commit()

    projected = serialize_message(message, db=db_session, user=viewer)

    assert projected['metadata']['status'] == 'evidence_unavailable'
    assert projected['source_ids'] == []


def test_source_changes_between_rag_retrieval_and_message_commit_fail_closed_without_raw_answer_persistence(
    db_session: Session,
) -> None:
    viewer = USERS['viewer']
    history, _, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    candidate = RagEvidenceCandidate(
        source_id=f'history_event:{history.id}',
        source_url=history.source_links[0],
        text=f'{history.title}\n{history.reason}',
        source_snippet=history.source_snippets[0],
        author=None,
        timestamp=history.created_at.isoformat(),
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    dependency = build_serving_dependency_snapshot(db_session, candidate)
    assert dependency is not None
    conversation = create_conversation(db_session, viewer, title='Commit race')
    run = AgentRun(
        agent_name='rag_orchestrator_agent',
        prompt_version='test:v1',
        status='complete',
        source_window='test',
        cache_key='race',
        model_name='fake',
        permission_level='internal',
        metadata_={},
    )
    db_session.add(run)
    db_session.flush([run])
    history.reason = 'changed after retrieval but before persistence'

    with pytest.raises(ValueError, match='dependency changed before commit'):
        append_assistant_message(
            db_session,
            viewer,
            conversation,
            content='Must never be stored',
            citations=[{'source_id': candidate.source_id}],
            source_ids=[candidate.source_id],
            source_links=[candidate.source_url],
            source_snippets=[candidate.source_snippet],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=run.id,
            metadata={},
            serving_dependencies=(dependency,),
        )

    assert db_session.query(AssistantMessage).count() == 0
    assert db_session.query(AgentRun).filter_by(cache_key='race').count() == 0


def test_revoked_after_answer_leaks_nothing_to_message_list_context_summary_or_email_draft(
    db_session: Session,
) -> None:
    viewer = USERS['viewer']
    history, _, _, approval = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    candidate = RagEvidenceCandidate(
        source_id=f'history_event:{history.id}',
        source_url=history.source_links[0],
        text=f'{history.title}\n{history.reason}',
        source_snippet=history.source_snippets[0],
        author=None,
        timestamp=history.created_at.isoformat(),
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    dependency = build_serving_dependency_snapshot(db_session, candidate)
    assert dependency is not None
    conversation = create_conversation(db_session, viewer, title='Revoke')
    message = append_assistant_message(
        db_session,
        viewer,
        conversation,
        content='Sensitive bound answer',
        citations=[{'source_id': candidate.source_id}],
        source_ids=[candidate.source_id],
        source_links=[candidate.source_url],
        source_snippets=[candidate.source_snippet],
        permission_level='internal',
        hidden_match_count=2,
        permission_notice=None,
        agent_run_id=None,
        metadata={},
        serving_dependencies=(dependency,),
    )

    approval.active = False
    db_session.commit()

    projected = serialize_message(message, db=db_session, user=viewer)
    context_messages = eligible_context_messages(
        db_session, viewer, list(conversation.messages)
    )
    conversation_projection = serialize_conversation(
        conversation, db=db_session, user=viewer
    )
    contextual_question = build_contextual_question(
        conversation=conversation,
        messages=list(conversation.messages),
        new_message='Draft an email',
        db=db_session,
        user=viewer,
    )

    assert projected['content'].startswith('이 답변의 근거를')
    assert projected['citations'] == []
    assert projected['source_ids'] == []
    assert projected['source_links'] == []
    assert projected['source_snippets'] == []
    assert projected['hidden_match_count'] == 0
    assert message not in context_messages
    assert conversation_projection['summary'] is None
    assert 'Sensitive bound answer' not in contextual_question


def test_assistant_dependency_write_failure_rolls_back_message_and_agent_run_linkage(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = USERS['viewer']
    conversation = create_conversation(db_session, viewer, title='Rollback')
    run = AgentRun(
        agent_name='rag_orchestrator_agent',
        prompt_version='test:v1',
        status='complete',
        source_window='test',
        cache_key='dependency-write-failure',
        model_name='fake',
        permission_level='internal',
        metadata_={},
    )
    db_session.add(run)
    db_session.flush([run])

    def fail_dependency_write(*_args, **_kwargs) -> None:
        raise RuntimeError('dependency write failed')

    monkeypatch.setattr(
        assistant_service,
        '_persist_serving_dependencies',
        fail_dependency_write,
    )
    with pytest.raises(RuntimeError, match='dependency write failed'):
        append_assistant_message(
            db_session,
            viewer,
            conversation,
            content='Must roll back',
            citations=[{'source_id': 'history_event:1'}],
            source_ids=['history_event:1'],
            source_links=['https://example.invalid'],
            source_snippets=['evidence'],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=run.id,
            metadata={},
            serving_dependencies=(object(),),
        )

    assert db_session.query(AssistantMessage).count() == 0
    assert (
        db_session.query(AgentRun)
        .filter_by(cache_key='dependency-write-failure')
        .count()
        == 0
    )
