from contextlib import suppress

from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import (
    fingerprint_key_material_verifier,
)
from backend.app.agents.rag_orchestrator_agent import answer_question_with_rag
from backend.app.agents.rag_orchestrator_agent.service import (
    build_default_rag_orchestrator_agent,
    retrieve_matching_knowledge_candidates,
    vector_documents_from_candidates,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AgentRun,
    AutoReviewRuntimeKeyState,
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
)
from backend.app.rag.vector_store import InMemoryVectorStore, VectorDocument


def seed_chunk(db: Session, source_type: str, source_id: str, text: str, permission_level: str) -> None:
    signature = 'a' * 64
    settings = Settings(database_url='sqlite://')
    if db.get(AutoReviewRuntimeKeyState, 'auto_review_trust_promotion') is None:
        db.add(
            AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                fingerprint_key_version=(
                    settings.agent_runtime_fingerprint_key_version
                ),
                fingerprint_key_material_verifier=(
                    fingerprint_key_material_verifier(
                        settings.agent_runtime_fingerprint_secret
                    )
                ),
                generation=1,
                ready=True,
            )
        )
    source = Source(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://{source_type}.mock/{source_id}',
        title=f'{source_type} evidence',
        author='owner@example.com',
        permission_level=permission_level,
        raw_metadata={
            'ts': '2026-04-30T10:00:00+00:00',
            **(
                {'mime_type': 'text/plain'}
                if source_type in {'drive', 'gmail_attachment'}
                else {}
            ),
        },
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
    )
    db.add(source)
    db.flush()
    db.add(
        ReviewItem(
            item_type='source_evidence',
            payload={'source_ids': [source.source_id]},
            source_links=[source.source_url],
            source_snippets=[text[:240]],
            confidence_score=1.0,
            permission_level=permission_level,
            status='approved',
            resolution_source='human',
        )
    )

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()

    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name=f'server_{source_type}_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type=(
            'message/rfc822'
            if source_type == 'gmail'
            else 'text/calendar'
            if source_type == 'calendar'
            else 'text/plain'
        ),
        document_version_label=version.version,
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version='server-source-parser-policy:v1',
        parser_version='source-event-paragraph-parser:v1',
        chunk_policy_version='paragraph-chunks:1200:v1',
    )
    db.add(parser_run)
    db.flush()

    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=parser_run.id,
            chunk_index=0,
            text=text,
            source_snippet=text[:240],
            permission_level=permission_level,
            metadata_={'source_url': source.source_url, 'source_type': source_type},
        )
    )
    document.current_document_version_id = version.id
    parser_run.chunk_count = 1
    db.commit()


def test_rag_service_answers_from_visible_chunks(db_session: Session) -> None:
    seed_chunk(
        db_session,
        'gmail',
        'gmail-redis',
        'Redis should be used for transient job state while PostgreSQL stores durable records.',
        'internal',
    )

    answer = answer_question_with_rag(db=db_session, user=USERS['viewer'], question='Redis job state')

    assert answer.answer
    assert answer.source_links == ['https://gmail.mock/gmail-redis']
    assert answer.permission_notice is None
    assert answer.hidden_match_count == 0


def test_unsigned_raw_evidence_is_never_new_trusted_serving_content(
    db_session: Session,
) -> None:
    seed_chunk(
        db_session,
        'gmail',
        'gmail-legacy-unsigned',
        'Unsigned Project Ash launch evidence must remain audit-only.',
        'internal',
    )
    source = db_session.query(Source).one()
    parser_run = db_session.query(DocumentParserRun).one()
    source.server_content_signature_schema = None
    source.server_content_signature = None
    parser_run.server_content_signature_schema = None
    parser_run.server_content_signature = None
    parser_run.parser_policy_version = None
    parser_run.parser_version = None
    parser_run.chunk_policy_version = None
    db_session.commit()

    answer = answer_question_with_rag(
        db=db_session,
        user=USERS['viewer'],
        question='Project Ash launch evidence',
    )

    assert answer.answer == '권한 내에서 확인 가능한 근거를 찾지 못했습니다.'
    assert answer.source_links == []
    assert answer.source_snippets == []
    assert answer.serving_dependencies == ()


def test_rag_service_hides_restricted_chunks_for_viewer(db_session: Session) -> None:
    seed_chunk(
        db_session,
        'drive',
        'drive-pricing',
        'Confidential pricing uses Redis reserved capacity.',
        'restricted',
    )

    answer = answer_question_with_rag(db=db_session, user=USERS['viewer'], question='confidential pricing')

    assert answer.answer == '권한 내에서 확인 가능한 근거를 찾지 못했습니다.'
    assert answer.source_links == []
    assert answer.hidden_match_count == 1
    assert answer.permission_notice == 'Some sources may be hidden by permissions.'


def test_rag_service_persists_agent_run_metadata(db_session: Session) -> None:
    seed_chunk(
        db_session,
        'gmail',
        'gmail-rag-agent-run',
        'Redis should be used for transient job state while PostgreSQL stores durable records.',
        'internal',
    )

    answer = answer_question_with_rag(db=db_session, user=USERS['viewer'], question='Redis job state')

    agent_run = db_session.query(AgentRun).one()
    assert answer.agent_run_id == agent_run.id
    assert agent_run.agent_name == 'rag_orchestrator_agent'
    assert agent_run.prompt_version == 'rag-answer:v1'
    assert agent_run.status == 'complete'
    assert agent_run.source_window == 'ask:Redis job state'
    assert agent_run.model_name == answer.cost.model_name
    assert agent_run.input_tokens == answer.cost.token_usage.input_tokens
    assert agent_run.output_tokens == answer.cost.token_usage.output_tokens
    assert agent_run.total_tokens == answer.cost.token_usage.total_tokens
    assert agent_run.estimated_cost_usd == answer.cost.estimated_cost_usd
    assert agent_run.permission_level == 'internal'
    assert agent_run.cache_key == answer.cache_key
    assert agent_run.metadata_['question'] == 'Redis job state'
    assert agent_run.metadata_['source_count'] == 1
    assert agent_run.metadata_['hidden_match_count'] == 0


def test_rag_service_answers_from_approved_knowledge_records(db_session: Session) -> None:
    db_session.add(
        DecisionRecord(
            title='Use Redis for queues',
            decision_summary='Redis should power queue and job progress updates.',
            source_links=['https://knowledge.mock/redis-decision'],
            source_snippets=['Approved Redis decision snippet'],
            confidence_score=0.91,
            permission_level='internal',
            review_status='approved',
        )
    )
    db_session.commit()

    answer = answer_question_with_rag(db=db_session, user=USERS['viewer'], question='Redis queues')

    assert answer.answer
    assert answer.source_links == ['https://knowledge.mock/redis-decision']
    assert answer.source_snippets == ['Approved Redis decision snippet']
    assert answer.hidden_match_count == 0
    assert answer.permission_notice is None


def test_keyword_knowledge_candidates_share_exact_authoritative_text_for_all_types(
    db_session: Session,
) -> None:
    records = [
        DecisionRecord(
            title='Decision shared canonical marker',
            decision_summary='Decision authoritative body',
            source_links=['https://knowledge.mock/decision-candidate'],
            source_snippets=['Decision user snippet'],
            confidence_score=0.91,
            permission_level='internal',
            review_status='approved',
        ),
        HistoryEvent(
            title='History shared canonical marker',
            reason='History authoritative body',
            source_links=['https://knowledge.mock/history-candidate'],
            source_snippets=['History user snippet'],
            confidence_score=0.92,
            permission_level='internal',
            review_status='approved',
        ),
        TimelineEvent(
            title='Timeline shared canonical marker',
            result_summary='Timeline authoritative body',
            source_links=['https://knowledge.mock/timeline-candidate'],
            source_snippets=['Timeline user snippet'],
            confidence_score=0.93,
            permission_level='internal',
            review_status='approved',
        ),
        Todo(
            title='Todo shared canonical marker',
            priority='medium',
            priority_reason='Todo authoritative body',
            source_links=['https://knowledge.mock/todo-candidate'],
            source_snippets=['Todo user snippet'],
            confidence_score=0.94,
            permission_level='internal',
            review_status='approved',
        ),
    ]
    db_session.add_all(records)
    db_session.commit()

    candidates = retrieve_matching_knowledge_candidates(
        db=db_session,
        question='shared canonical marker',
    )

    assert {candidate.metadata['source_type']: candidate.text for candidate in candidates} == {
        'decision_record': (
            'Decision shared canonical marker\nDecision authoritative body'
        ),
        'history_event': (
            'History shared canonical marker\nHistory authoritative body'
        ),
        'timeline_event': (
            'Timeline shared canonical marker\nTimeline authoritative body'
        ),
        'todo': 'Todo shared canonical marker\nmedium\nTodo authoritative body',
    }
    assert {candidate.source_snippet for candidate in candidates} == {
        'Decision user snippet',
        'History user snippet',
        'Timeline user snippet',
        'Todo user snippet',
    }

    vector_store = InMemoryVectorStore()
    vector_store.upsert_many(vector_documents_from_candidates(candidates))
    answer = answer_question_with_rag(
        db=db_session,
        user=USERS['viewer'],
        question='shared canonical marker',
        vector_store=vector_store,
    )

    assert set(answer.source_ids) == {
        f'decision_record:{records[0].id}',
        f'history_event:{records[1].id}',
        f'timeline_event:{records[2].id}',
        f'todo:{records[3].id}',
    }
    assert set(answer.source_snippets) == {
        'Decision user snippet',
        'History user snippet',
        'Timeline user snippet',
        'Todo user snippet',
    }


def test_rag_service_hides_restricted_approved_knowledge_for_viewer(db_session: Session) -> None:
    db_session.add(
        Todo(
            title='Review confidential pricing',
            priority='high',
            priority_reason='Confidential pricing requires finance approval.',
            source_links=['https://knowledge.mock/restricted-pricing'],
            source_snippets=['Restricted pricing snippet'],
            confidence_score=0.8,
            permission_level='restricted',
            review_status='approved',
        )
    )
    db_session.commit()

    answer = answer_question_with_rag(db=db_session, user=USERS['viewer'], question='confidential pricing')

    assert answer.answer == '권한 내에서 확인 가능한 근거를 찾지 못했습니다.'
    assert answer.source_links == []
    assert answer.hidden_match_count == 1
    assert answer.permission_notice == 'Some sources may be hidden by permissions.'


def test_rag_service_can_answer_from_vector_store_matches(db_session: Session) -> None:
    text = (
        'Project Alpha launch history came from the indexed company memory '
        'vector store.'
    )
    seed_chunk(
        db_session,
        'gmail',
        'gmail-vector-alpha',
        text,
        'internal',
    )
    chunk = db_session.query(DocumentChunk).one()
    source = db_session.query(Source).one()
    vector_store = InMemoryVectorStore()
    vector_store.upsert(
        VectorDocument(
            document_id=source.source_id,
            text=text,
            source_url=source.source_url,
            source_snippet=chunk.source_snippet,
            permission_level='internal',
            metadata={
                'source_type': 'gmail',
                'chunk_id': chunk.id,
            },
        )
    )

    answer = answer_question_with_rag(
        db=db_session,
        user=USERS['viewer'],
        question='Project Alpha launch history',
        vector_store=vector_store,
    )

    assert answer.answer
    assert answer.source_links == ['https://gmail.mock/gmail-vector-alpha']
    assert answer.source_snippets == [text]
    assert answer.hidden_match_count == 0


def test_rag_service_uses_configured_stronger_primary_model(monkeypatch) -> None:
    captured = {}

    def fake_build_langchain_rag_orchestrator_model(settings):
        captured['primary'] = settings.openai_primary_model
        captured['fallback'] = settings.openai_fallback_model
        raise RuntimeError('stop after settings capture')

    monkeypatch.setattr(
        'backend.app.agents.rag_orchestrator_agent.service.build_langchain_rag_orchestrator_model',
        fake_build_langchain_rag_orchestrator_model,
    )

    with suppress(RuntimeError):
        build_default_rag_orchestrator_agent(
            Settings(
                paraworks_demo_mode=False,
                openai_api_key='test-key',
                agent_llm_enabled=True,
                agent_llm_openai_primary_model='gpt-5.4',
                agent_llm_openai_model='gpt-5.4-mini',
            )
        )

    assert captured == {
        'primary': 'gpt-5.4',
        'fallback': 'gpt-5.4-mini',
    }
