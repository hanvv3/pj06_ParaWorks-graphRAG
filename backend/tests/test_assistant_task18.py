"""Assistant cutover integration: real SQLite constraints, no provider calls."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
from sqlalchemy import select

from backend.app.agent_runtime.rag_application import RagApplicationFacade
from backend.app.assistant.service import create_conversation
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AgentRun,
    AgentRunCostComponent,
    AssistantMessage,
    RagServingCorpusGeneration,
)
from backend.tests import assistant_evidence_helpers as helpers
from backend.tests.test_assistant_capability import CAPABILITY_HEADERS


def enforce(client):
    from backend.app.api.v1 import assistant

    facade = client.app.state.rag_application_facade
    settings = facade._settings.model_copy(update={
        'langgraph_rag_v2_mode': 'enforce', 'langgraph_rag_v2_stage': 'assistant',
        'paraworks_env': 'test',
    })
    client.app.state.rag_application_facade = replace(facade, _settings=settings)
    client.app.dependency_overrides[assistant.get_settings] = lambda: settings
    with facade._session_factory() as db:
        if db.get(RagServingCorpusGeneration, 1) is None:
            from backend.app.admin.auto_review_keys import (
                fingerprint_key_material_verifier,
            )
            from backend.app.rag.serving_generation import (
                RAG_COSINE_POLICY_VERSION,
                RAG_INDEX_POLICY_VERSION,
            )
            db.add(RagServingCorpusGeneration(id=1, corpus_generation=0,
                embedding_model=settings.openai_embedding_model,
                embedding_dimensions=settings.openai_embedding_dimensions,
                index_policy_version=RAG_INDEX_POLICY_VERSION,
                pgvector_cosine_policy_version=RAG_COSINE_POLICY_VERSION,
                fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
                fingerprint_key_material_verifier=fingerprint_key_material_verifier(settings.agent_runtime_fingerprint_secret)))
        helpers.ensure_fingerprint_runtime(db, settings)
        db.commit()
    return client.app.state.rag_application_facade


def test_ordinary_writer_needs_no_reserved_id_or_flush_hook(db_session, monkeypatch):
    conversation = create_conversation(db_session, USERS['viewer'])
    monkeypatch.setattr(helpers, 'sqlite_writer_insert', lambda *a: nullcontext())
    message = helpers.write_canned(db_session, conversation, ' exact bytes ')
    assert message.content == ' exact bytes '
    assert message.assistant_message_content_hmac
    assert db_session.scalar(select(AssistantMessage.id)) == message.id


def test_enforce_assistant_uses_one_committed_v2_message(client, db_session, monkeypatch):
    enforce(client)
    captured = []
    original = RagApplicationFacade.invoke_graph
    def invoke(self, **kwargs):
        try:
            result = original(self, **kwargs)
            captured.append(result)
            return result
        except Exception as exc:
            captured.append(repr(exc))
            raise
    monkeypatch.setattr(RagApplicationFacade, 'invoke_graph', invoke)
    conversation = create_conversation(db_session, USERS['viewer'])
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'unmatched sentinel'}, headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 200, (response.text, captured)
    message_id = response.json()['assistant_message']['id']
    db_session.expire_all()
    message = db_session.get(AssistantMessage, message_id)
    assert message.content_write_mode == 'rag_v2_exact'
    assert len(list(db_session.scalars(select(AssistantMessage)))) == 2
    parent = db_session.get(AgentRun, message.agent_run_id)
    assert parent.run_record_phase == 'final'
    assert parent.run_contract_version == 'rag-run:v2'


def test_prior_context_invalid_refuses_before_user_row(client, db_session):
    enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    db_session.add(AssistantMessage(
        conversation_id=conversation.id, role='system', content='invalid prior role',
    ))
    db_session.commit()
    before = (conversation.title, conversation.updated_at, conversation.summary)
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 502, response.text
    db_session.expire_all()
    assert len(list(db_session.scalars(select(AssistantMessage)))) == 1
    assert db_session.query(AgentRun).count() == 0
    assert (conversation.title, conversation.updated_at, conversation.summary) == before


@pytest.mark.parametrize('refusal', ('application', 'registry', 'budget', 'readiness'))
def test_post_user_registry_failure_commits_safe_parent_and_message(
    client, db_session, monkeypatch, refusal,
):
    enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])

    def unavailable(*a, **k):
        from backend.app.agent_runtime.rag_application import RagApplicationError
        from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
        from backend.app.agent_runtime.rag_v2_registry import (
            RagRuntimeVersionUnavailableError,
        )
        from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
            QueryEmbeddingReadinessError,
        )
        if refusal == 'registry':
            raise RagRuntimeVersionUnavailableError('missing')
        if refusal == 'budget':
            raise RagBudgetExceededError()
        if refusal == 'readiness':
            raise QueryEmbeddingReadinessError('unavailable')
        raise RagApplicationError('runtime_version_unavailable')

    # Exercise the real affirmative preclaim boundary, not an unproven arbitrary
    # invoke_graph exception (which could equally be raised after commit).
    monkeypatch.setattr(type(client.app.state.rag_application_facade._graph_registry), 'resolve', unavailable)
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == (409 if refusal == 'budget' else 502)
    if refusal == 'budget':
        assert response.json() == {'detail': {'code': 'budget_exceeded'}}
    db_session.expire_all()
    rows = list(db_session.scalars(select(AssistantMessage).order_by(AssistantMessage.id)))
    assert len(rows) == 2
    assert rows[-1].content_write_mode == 'rag_v2_exact'
    assert rows[-1].metadata_['failure_class'] == 'RagV2SafeFailure'
    parent = db_session.get(AgentRun, rows[-1].agent_run_id)
    assert parent.status == 'failed' and parent.run_record_phase == 'final'
    assert db_session.query(AgentRunCostComponent).filter_by(agent_run_id=parent.id).count() == 2


@pytest.mark.parametrize('delivery_state', ('commit_unknown',))
def test_unknown_delivery_never_appends_second_message(
    client, db_session, monkeypatch, delivery_state,
):
    from backend.app.assistant.delivery import AssistantDeliveryResult

    enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    monkeypatch.setattr(RagApplicationFacade, 'invoke_assistant', lambda *a, **k:
        AssistantDeliveryResult('commit_unknown', None, 'reconciliation_required',
                                delivery_state, None, 500), raising=False)
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 500
    assert db_session.query(AssistantMessage).count() == 1
    assert db_session.query(AgentRun).count() == 0


def test_graph_retriever_failure_is_one_atomic_safe_failure(client, db_session, monkeypatch):
    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
    enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    def fail(*a, **k):
        raise RuntimeError('private retriever bytes')
    monkeypatch.setattr(KeywordEvidenceRetriever, 'invoke', fail)
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 502, response.text
    db_session.expire_all()
    assert db_session.query(AgentRun).count() == 1
    messages = list(db_session.scalars(select(AssistantMessage).order_by(AssistantMessage.id)))
    assert len(messages) == 2
    assert messages[-1].metadata_['failure_reason'] == 'retriever_unavailable'
    assert 'private retriever bytes' not in str(messages[-1].metadata_)
    assert messages[-1].content == '답변 생성 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.'
    parent = db_session.get(AgentRun, messages[-1].agent_run_id)
    assert parent.source_window == 'rag-v2:final-error:assistant:keyword'
    assert parent.cache_key.startswith('rag-v2-final-error:')
    assert parent.metadata_['terminal_identity_hmac'] == parent.cache_key.removeprefix('rag-v2-final-error:')


def test_forged_prepared_ingress_does_not_dispatch(client, db_session, monkeypatch):
    from backend.app.assistant.service import append_user_message
    facade = enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    prepared = facade.prepare_assistant_ingress(actor=USERS['viewer'],
        conversation_id=conversation.id, caller_text='question')
    user_message = append_user_message(db_session, USERS['viewer'], conversation, 'question')
    def forbidden(*a, **k):
        pytest.fail('forged ingress reached graph')
    monkeypatch.setattr(RagApplicationFacade, 'invoke_graph', forbidden)
    result = facade.invoke_assistant(actor=USERS['viewer'], conversation_id=conversation.id,
        user_message_id=user_message.id,
        prepared_ingress=replace(prepared, prior_context_hmac='0' * 64))
    assert result.public_status == 500
    assert db_session.query(AgentRun).count() == 0


def test_tampered_assistant_prior_refuses_without_new_user(client, db_session):
    enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    message = helpers.write_canned(db_session, conversation, 'safe answer')
    message.content = 'tampered answer'
    db_session.commit()
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 502
    assert db_session.query(AssistantMessage).count() == 1


@pytest.mark.parametrize('outcome', ('model_provider_failed', 'structured_output_invalid',
    'citation_validation_failed', 'provider_usage_overrun'))
@pytest.mark.parametrize('backend', ('keyword', 'pgvector'))
def test_projectionless_terminal_run_gets_one_safe_message(client, db_session, monkeypatch, outcome, backend):
    from datetime import UTC, datetime

    from backend.app.agent_runtime import rag_finalization
    from backend.app.agent_runtime.rag_finalization import assistant_terminal_zero_child
    payloads = []
    original_fingerprint = rag_finalization.keyed_fingerprint
    def fingerprint(payload, **kwargs):
        if kwargs.get('schema_version') == 'rag-result:v1':
            payloads.append(payload)
        return original_fingerprint(payload, **kwargs)
    monkeypatch.setattr(rag_finalization, 'keyed_fingerprint', fingerprint)
    facade = enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    def terminal(self, **kwargs):
        with self._session_factory() as db:
            parent = AgentRun(agent_name='rag_orchestrator_agent', prompt_version='rag-answer:v2',
                status='failed', run_contract_version='rag-run:v2', run_record_phase='final',
                completed_at=datetime.now(UTC), total_charged_cost_usd=0,
                source_window='rag-v2:final-error:assistant:keyword', cache_key='terminal',
                model_name='fake', permission_level='restricted',
                metadata_={'outcome': outcome, 'configured_backend': backend})
            db.add(parent)
            db.flush()
            db.add_all([assistant_terminal_zero_child(settings=facade._settings, parent_id=parent.id,
                component=c) for c in ('query_embedding', 'answer_generation')])
            parent_id = parent.id
            db.commit()
        return {'outcome': outcome, 'run_id': parent_id,
            'effective_backend': 'pgvector' if backend == 'pgvector' else 'deterministic_lexical',
            'fallback_category': None}
    monkeypatch.setattr(RagApplicationFacade, 'invoke_graph', terminal)
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 502
    db_session.expire_all()
    rows = list(db_session.scalars(select(AssistantMessage).order_by(AssistantMessage.id)))
    assert len(rows) == 2
    assert rows[-1].metadata_['failure_reason'] == outcome
    assert db_session.query(AgentRun).count() == 1
    assert db_session.query(AgentRunCostComponent).count() == 2
    assert payloads[-1]['effective_backend'] == ('pgvector' if backend == 'pgvector' else 'deterministic_lexical')


@pytest.mark.parametrize('failure_kind', ('generic', 'application', 'registry', 'budget', 'readiness'))
def test_post_commit_factory_exception_preserves_one_message(client, db_session, failure_kind):
    from contextlib import contextmanager
    facade = enforce(client)
    original = facade._request_factory
    committed = []
    def snapshot(db):
        parent = db.scalar(select(AgentRun))
        children = list(db.scalars(select(AgentRunCostComponent).order_by(AgentRunCostComponent.component_ordinal)))
        return (parent.id, parent.status, parent.run_record_phase, parent.cache_key,
            parent.source_window, parent.total_charged_cost_usd, dict(parent.metadata_),
            [(child.id, child.dispatch_state, child.dispatch_count, child.charged_cost_usd,
              child.charge_basis, child.authorized_model_config_snapshot_hmac) for child in children])
    @contextmanager
    def lose_ack(**kwargs):
        with original(**kwargs) as services:
            yield services
        with facade._session_factory() as db:
            committed.append(snapshot(db))
        if failure_kind == 'application':
            from backend.app.agent_runtime.rag_application import RagApplicationError
            raise RagApplicationError('runtime_version_unavailable')
        if failure_kind == 'registry':
            from backend.app.agent_runtime.rag_v2_registry import (
                RagRuntimeVersionUnavailableError,
            )
            raise RagRuntimeVersionUnavailableError('cleanup')
        if failure_kind == 'budget':
            from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
            raise RagBudgetExceededError()
        if failure_kind == 'readiness':
            from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
                QueryEmbeddingReadinessError,
            )
            raise QueryEmbeddingReadinessError('cleanup')
        raise RuntimeError('private post-commit bytes')
    client.app.state.rag_application_facade = replace(facade, _request_factory=lose_ack)
    conversation = create_conversation(db_session, USERS['viewer'])
    response = client.post(f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS)
    assert response.status_code == 500
    db_session.expire_all()
    assert db_session.query(AgentRun).count() == 1
    assert db_session.query(AssistantMessage).count() == 2
    assert snapshot(db_session) == committed[0]


def test_failure_after_admission_before_state_publish_never_authorizes_new_parent(client, db_session, monkeypatch):
    from backend.app.agent_runtime.rag_application import RagApplicationError
    from backend.app.agent_runtime.rag_sqlite_smoke import SQLiteRagGraphScope
    enforce(client)
    original = SQLiteRagGraphScope.admit
    def admit_then_fail(self, **kwargs):
        original(self, **kwargs)
        raise RagApplicationError('runtime_version_unavailable')
    monkeypatch.setattr(SQLiteRagGraphScope, 'admit', admit_then_fail)
    conversation = create_conversation(db_session, USERS['viewer'])
    response = client.post(f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'question'}, headers=CAPABILITY_HEADERS)
    assert response.status_code == 500
    assert db_session.query(AgentRun).count() == 0
    assert db_session.query(AssistantMessage).count() == 1


@pytest.mark.parametrize('mode', ('disabled', 'enforce'))
@pytest.mark.parametrize('action', ('contact', 'email', 'email_fallthrough'))
def test_stale_rag_prior_is_not_independent_action_authority(client, db_session, monkeypatch, mode, action):
    from backend.app.api.v1 import assistant
    from backend.tests.test_assistant_api import _email_intent, _patch_email_flow
    if mode == 'enforce':
        enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    message = helpers.write_canned(db_session, conversation, 'old safe answer')
    message.content = 'unavailable prior must not enter any model'
    db_session.commit()
    def compose(**kwargs):
        assert 'unavailable prior' not in str(kwargs)
        return assistant.EmailActionDecision(action_type='not_email' if action == 'email_fallthrough' else 'email_draft',
            to=['partner@example.com'], subject='회의 취소 안내', body='오늘 회의가 취소되었습니다.')
    _patch_email_flow(monkeypatch, intent_decision=_email_intent(email_intent=True), draft_decision=compose)
    text = '김종우님 이메일 알려줘.' if action == 'contact' else 'partner@example.com에 오늘 회의 취소됐다고 메일 보내줘.'
    response = client.post(f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': text}, headers=CAPABILITY_HEADERS)
    if mode == 'enforce' and action == 'email_fallthrough':
        assert response.status_code == 502
        assert db_session.query(AssistantMessage).count() == 1
    else:
        assert response.status_code == 200, response.text
        if action != 'email_fallthrough':
            assert response.json()['assistant_message']['metadata']['action_type'] == ('contact_lookup' if action == 'contact' else 'email_draft')


@pytest.mark.parametrize('mode', ('disabled', 'enforce'))
def test_recipient_correction_uses_eligible_draft_not_stale_rag_prior(client, db_session, mode):
    from backend.app.assistant.service import append_assistant_message
    if mode == 'enforce':
        enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    stale = helpers.write_canned(db_session, conversation, 'old safe answer')
    stale.content = 'tampered unrelated RAG answer'
    db_session.commit()
    append_assistant_message(db_session, USERS['viewer'], conversation,
        content='메일 초안', citations=[], source_ids=[], source_links=[], source_snippets=[],
        permission_level=None, hidden_match_count=0, permission_notice=None, agent_run_id=None,
        metadata={'action_type': 'email_draft', 'status': 'pending_approval',
            'email_draft': {'to': ['partner@example.com'], 'subject': '회의', 'body': '오늘 회의 취소'}})
    response = client.post(f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': '수신자가 잘못됐어요.'}, headers=CAPABILITY_HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()['assistant_message']['metadata']['reason'] == 'recipient_correction_requested'
    assert db_session.query(AssistantMessage).count() == 4


def test_real_writer_rollback_removes_allocated_row(db_session):
    from datetime import UTC, datetime

    from backend.app.assistant.evidence_persistence import (
        AssistantEvidenceWriter,
        AssistantMessageProjection,
        _mint_assistant_exact_write_authority,
    )
    from backend.app.core.config import get_settings
    from backend.tests.test_assistant_evidence_writer_v2 import _empty_projection
    conversation = create_conversation(db_session, USERS['viewer'])
    settings = get_settings()
    parent = AgentRun(agent_name='rag_orchestrator_agent', prompt_version='rag-answer:v2',
        status='complete', run_contract_version='rag-run:v2', run_record_phase='final',
        completed_at=datetime.now(UTC), total_charged_cost_usd=0, source_window='test',
        cache_key='test', model_name='fake', permission_level='internal', metadata_={})
    db_session.add(parent)
    db_session.flush()
    projection = AssistantMessageProjection(content=' exact ', metadata={}, evidence=_empty_projection(),
        permission_level=None, permission_notice=None, hidden_match_count=0,
        assembled_answer_hmac=None, canned_message_identity='rag-canned-no-evidence:v1', result_hmac='a'*64)
    writer = AssistantEvidenceWriter(fingerprint_secret=settings.agent_runtime_fingerprint_secret.encode(),
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version, settings=settings)
    writer.append_final(db=db_session, conversation=conversation, projection=projection,
        authority=_mint_assistant_exact_write_authority(parent_agent_run_id=parent.id,
            conversation_id=conversation.id, projection=projection, parent_result_hmac=projection.result_hmac))
    db_session.rollback()
    assert db_session.query(AgentRun).count() == 0
    assert db_session.query(AssistantMessage).count() == 0


def test_legacy_projection_only_writer_cannot_create_unsigned_evidence(db_session):
    from backend.app.assistant.evidence_persistence import (
        AssistantEvidenceWriter,
        LegacyAssistantMessageProjection,
    )
    from backend.app.core.config import get_settings
    from backend.tests.test_assistant_evidence_writer_v2 import _empty_projection
    settings = get_settings()
    conversation = create_conversation(db_session, USERS['viewer'])
    writer = AssistantEvidenceWriter(fingerprint_secret=settings.agent_runtime_fingerprint_secret.encode(),
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version, settings=settings)
    projection = LegacyAssistantMessageProjection(content='unbound evidence', metadata={},
        evidence=replace(_empty_projection(), source_ids=('source',)))
    with pytest.raises(ValueError, match='snapshot'):
        writer.append_legacy_evidence(db=db_session, conversation=conversation, projection=projection)
    assert db_session.query(AssistantMessage).count() == 0


def test_finalization_failure_returns_only_current_failed_parent_proof(client, db_session, monkeypatch):
    from datetime import UTC, datetime

    from backend.app.agent_runtime.rag_finalization import RagFinalizationError
    from backend.app.assistant.service import append_user_message
    facade = enforce(client)
    conversation = create_conversation(db_session, USERS['viewer'])
    ingress = facade.prepare_assistant_ingress(actor=USERS['viewer'],
        conversation_id=conversation.id, caller_text='question')
    user_message = append_user_message(db_session, USERS['viewer'], conversation, 'question')
    parent = AgentRun(agent_name='rag_orchestrator_agent', prompt_version='rag-answer:v2',
        status='failed', run_contract_version='rag-run:v2', run_record_phase='final',
        completed_at=datetime.now(UTC), total_charged_cost_usd=0,
        source_window='rag-v2:final-error:assistant:keyword', cache_key='recovered',
        model_name='fake', permission_level='restricted', metadata_={'outcome': 'persistence_failed'})
    db_session.add(parent)
    db_session.commit()
    def lost_product(*a, **k):
        failure = RagFinalizationError('failed message transaction')
        failure.run_id = parent.id
        raise failure
    monkeypatch.setattr(RagApplicationFacade, 'invoke_graph', lost_product)
    result = facade.invoke_assistant(actor=USERS['viewer'], conversation_id=conversation.id,
        user_message_id=user_message.id, prepared_ingress=ingress)
    assert result.delivery_state == 'committed_run_failure'
    assert result.parent_agent_run_id == parent.id and result.assistant_message_id is None
    assert db_session.query(AssistantMessage).count() == 1
    assert db_session.query(AgentRun).count() == 1


@pytest.mark.parametrize('kind', ('raw', 'legacy_knowledge'))
def test_disabled_evidence_write_is_keyed_and_still_readable(client, db_session, kind):
    from backend.app.core.config import get_settings
    from backend.app.models import AssistantMessageEvidenceDependency, DecisionRecord
    from backend.tests.test_rag_orchestrator_service import seed_chunk
    if kind == 'raw':
        seed_chunk(db_session, 'gmail', 'legacy-source', 'Redis durable storage decision', 'internal')
    else:
        helpers.ensure_fingerprint_runtime(db_session, get_settings())
        db_session.add(DecisionRecord(title='Redis decision', decision_summary='Redis durable storage decision',
            source_links=['https://example.test/legacy'], source_snippets=['Redis durable storage decision'],
            permission_level='internal', review_status='approved', confidence_score=1))
        db_session.commit()
    conversation = create_conversation(db_session, USERS['viewer'])
    response = client.post(f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'Redis'}, headers={'X-Demo-User': 'viewer'})
    assert response.status_code == 200, response.text
    body = response.json()['assistant_message']
    assert body['citations']
    db_session.expire_all()
    message = db_session.get(AssistantMessage, body['id'])
    assert message.content_origin == 'legacy_evidence'
    assert message.dependency_set_hmac_schema_version == 'assistant-dependency-set-hmac:v2'
    child = db_session.scalar(select(AssistantMessageEvidenceDependency).where(
        AssistantMessageEvidenceDependency.assistant_message_id == message.id))
    assert child.dependency_serving_scope == 'legacy_v1_only'
    assert child.serving_identity_hmac is None and child.serving_version_fingerprint is None
    fetched = client.get(f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers={'X-Demo-User': 'viewer'})
    assert fetched.json()['messages'][-1]['content'] == body['content']


@pytest.mark.parametrize('existing', (False, True))
def test_completed_facade_scanner_outage_is_zero_mutation(client, db_session, monkeypatch, existing):
    from backend.tests.test_assistant_capability import (
        test_scanner_unavailable_existing_turn_preserves_rows_and_skips_provider,
        test_scanner_unavailable_first_turn_is_generic_500_before_any_insert,
    )
    enforce(client)
    test = (test_scanner_unavailable_existing_turn_preserves_rows_and_skips_provider
            if existing else test_scanner_unavailable_first_turn_is_generic_500_before_any_insert)
    test(client, db_session, monkeypatch)


def test_legacy_signature_records_actual_vector_backend(client, db_session, monkeypatch):
    from backend.app.agents.rag_orchestrator_agent.service import (
        retrieve_matching_evidence_candidates,
        vector_documents_from_candidates,
    )
    from backend.app.api.v1 import assistant
    from backend.app.rag.vector_store import InMemoryVectorStore
    def fake_store(*, db, settings):
        store = InMemoryVectorStore()
        store.upsert_many(vector_documents_from_candidates(
            retrieve_matching_evidence_candidates(db=db, question='Redis')))
        return store
    monkeypatch.setattr(assistant, 'build_pgvector_search_store', fake_store)
    test_disabled_evidence_write_is_keyed_and_still_readable(client, db_session, 'raw')
    message = db_session.scalar(select(AssistantMessage).where(AssistantMessage.role == 'assistant'))
    assert message.metadata_['effective_backend'] == 'pgvector'


@pytest.mark.parametrize('kind', ('raw', 'legacy_knowledge'))
@pytest.mark.parametrize('drift', ('content', 'permission', 'revoke'))
def test_production_signed_legacy_drift_blocks_get_and_prior_ingress(
    client, db_session, kind, drift,
):
    from backend.app.assistant.evidence_reader import UNAVAILABLE_CONTENT
    from backend.app.models import DecisionRecord, DocumentChunk, ReviewItem, Source
    test_disabled_evidence_write_is_keyed_and_still_readable(client, db_session, kind)
    message = db_session.scalar(select(AssistantMessage).where(AssistantMessage.role == 'assistant'))
    conversation_id = message.conversation_id
    if kind == 'raw':
        if drift == 'content':
            db_session.scalar(select(DocumentChunk)).text = 'changed current bytes'
        elif drift == 'permission':
            db_session.scalar(select(Source)).permission_level = 'restricted'
        else:
            db_session.scalar(select(ReviewItem)).status = 'rejected'
    else:
        knowledge = db_session.scalar(select(DecisionRecord))
        if drift == 'content':
            knowledge.decision_summary = 'changed current bytes'
        elif drift == 'permission':
            knowledge.permission_level = 'restricted'
        else:
            knowledge.review_status = 'rejected'
    db_session.commit()
    fetched = client.get(f'/api/v1/assistant/conversations/{conversation_id}/messages',
        headers={'X-Demo-User': 'viewer'})
    assert fetched.json()['messages'][-1]['content'] == UNAVAILABLE_CONTENT
    before = (db_session.query(AssistantMessage).count(), db_session.query(AgentRun).count())
    enforce(client)
    response = client.post(f'/api/v1/assistant/conversations/{conversation_id}/messages',
        json={'content': 'follow up'}, headers=CAPABILITY_HEADERS)
    assert response.status_code == 502
    assert (db_session.query(AssistantMessage).count(), db_session.query(AgentRun).count()) == before


@pytest.mark.parametrize('failure', (None, 'unexpected'))
def test_real_graph_supported_or_generation_failure_keeps_single_product(
    client, db_session, monkeypatch, failure,
):
    from backend.app.agent_runtime.rag_sqlite_smoke import SQLiteRagGraphScope
    from backend.tests.test_rag_v2_keyword_retriever import (
        _seed_sqlite_raw_projection,
        _settings,
    )
    settings = _settings()
    monkeypatch.setenv('AGENT_RUNTIME_FINGERPRINT_SECRET', settings.agent_runtime_fingerprint_secret)
    monkeypatch.setenv('AGENT_RUNTIME_FINGERPRINT_KEY_VERSION', settings.agent_runtime_fingerprint_key_version)
    from backend.app.core.config import get_settings
    get_settings.cache_clear()
    _seed_sqlite_raw_projection(db_session)
    facade = client.app.state.rag_application_facade
    client.app.state.rag_application_facade = replace(facade, _settings=settings)
    enforce(client)
    errors = []
    original = RagApplicationFacade.invoke_graph
    def observe(self, **kwargs):
        try:
            return original(self, **kwargs)
        except Exception as exc:
            errors.append(repr(exc))
            raise
    monkeypatch.setattr(RagApplicationFacade, 'invoke_graph', observe)
    if failure:
        def fail(*a, **k):
            raise RuntimeError('private model bytes')
        monkeypatch.setattr(SQLiteRagGraphScope, 'generate_deterministic_answer', fail)
    conversation = create_conversation(db_session, USERS['viewer'])
    response = client.post(f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'Exact raw observation'}, headers=CAPABILITY_HEADERS)
    assert response.status_code == (502 if failure else 200), (response.text, errors)
    db_session.expire_all()
    rows = list(db_session.scalars(select(AssistantMessage).order_by(AssistantMessage.id)))
    assert len(rows) == 2 and db_session.query(AgentRun).count() == 1
    if failure:
        assert rows[-1].metadata_['failure_reason'] == 'unexpected_internal_error'
    else:
        assert rows[-1].content_origin == 'rag_assembled'
        assert response.json()['assistant_message']['content'] == rows[-1].content
        assert response.json()['assistant_message']['citations']
