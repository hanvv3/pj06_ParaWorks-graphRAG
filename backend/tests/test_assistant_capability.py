import json
from typing import Annotated

import pytest
from fastapi import Depends, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.agents.rag_orchestrator_agent.v2_input import (
    RagInputScannerUnavailableError,
)
from backend.app.api.v1 import assistant as assistant_api
from backend.app.assistant.capability import (
    RAG_RENDER_CAPABILITY_HEADER,
    RAG_RENDER_CAPABILITY_VALUE,
    require_rag_render_capability,
)
from backend.app.core import demo_auth
from backend.app.core.config import Settings
from backend.app.models import AgentRun, AssistantConversation, AssistantMessage

CAPABILITY_HEADERS = {
    'X-ParaWorks-Rag-Render-Capability': 'rag-v2-plain-text-citations:v1',
    'X-Demo-User': 'viewer',
}
NO_STORE_HEADERS = {
    'cache-control': 'private, no-store',
    'vary': 'X-ParaWorks-Rag-Render-Capability',
}


def _enforce_assistant(client: TestClient) -> None:
    client.app.dependency_overrides[assistant_api.get_settings] = lambda: Settings(
        paraworks_demo_mode=True,
        langgraph_rag_v2_mode='enforce',
        langgraph_rag_v2_stage='assistant',
    )


def _assert_no_store(response) -> None:
    assert response.headers['cache-control'] == NO_STORE_HEADERS['cache-control']
    assert response.headers['vary'] == NO_STORE_HEADERS['vary']


@pytest.mark.parametrize(
    'raw_headers',
    [
        (),
        ((RAG_RENDER_CAPABILITY_HEADER, b'wrong'),),
        (
            (RAG_RENDER_CAPABILITY_HEADER, RAG_RENDER_CAPABILITY_VALUE),
            (RAG_RENDER_CAPABILITY_HEADER, RAG_RENDER_CAPABILITY_VALUE),
        ),
        (
            (
                RAG_RENDER_CAPABILITY_HEADER,
                RAG_RENDER_CAPABILITY_VALUE + b', rag-v2-plain-text-citations:v1',
            ),
        ),
    ],
)
def test_raw_asgi_render_capability_rejects_missing_duplicate_folded_or_wrong(
    raw_headers: tuple[tuple[bytes, bytes], ...],
) -> None:
    with pytest.raises(HTTPException) as caught:
        require_rag_render_capability(raw_headers)
    assert caught.value.status_code == 409
    assert caught.value.detail == {'code': 'client_upgrade_required'}


def test_raw_asgi_render_capability_accepts_exactly_one_exact_declaration() -> None:
    require_rag_render_capability(
        (
            (b'x-demo-user', b'viewer'),
            (RAG_RENDER_CAPABILITY_HEADER, RAG_RENDER_CAPABILITY_VALUE),
        )
    )


def test_first_turn_valid_body_checks_capability_before_conversation_creation(
    client: TestClient,
    db_session: Session,
) -> None:
    _enforce_assistant(client)
    response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'first turn sentinel'},
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 409
    assert response.json() == {'detail': {'code': 'client_upgrade_required'}}
    assert db_session.query(AssistantConversation).count() == 0
    _assert_no_store(response)


def test_authentication_precedes_body_and_capability_validation(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enforce_assistant(client)
    monkeypatch.setattr(
        demo_auth,
        'get_settings',
        lambda: Settings(paraworks_demo_mode=False),
    )
    response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'x' * 161},
    )
    assert response.status_code == 401
    assert response.json() == {'detail': 'Authentication required.'}
    assert db_session.query(AssistantConversation).count() == 0
    _assert_no_store(response)


@pytest.mark.parametrize(
    'path',
    (
        '/api/v1/assistant/conversations',
        '/api/v1/assistant/conversations/999999/messages',
    ),
)
def test_unauthenticated_malformed_json_is_401_before_body_or_owner_lookup(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    _enforce_assistant(client)
    monkeypatch.setattr(
        demo_auth,
        'get_settings',
        lambda: Settings(paraworks_demo_mode=False),
    )

    response = client.post(
        path,
        content=b'{',
        headers={'Content-Type': 'application/json'},
    )

    assert response.status_code == 401
    assert response.json() == {'detail': 'Authentication required.'}
    assert db_session.query(AssistantConversation).count() == 0
    assert db_session.query(AssistantMessage).count() == 0
    assert db_session.query(AgentRun).count() == 0
    _assert_no_store(response)


@pytest.mark.parametrize(
    'path',
    (
        '/api/v1/assistant/conversations',
        '/api/v1/assistant/conversations/999999/messages',
    ),
)
def test_authenticated_malformed_json_is_422_with_capability_headers(
    client: TestClient,
    db_session: Session,
    path: str,
) -> None:
    _enforce_assistant(client)

    response = client.post(
        path,
        content=b'{',
        headers={'Content-Type': 'application/json', 'X-Demo-User': 'viewer'},
    )

    assert response.status_code == 422
    assert isinstance(response.json()['detail'], list)
    assert db_session.query(AssistantConversation).count() == 0
    assert db_session.query(AssistantMessage).count() == 0
    assert db_session.query(AgentRun).count() == 0
    _assert_no_store(response)


def test_pre_body_auth_reuses_override_with_one_db_session_and_auth_call(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {'db': 0, 'db_closed': 0, 'user': 0, 'settings': 0}
    settings = Settings(
        paraworks_demo_mode=True,
        langgraph_rag_v2_mode='enforce',
        langgraph_rag_v2_stage='assistant',
    )
    capability_setting_ids: list[int] = []

    def override_db():
        calls['db'] += 1
        try:
            yield db_session
        finally:
            calls['db_closed'] += 1

    def override_user(
        db: Annotated[Session, Depends(assistant_api.get_db)],
    ):
        calls['user'] += 1
        assert db is db_session
        return demo_auth.USERS['viewer']

    def override_settings() -> Settings:
        calls['settings'] += 1
        return settings

    def require_capability(value: Settings) -> bool:
        capability_setting_ids.append(id(value))
        return True

    client.app.dependency_overrides[assistant_api.get_db] = override_db
    client.app.dependency_overrides[assistant_api.get_demo_user] = override_user
    client.app.dependency_overrides[assistant_api.get_settings] = override_settings
    monkeypatch.setattr(
        assistant_api,
        'assistant_post_requires_render_capability',
        require_capability,
    )

    response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'dependency identity'},
        headers=CAPABILITY_HEADERS,
    )

    assert response.status_code == 200
    assert calls['db'] == 1
    assert calls['db_closed'] == 1
    assert calls['user'] == 1
    assert calls['settings'] >= 1
    assert capability_setting_ids == [id(settings), id(settings)]
    conversation = db_session.query(AssistantConversation).one()
    assert conversation.user_id == demo_auth.USERS['viewer'].id
    _assert_no_store(response)


def test_malformed_first_turn_body_precedes_capability_guard(
    client: TestClient,
    db_session: Session,
) -> None:
    _enforce_assistant(client)
    response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'x' * 161},
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 422
    assert db_session.query(AssistantConversation).count() == 0
    _assert_no_store(response)


def test_strict_unicode_validation_precedes_capability_guard(
    client: TestClient,
    db_session: Session,
) -> None:
    _enforce_assistant(client)
    response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'unsafe\x00title'},
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 422
    assert db_session.query(AssistantConversation).count() == 0
    _assert_no_store(response)


@pytest.mark.parametrize(
    ('path', 'field'),
    (
        ('/api/v1/assistant/conversations', 'title'),
        ('/api/v1/assistant/conversations/999999/messages', 'content'),
    ),
)
def test_escaped_surrogate_is_safe_422_before_writes_owner_or_provider(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    field: str,
) -> None:
    provider_calls = 0

    def provider_must_not_run(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError('provider ran for invalid Unicode')

    monkeypatch.setattr(
        assistant_api, 'answer_question_with_rag', provider_must_not_run
    )
    _enforce_assistant(client)
    response = client.post(
        path,
        content=json.dumps({field: '\ud800'}),
        headers={
            'Content-Type': 'application/json',
            **CAPABILITY_HEADERS,
        },
    )

    assert response.status_code == 422
    detail = response.json()['detail']
    assert isinstance(detail, list)
    assert detail[0]['loc'] == ['body', field]
    assert detail[0]['input'] == '\ud800'
    assert db_session.query(AssistantConversation).count() == 0
    assert db_session.query(AssistantMessage).count() == 0
    assert db_session.query(AgentRun).count() == 0
    assert provider_calls == 0
    _assert_no_store(response)


@pytest.mark.parametrize('raise_server_exceptions', (False, True))
def test_unexpected_failure_reaches_dependency_unwind_before_sanitized_500(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    raise_server_exceptions: bool,
) -> None:
    events: list[str] = []

    def override_db():
        events.append('open')
        try:
            yield db_session
        except RuntimeError:
            events.append('exception_seen')
            raise
        else:
            events.append('normal_exit')
        finally:
            events.append('closed')

    client.app.dependency_overrides[assistant_api.get_db] = override_db
    _enforce_assistant(client)
    monkeypatch.setattr(
        assistant_api,
        'create_conversation',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('synthetic')),
    )
    client._transport.raise_server_exceptions = raise_server_exceptions

    def request():
        return client.post(
            '/api/v1/assistant/conversations',
            json={'title': 'persistence exception'},
            headers=CAPABILITY_HEADERS,
        )

    if raise_server_exceptions:
        with pytest.raises(RuntimeError, match='synthetic'):
            request()
    else:
        response = request()
        assert response.status_code == 500
        assert response.text == 'Internal Server Error'
        _assert_no_store(response)

    assert events == ['open', 'exception_seen', 'closed']
    assert db_session.query(AssistantConversation).count() == 0


def test_email_send_error_remains_outside_render_capability_headers(
    client: TestClient,
) -> None:
    _enforce_assistant(client)

    response = client.post(
        '/api/v1/assistant/messages/999999/email/send',
        headers=CAPABILITY_HEADERS,
    )

    assert response.status_code == 404
    assert 'cache-control' not in response.headers
    assert 'vary' not in response.headers


def test_owner_hidden_message_post_precedes_capability_guard(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='owned')
    db_session.add(conversation)
    db_session.commit()
    _enforce_assistant(client)

    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'valid content'},
        headers={'X-Demo-User': 'hanvv-employee'},
    )
    assert response.status_code == 404
    assert response.json() == {'detail': 'assistant conversation not found'}
    assert db_session.query(AssistantMessage).count() == 0
    _assert_no_store(response)


def test_message_body_validation_precedes_owner_lookup_and_capability(
    client: TestClient,
    db_session: Session,
) -> None:
    _enforce_assistant(client)
    response = client.post(
        '/api/v1/assistant/conversations/999999/messages',
        json={'content': 'x' * 4001},
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 422
    assert db_session.query(AssistantMessage).count() == 0
    _assert_no_store(response)


def test_existing_conversation_capability_guard_precedes_message_mutation(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='owned')
    db_session.add(conversation)
    db_session.commit()
    _enforce_assistant(client)

    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'valid content'},
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 409
    assert response.json() == {'detail': {'code': 'client_upgrade_required'}}
    assert db_session.query(AssistantMessage).count() == 0
    _assert_no_store(response)


def test_capability_dependent_message_post_success_keeps_legacy_writer_contract(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='contact')
    db_session.add(conversation)
    db_session.commit()
    _enforce_assistant(client)
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': '김종우님 이메일 알려줘.'},
        headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 200
    assert (
        response.json()['assistant_message']['metadata']['action_type']
        == 'contact_lookup'
    )
    assert db_session.query(AssistantMessage).count() == 2
    _assert_no_store(response)


def test_scanner_unavailable_first_turn_is_generic_500_before_any_insert(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enforce_assistant(client)
    monkeypatch.setattr(
        assistant_api,
        '_scan_value',
        lambda _value: (_ for _ in ()).throw(RagInputScannerUnavailableError()),
    )
    response = client.post(
        '/api/v1/assistant/conversations',
        json={'title': 'scanner readiness'},
        headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 500
    assert response.json() == {'detail': 'assistant request failed'}
    assert db_session.query(AssistantConversation).count() == 0
    assert db_session.query(AssistantMessage).count() == 0
    assert db_session.query(AgentRun).count() == 0
    _assert_no_store(response)


def test_scanner_unavailable_existing_turn_preserves_rows_and_skips_provider(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='owned')
    db_session.add(conversation)
    db_session.commit()
    db_session.refresh(conversation)
    original_updated_at = conversation.updated_at
    provider_calls = 0

    def provider_must_not_run(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError('provider path ran before scanner readiness')

    monkeypatch.setattr(
        assistant_api, 'answer_question_with_rag', provider_must_not_run
    )
    monkeypatch.setattr(
        assistant_api,
        '_scan_value',
        lambda _value: (_ for _ in ()).throw(RagInputScannerUnavailableError()),
    )
    _enforce_assistant(client)
    response = client.post(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        json={'content': 'scanner readiness'},
        headers=CAPABILITY_HEADERS,
    )
    db_session.refresh(conversation)
    assert response.status_code == 500
    assert response.json() == {'detail': 'assistant request failed'}
    assert conversation.updated_at == original_updated_at
    assert db_session.query(AssistantMessage).count() == 0
    assert db_session.query(AgentRun).count() == 0
    assert provider_calls == 0
    _assert_no_store(response)


def _add_v2_canned_message(
    db: Session,
    conversation: AssistantConversation,
    content: str,
) -> AssistantMessage:
    from backend.tests.assistant_evidence_helpers import write_canned

    return write_canned(db, conversation, content)


def _add_stale_v2_message(
    db: Session,
    conversation: AssistantConversation,
    content: str,
) -> AssistantMessage:
    message = _add_v2_canned_message(db, conversation, content)
    message.content_origin = 'rag_assembled'
    message.evidence_contract_version = 'assistant-evidence:v1'
    message.serving_dependency_count = 1
    message.dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2'
    message.dependency_set_hmac = '1' * 64
    message.parent_selected_evidence_projection_hmac = '2' * 64
    message.model_influence_set_hmac = '3' * 64
    db.commit()
    return message


def _add_legacy_message(
    db: Session,
    conversation: AssistantConversation,
    content: str,
) -> AssistantMessage:
    message = AssistantMessage(
        conversation_id=conversation.id,
        role='assistant',
        content=content,
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        hidden_match_count=0,
        evidence_contract_version='none-v1',
        serving_dependency_count=0,
        metadata_={'prompt_version': 'rag-answer:v1'},
    )
    db.add(message)
    db.commit()
    return message


def test_message_get_refuses_live_v2_bytes_without_capability(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='live v2')
    db_session.add(conversation)
    db_session.commit()
    _add_v2_canned_message(db_session, conversation, 'LIVE_V2_SENTINEL')

    response = client.get(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 409
    assert response.json() == {'detail': {'code': 'client_upgrade_required'}}
    assert b'LIVE_V2_SENTINEL' not in response.content
    _assert_no_store(response)


def test_message_get_allows_live_v2_projection_with_capability(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='live v2')
    db_session.add(conversation)
    db_session.commit()
    _add_v2_canned_message(db_session, conversation, 'LIVE_V2_SENTINEL')

    response = client.get(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers=CAPABILITY_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()['messages'][0]['content'] == 'LIVE_V2_SENTINEL'
    _assert_no_store(response)


def test_message_get_does_not_require_capability_for_stale_redacted_v2(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='stale v2')
    db_session.add(conversation)
    db_session.commit()
    _add_stale_v2_message(db_session, conversation, 'STALE_V2_SENTINEL')

    response = client.get(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 200
    assert b'STALE_V2_SENTINEL' not in response.content
    assert response.json()['messages'][0]['permission_notice'] == 'evidence_unavailable'
    assert 'cache-control' not in response.headers
    assert 'vary' not in response.headers


def test_missing_parent_v2_is_redacted_before_message_guard_and_summary(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(
        user_id='employee-mina', title='missing parent'
    )
    db_session.add(conversation)
    db_session.commit()
    message = _add_v2_canned_message(db_session, conversation, 'ORPHANED_V2_SENTINEL')
    linked_agent_run_id = message.linked_agent_run_id
    # Corrupt the actual SQLite authority, not one Session's identity-map method.
    # The reader uses a separate identity map by design.
    db_session.execute(
        AgentRun.__table__.delete().where(AgentRun.id == linked_agent_run_id)
    )
    db_session.commit()

    messages_response = client.get(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )
    conversations_response = client.get(
        '/api/v1/assistant/conversations',
        headers={'X-Demo-User': 'viewer'},
    )

    assert messages_response.status_code == 200
    assert b'ORPHANED_V2_SENTINEL' not in messages_response.content
    assert messages_response.json()['messages'][0]['permission_notice'] == (
        'evidence_unavailable'
    )
    assert 'cache-control' not in messages_response.headers
    assert conversations_response.status_code == 200
    assert b'ORPHANED_V2_SENTINEL' not in conversations_response.content
    assert 'cache-control' not in conversations_response.headers


def test_hidden_only_canned_v2_requires_guard_and_preserves_original_projection(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='hidden v2')
    db_session.add(conversation)
    db_session.commit()
    message = _add_v2_canned_message(db_session, conversation, 'HIDDEN_V2_SENTINEL')
    message.hidden_match_count = 1
    message.permission_notice = 'Some sources may be hidden by permissions.'
    parent = db_session.get(AgentRun, message.agent_run_id)
    parent.metadata_ = {
        **parent.metadata_,
        'hidden_match_count': 1,
        'outcome': 'hidden_only',
    }
    db_session.commit()

    refused_messages = client.get(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )
    refused_conversations = client.get(
        '/api/v1/assistant/conversations',
        headers={'X-Demo-User': 'viewer'},
    )
    messages_response = client.get(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers=CAPABILITY_HEADERS,
    )
    conversations_response = client.get(
        '/api/v1/assistant/conversations',
        headers=CAPABILITY_HEADERS,
    )

    assert refused_messages.status_code == 409
    assert b'HIDDEN_V2_SENTINEL' not in refused_messages.content
    _assert_no_store(refused_messages)
    assert refused_conversations.status_code == 409
    assert b'HIDDEN_V2_SENTINEL' not in refused_conversations.content
    _assert_no_store(refused_conversations)
    assert messages_response.status_code == 200
    projected = messages_response.json()['messages'][0]
    assert projected['content'] == 'HIDDEN_V2_SENTINEL'
    assert projected['hidden_match_count'] == 1
    assert projected['permission_notice'] == (
        'Some sources may be hidden by permissions.'
    )
    _assert_no_store(messages_response)
    assert conversations_response.status_code == 200
    assert conversations_response.json()['conversations'][0]['summary'] == (
        'HIDDEN_V2_SENTINEL'
    )
    _assert_no_store(conversations_response)


def test_message_get_ignores_live_v2_row_in_unrelated_conversation(
    client: TestClient,
    db_session: Session,
) -> None:
    legacy = AssistantConversation(user_id='employee-mina', title='legacy')
    unrelated = AssistantConversation(user_id='employee-mina', title='unrelated')
    db_session.add_all([legacy, unrelated])
    db_session.commit()
    _add_legacy_message(db_session, legacy, 'LEGACY_SENTINEL')
    _add_v2_canned_message(db_session, unrelated, 'UNRELATED_V2_SENTINEL')

    response = client.get(
        f'/api/v1/assistant/conversations/{legacy.id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 200
    assert response.json()['messages'][0]['content'] == 'LEGACY_SENTINEL'
    assert b'UNRELATED_V2_SENTINEL' not in response.content
    assert 'cache-control' not in response.headers


def test_conversation_list_requires_capability_when_live_v2_contributes_summary(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='live v2')
    db_session.add(conversation)
    db_session.commit()
    _add_v2_canned_message(db_session, conversation, 'SUMMARY_V2_SENTINEL')

    response = client.get(
        '/api/v1/assistant/conversations',
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 409
    assert b'SUMMARY_V2_SENTINEL' not in response.content
    _assert_no_store(response)


def test_conversation_list_ignores_live_v2_that_does_not_contribute_summary(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(
        user_id='employee-mina', title='summary window'
    )
    db_session.add(conversation)
    db_session.commit()
    _add_v2_canned_message(db_session, conversation, 'OLD_V2_SENTINEL')
    for index in range(5):
        _add_legacy_message(db_session, conversation, f'legacy summary {index}')

    response = client.get(
        '/api/v1/assistant/conversations',
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 200
    summary = response.json()['conversations'][0]['summary']
    assert 'OLD_V2_SENTINEL' not in summary
    assert 'legacy summary 1' in summary
    assert 'cache-control' not in response.headers


def test_legacy_only_message_get_preserves_existing_contract_without_capability(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation = AssistantConversation(user_id='employee-mina', title='legacy')
    db_session.add(conversation)
    db_session.commit()
    _add_legacy_message(db_session, conversation, 'LEGACY_ONLY')
    response = client.get(
        f'/api/v1/assistant/conversations/{conversation.id}/messages',
        headers={'X-Demo-User': 'viewer'},
    )
    assert response.status_code == 200
    assert response.json()['messages'][0]['content'] == 'LEGACY_ONLY'
    assert 'cache-control' not in response.headers
