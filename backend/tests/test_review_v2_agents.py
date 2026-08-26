import json
import subprocess
import sys

import pytest

from backend.app.agent_runtime.contracts import (
    AgentManifest,
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
)
from backend.app.agent_runtime.model_router import ReviewModelUnavailableError
from backend.app.agent_runtime.review_v2_agents import build_review_agent_catalog
from backend.app.agents.mail_document_agent import MailDocumentAgentModelResponse
from backend.app.core.config import Settings
from backend.app.schemas.review_workflow import DEFAULT_REVIEW_AGENT_NAMES


def test_existing_mail_agent_import_has_no_review_v2_cycle() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'from backend.app.agents.mail_document_agent import MailDocumentAgent; '
                'from backend.app.agent_runtime import ReviewDraftService'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


class _FakeMailModel:
    def extract(self, packet: EvidencePacket) -> MailDocumentAgentModelResponse:
        return MailDocumentAgentModelResponse(
            title='메일 후보',
            summary='메일 근거에서 검토 후보를 만들었습니다.',
            item_type='history_event',
            confidence_score=0.9,
            input_tokens=12,
            output_tokens=5,
            model_name='fake-configured-mail-model',
        )


class _FakeStructuredInvoker:
    def __init__(self, owner: '_FakeStructuredChatModel') -> None:
        self.owner = owner

    def invoke(self, messages):
        self.owner.invoke_count += 1
        prompt = json.loads(messages[1][1])
        item_type = prompt['expected_item_type']
        payload_key = {
            'timeline_event': 'result_summary',
            'history_event': 'reason',
            'decision_record': 'decision_summary',
            'todo': 'priority',
        }[item_type]
        payload_fields = {payload_key: 'structured evidence result'}
        if item_type == 'todo':
            payload_fields['priority_reason'] = 'structured evidence result'
        return {
            'title': f'{item_type} 후보',
            'summary': '구조화된 모델 응답입니다.',
            'item_type': item_type,
            'confidence_score': 0.91,
            'payload_fields': payload_fields,
            'uncertainty_reason': None,
        }


class _FakeStructuredChatModel:
    def __init__(self) -> None:
        self.structured_schema_count = 0
        self.invoke_count = 0

    def with_structured_output(self, schema):
        assert schema.__name__ == 'StructuredMemoryExtractionOutput'
        self.structured_schema_count += 1
        return _FakeStructuredInvoker(self)


def _packet() -> EvidencePacket:
    return EvidencePacket(
        source_type='company_memory',
        source_window='company-memory-review-selection:v1:ranked:12',
        messages=[
            EvidenceMessage(
                source_id='gmail:message-1',
                source_url='https://mail.example.test/message-1',
                text='고객 데모 일정 때문에 QA를 완료하고 배포하기로 결정했습니다.',
                author='owner@example.test',
                timestamp='2026-08-27T09:00:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
                source_snippet_override='고객 데모 일정 때문에 QA를 완료했습니다.',
            )
        ],
        permission_context=PermissionContext(
            user_id='actor-1',
            role='workflow_actor',
            allowed_permission_levels=('internal',),
        ),
    )


def _configured_settings() -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        agent_llm_enabled=True,
        openai_api_key='test-only-key',
        gemini_api_key=None,
        agent_llm_provider_order='openai',
        agent_llm_max_estimated_cost_usd=1.0,
    )


def test_catalog_registers_only_exact_public_manifest_names() -> None:
    catalog = build_review_agent_catalog(
        Settings(_env_file=None, paraworks_demo_mode=True)
    )

    assert catalog.registry.names == DEFAULT_REVIEW_AGENT_NAMES
    assert tuple(catalog.get(name).manifest.name for name in catalog.registry.names) == (
        DEFAULT_REVIEW_AGENT_NAMES
    )
    with pytest.raises(KeyError):
        catalog.get('memory_extraction_agent')
    with pytest.raises(ValueError):
        catalog.registry.register(
            AgentManifest(
                name='sixth_agent',
                owner='Invalid',
                input_contract='EvidencePacket',
                output_contract='AgentRunResult',
                prompt_versions=('invalid:v1',),
                supported_permissions=('internal',),
                capabilities=('review_draft',),
            )
        )
    assert catalog.registry.names == DEFAULT_REVIEW_AGENT_NAMES


def test_configured_mail_adapter_uses_existing_langchain_builder(monkeypatch) -> None:
    calls = []

    def fake_mail_builder(settings):
        calls.append(settings)
        return _FakeMailModel()

    fake_chat = _FakeStructuredChatModel()
    monkeypatch.setattr(
        'backend.app.agent_runtime.model_router.build_langchain_mail_document_agent_model',
        fake_mail_builder,
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.model_router.build_langchain_review_chat_model',
        lambda _settings: fake_chat,
    )

    catalog = build_review_agent_catalog(_configured_settings())
    result = catalog.get('mail_document_agent').run(_packet())

    assert len(calls) == 1
    assert calls[0].enabled is True
    assert calls[0].provider_order == ('openai',)
    assert result.agent_name == 'mail_document_agent'
    assert result.cost.model_name == 'fake-configured-mail-model'


def test_configured_memory_adapters_use_langchain_structured_output(
    monkeypatch,
) -> None:
    fake_chat = _FakeStructuredChatModel()
    monkeypatch.setattr(
        'backend.app.agent_runtime.model_router.build_langchain_mail_document_agent_model',
        lambda _settings: _FakeMailModel(),
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.model_router.build_langchain_review_chat_model',
        lambda _settings: fake_chat,
    )
    catalog = build_review_agent_catalog(_configured_settings())

    results = [catalog.get(name).run(_packet()) for name in DEFAULT_REVIEW_AGENT_NAMES[1:]]

    assert [result.agent_name for result in results] == list(
        DEFAULT_REVIEW_AGENT_NAMES[1:]
    )
    assert fake_chat.structured_schema_count == 4
    assert fake_chat.invoke_count == 4
    assert all(result.cost.model_name == 'gpt-5.4-mini' for result in results)


@pytest.mark.parametrize('enabled', [False, True])
def test_production_provider_absence_fails_closed_instead_of_using_rules(
    enabled: bool,
) -> None:
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        agent_llm_enabled=enabled,
        openai_api_key=None,
        gemini_api_key=None,
    )

    with pytest.raises(ReviewModelUnavailableError) as exc_info:
        build_review_agent_catalog(settings)

    assert exc_info.value.code == 'model_unavailable'
    assert 'credential' not in str(exc_info.value).lower()


def test_demo_catalog_uses_deterministic_models_without_network(monkeypatch) -> None:
    def unexpected_provider_builder(*_args, **_kwargs):
        raise AssertionError('demo catalog must not construct a provider model')

    monkeypatch.setattr(
        'backend.app.agent_runtime.model_router.build_langchain_mail_document_agent_model',
        unexpected_provider_builder,
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.model_router.build_langchain_review_chat_model',
        unexpected_provider_builder,
    )
    catalog = build_review_agent_catalog(
        Settings(
            _env_file=None,
            paraworks_demo_mode=True,
            agent_llm_enabled=True,
            openai_api_key='must-not-be-used',
        )
    )

    results = [catalog.get(name).run(_packet()) for name in DEFAULT_REVIEW_AGENT_NAMES]

    assert [result.agent_name for result in results] == list(DEFAULT_REVIEW_AGENT_NAMES)
    assert all(result.cost.estimated_cost_usd == 0.0 for result in results)
