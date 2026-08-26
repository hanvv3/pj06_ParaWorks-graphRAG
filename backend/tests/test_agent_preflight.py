from backend.app.agent_runtime import PermissionContext
from backend.app.agents.mail_document_agent import build_mail_document_agent_preflight
from backend.app.agents.memory_extraction_agent import (
    build_memory_extraction_agent_preflight,
)
from backend.tests.test_mail_document_agent_review_bridge import seed_chunk


def test_mail_document_agent_preflight_reports_cost_without_running_llm(db_session) -> None:
    seed_chunk(
        db_session,
        'drive',
        'drive-preflight-plan',
        'restricted',
        text='담당: 김하나\n업무: 고객사 공유본 준비\n기한: 2026-05-20',
    )

    preflight = build_mail_document_agent_preflight(
        db=db_session,
        permission_context=PermissionContext(
            user_id='demo-admin',
            role='admin',
            allowed_permission_levels=('public', 'internal', 'restricted'),
        ),
        source_window='mail-docs:preflight',
    )

    assert preflight['action'] == 'preview_only'
    assert preflight['live_llm_execution'] is False
    assert preflight['evidence_message_count'] == 1
    assert preflight['estimated_input_tokens'] > 0
    assert preflight['estimated_cost_usd'] > 0


def test_memory_extraction_agent_preflight_includes_calendar_evidence(db_session) -> None:
    seed_chunk(
        db_session,
        'calendar',
        'calendar-preflight-deadline',
        'internal',
        text='고객사 공유본 준비 마감 일정',
        metadata={'start': '2026-05-20T09:00:00+09:00'},
    )

    preflight = build_memory_extraction_agent_preflight(
        db=db_session,
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
        source_window='memory:preflight',
    )

    assert preflight['action'] == 'preview_only'
    assert preflight['live_llm_execution'] is False
    assert preflight['evidence_message_count'] == 1
    assert preflight['included_source_types'] == ['calendar']
