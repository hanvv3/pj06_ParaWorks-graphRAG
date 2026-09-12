"""SQLite-only real Assistant writer fixtures shared across reader/capability tests."""

from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import event, func, select

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import keyed_fingerprint
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.assistant.evidence_persistence import (
    AssistantEvidenceWriter,
    AssistantMessageProjection,
    _mint_assistant_exact_write_authority,
)
from backend.app.core.config import get_settings
from backend.app.models import AgentRun, AssistantMessage, AutoReviewRuntimeKeyState
from backend.app.rag.evidence_projection import (
    V1EvidenceProjection,
    build_model_influence_set_hmac,
)


def ensure_fingerprint_runtime(db, settings):
    runtime = db.scalar(
        select(AutoReviewRuntimeKeyState).where(
            AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
        )
    )
    if runtime is None:
        db.add(
            AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                ready=True,
                generation=1,
                fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
                fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                    settings.agent_runtime_fingerprint_secret
                ),
            )
        )
        db.flush()


@contextmanager
def sqlite_writer_insert(db, projection, settings):
    """Task13's first flush needs Task18 repair; same bridge as SQLite smoke.

    Keep real CHECK constraints and writer hashes. Reserve ID before first flush;
    the writer replaces the temporary set marker before committing real children.
    """
    ensure_fingerprint_runtime(db, settings)
    reserved_id = (db.scalar(select(func.max(AssistantMessage.id))) or 0) + 1

    def seed(session, flush_context, instances):
        for message in session.new:
            if (
                type(message) is not AssistantMessage
                or message.assistant_message_content_hmac is not None
            ):
                continue
            message.id = reserved_id
            message.assistant_message_content_hmac = keyed_fingerprint(
                {
                    'agent_name_bytes': exact_utf8_bytes('rag_orchestrator_agent'),
                    'assistant_message_id': message.id,
                    'content_bytes': exact_utf8_bytes(message.content),
                    'content_origin': message.content_origin,
                    'content_origin_hmac': message.content_origin_hmac,
                    'content_write_mode': 'rag_v2_exact',
                    'conversation_id': message.conversation_id,
                    'linked_agent_run_id': message.linked_agent_run_id,
                    'message_role': 'assistant',
                    'prompt_version_bytes': exact_utf8_bytes('rag-answer:v2'),
                    'rag_result_hmac': projection.result_hmac,
                },
                secret=settings.agent_runtime_fingerprint_secret.encode(),
                schema_version='assistant-message-content-hmac:v1',
                policy_version='assistant-evidence:v1',
            )
            if projection.model_influence:
                message.model_influence_set_hmac = build_model_influence_set_hmac(
                    dependencies=projection.model_influence, settings=settings
                )
                message.dependency_set_hmac = '0' * 64

    event.listen(db, 'before_flush', seed)
    try:
        yield
    finally:
        event.remove(db, 'before_flush', seed)


def write_canned(
    db,
    conversation,
    content,
    *,
    settings=None,
    hidden_count=0,
    outcome='no_match',
    canned_identity='rag-canned-no-evidence:v1',
    parent_status='complete',
):
    settings = settings or get_settings()
    parent = AgentRun(
        agent_name='rag_orchestrator_agent',
        prompt_version='rag-answer:v2',
        status=parent_status,
        source_window='rag-v2:product:enforce:assistant:keyword',
        cache_key='rag-v2-final:' + 'e' * 64,
        model_name='deterministic',
        permission_level='internal',
        run_contract_version='rag-run:v2',
        run_record_phase='final',
        total_charged_cost_usd=0,
        completed_at=datetime.now(UTC),
        metadata_={
            'outcome': outcome,
            'rag_result_hmac': 'd' * 64,
            'hidden_match_count': hidden_count,
        },
    )
    db.add(parent)
    db.flush()
    projection = AssistantMessageProjection(
        content=content,
        metadata={'status': outcome, 'prompt_version': 'rag-answer:v2'},
        evidence=V1EvidenceProjection((), (), (), (), (), 'a' * 64),
        permission_level=None,
        hidden_match_count=hidden_count,
        permission_notice='Some sources may be hidden by permissions.'
        if hidden_count
        else None,
        assembled_answer_hmac=None,
        canned_message_identity=canned_identity,
        result_hmac='d' * 64,
    )
    with sqlite_writer_insert(db, projection, settings):
        message = AssistantEvidenceWriter(
            fingerprint_secret=settings.agent_runtime_fingerprint_secret.encode(),
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            settings=settings,
        ).append_final(
            db=db,
            conversation=conversation,
            projection=projection,
            authority=_mint_assistant_exact_write_authority(
                parent_agent_run_id=parent.id,
                conversation_id=conversation.id,
                projection=projection,
                parent_result_hmac=projection.result_hmac,
            ),
        )
    db.commit()
    return message
