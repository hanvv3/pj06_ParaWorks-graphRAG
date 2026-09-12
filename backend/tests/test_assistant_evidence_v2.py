"""Real writer/current resolver regressions; no provider or eligibility doubles."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import keyed_fingerprint
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.assistant.evidence_persistence import (
    AssistantEvidenceWriter,
    AssistantMessageProjection,
    _mint_assistant_exact_write_authority,
)
from backend.app.assistant.service import (
    assistant_message_projection_is_live,
    build_contextual_question,
    create_conversation,
    eligible_context_messages,
    serialize_conversation,
    serialize_message,
)
from backend.app.core.config import get_settings
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AgentRun,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AutoReviewRuntimeKeyState,
    ReviewItem,
)
from backend.app.rag.evidence_projection import (
    CanonicalEvidenceProjector,
    ProjectionFence,
)
from backend.app.rag.retrieval import EvidenceSlot
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver
from backend.tests.assistant_evidence_helpers import (
    ensure_fingerprint_runtime,
    sqlite_writer_insert,
    write_canned,
)
from backend.tests.test_rag_v2_keyword_retriever import (
    _scope,
    _seed_sqlite_raw_projection,
    _seed_sqlite_trusted_projection,
    _settings,
)

UNAVAILABLE = '이 답변의 근거를 더 이상 확인할 수 없습니다. 다시 생성해 주세요.'


def test_unrelated_dirty_parent_does_not_require_evidence_for_user_input(written):
    db, conversation, _, parent, *_ = written
    user_message = AssistantMessage(
        conversation_id=conversation.id, role='user', content='user input'
    )
    db.add(user_message)
    db.commit()
    parent.model_name = 'pending unrelated edit'
    views = eligible_context_messages(db, USERS['viewer'], [user_message])
    assert [view.content for view in views] == ['user input']
    assert parent.model_name == 'pending unrelated edit' and parent in db.dirty


def test_database_read_failure_has_safe_projection_for_expired_row(
    written, monkeypatch
):
    from sqlalchemy.exc import SQLAlchemyError

    db, _, message, *_ = written
    message_id = message.id
    db.expire(message)
    original = Session.get

    def fail_message_read(self, entity, ident, *args, **kwargs):
        if entity is AssistantMessage and ident == message_id:
            raise SQLAlchemyError('PRIVATE DATABASE FAILURE')
        return original(self, entity, ident, *args, **kwargs)

    monkeypatch.setattr(Session, 'get', fail_message_read)
    result = serialize_message(message, db=db, user=USERS['viewer'])
    assert result['id'] == message_id and result['content'] == UNAVAILABLE
    assert 'PRIVATE DATABASE FAILURE' not in str(result)


@pytest.mark.parametrize('role', ('user', 'system', 'tool', 'ASSISTANT'))
def test_role_mutation_cannot_bypass_context_evidence_validation(written, role):
    db, conversation, message, *_ = written
    message.role = role
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )
    assert eligible_context_messages(db, USERS['viewer'], [message]) == []
    assert (
        serialize_conversation(conversation, db=db, user=USERS['viewer'])['summary']
        is None
    )


@pytest.mark.parametrize('consumer', ('context', 'summary', 'email'))
def test_consumers_use_verified_snapshot_after_interleaved_row_change(
    written,
    monkeypatch,
    consumer,
):
    from backend.app.assistant.email_draft_context import find_latest_sendable_artifact
    from backend.app.assistant.evidence_reader import AssistantEvidenceReader

    db, conversation, first, *_ = written
    second = write_canned(db, conversation, 'safe second')
    first_id, second_id = first.id, second.id
    verified_content = first.content
    original = AssistantEvidenceReader.project_message

    def interleave(self, **kwargs):
        if kwargs['message'].id == second_id:
            with Session(db.get_bind()) as other:
                other.execute(
                    update(AssistantMessage)
                    .where(
                        AssistantMessage.id == first_id,
                    )
                    .values(content='UNVERIFIED_INJECTION')
                )
                other.commit()
        return original(self, **kwargs)

    monkeypatch.setattr(AssistantEvidenceReader, 'project_message', interleave)
    if consumer == 'context':
        result = build_contextual_question(
            conversation=conversation,
            messages=[first, second],
            new_message='next',
            db=db,
            user=USERS['viewer'],
        )
    elif consumer == 'summary':
        result = serialize_conversation(conversation, db=db, user=USERS['viewer'])[
            'summary'
        ]
    else:
        snapshots = eligible_context_messages(db, USERS['viewer'], [first, second])
        result = find_latest_sendable_artifact(snapshots[:1])['content']
    assert 'UNVERIFIED_INJECTION' not in result
    assert verified_content.strip() in result


@pytest.mark.parametrize('expire', (False, True))
def test_deleted_message_returns_safe_projection_without_refreshing_descriptors(
    written, expire
):
    db, _, message, *_ = written
    message_id = message.id
    if expire:
        db.expire(message)
    with Session(db.get_bind()) as other:
        other.execute(
            AssistantMessage.__table__.delete().where(AssistantMessage.id == message_id)
        )
        other.commit()
    result = serialize_message(message, db=db, user=USERS['viewer'])
    assert result['id'] == message_id
    assert result['content'] == UNAVAILABLE
    assert eligible_context_messages(db, USERS['viewer'], [message]) == []


@pytest.mark.parametrize(
    'value', ({'question': 'PRIVATE RAW QUESTION'}, ['PRIVATE RAW QUESTION'])
)
@pytest.mark.parametrize(
    'key', ('status', 'failure_reason', 'failure_class', 'agent_name', 'prompt_version')
)
def test_allowlisted_metadata_keys_reject_structured_private_values(
    written, key, value
):
    db, _, message, *_ = written
    message.metadata_ = {
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v2',
        key: value,
    }
    db.commit()
    assert 'PRIVATE RAW QUESTION' not in str(
        serialize_message(message, db=db, user=USERS['viewer'])
    )


def test_dirty_source_does_not_mask_committed_revocation_or_get_flushed(written):
    from backend.app.models import Source

    db, _, message, _, _, source, *_ = written
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        != UNAVAILABLE
    )
    source_id, original_author = source.id, source.author
    source.author = 'pending caller update'
    with Session(db.get_bind()) as other:
        other.execute(
            update(Source)
            .where(Source.id == source_id)
            .values(permission_level='restricted')
        )
        other.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )
    assert source.author == 'pending caller update' and source in db.dirty
    with db.no_autoflush:
        assert (
            db.scalar(select(Source.author).where(Source.id == source_id))
            == original_author
        )


def test_context_snapshots_keep_user_and_nested_email_metadata_without_mutable_aliases(
    db_session,
):
    from dataclasses import FrozenInstanceError

    from backend.app.assistant.email_draft_context import (
        find_latest_pending_email_draft,
    )

    db = db_session
    conversation = create_conversation(db, USERS['viewer'])
    user_message = AssistantMessage(
        conversation_id=conversation.id, role='user', content='legitimate question'
    )
    email = AssistantMessage(
        conversation_id=conversation.id,
        role='assistant',
        content='draft',
        metadata_={
            'action_type': 'email_draft',
            'status': 'pending_approval',
            'email_draft': {
                'to': ['person@example.test'],
                'subject': 'subject',
                'body': 'verified body',
            },
        },
    )
    db.add_all([user_message, email])
    db.commit()
    views = eligible_context_messages(db, USERS['viewer'], [user_message, email])
    assert [view.role for view in views] == ['user', 'assistant']
    with pytest.raises(FrozenInstanceError):
        views[0].content = 'injection'
    with pytest.raises(TypeError):
        views[1].metadata['email_draft']['body'] = 'injection'
    detached = views[1].metadata_
    detached['email_draft']['body'] = 'injection'
    assert find_latest_pending_email_draft(views).body == 'verified body'
    assert views[1].to_response()['metadata']['email_draft']['body'] == 'verified body'


def test_missing_current_key_runtime_never_authenticates_retained_v2_bytes(written):
    db, _, message, *_ = written
    db.delete(db.scalar(select(AutoReviewRuntimeKeyState)))
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )


def test_reader_does_not_release_other_owners_conversation_even_with_source_permission(
    written,
):
    db, _, message, *_ = written
    assert (
        serialize_message(message, db=db, user=USERS['hanvv-employee'])['content']
        == UNAVAILABLE
    )


@pytest.fixture
def written(db_session, monkeypatch):
    settings = _settings()
    monkeypatch.setenv(
        'AGENT_RUNTIME_FINGERPRINT_SECRET', settings.agent_runtime_fingerprint_secret
    )
    monkeypatch.setenv(
        'AGENT_RUNTIME_FINGERPRINT_KEY_VERSION',
        settings.agent_runtime_fingerprint_key_version,
    )
    get_settings.cache_clear()
    db = db_session
    source, chunk = _seed_sqlite_raw_projection(db)
    db.add(
        ReviewItem(
            item_type='document',
            payload={'source_ids': [source.source_id]},
            source_links=[source.source_url],
            source_snippets=[chunk.source_snippet],
            confidence_score=1,
            permission_level='internal',
            status='approved',
            resolution_source='human',
        )
    )
    db.commit()
    decision = _seed_sqlite_trusted_projection(db)
    scope = _scope(allowed_permission_levels=('public', 'internal'))
    raw = CanonicalSourceObservationResolver(
        db=db, settings=settings
    ).resolve_for_index(chunk.id)
    trusted = TrustedServingEnvelopeResolver(
        db=db, settings=settings
    ).resolve_for_index('decision_record', decision.id)
    slots = (
        EvidenceSlot(
            'E1', raw.evidence.support_mode, raw.evidence, 0.75, ('raw', 'ordered')
        ),
        EvidenceSlot(
            'E2', trusted.evidence.support_mode, trusted.evidence, 0.5, ('trusted',)
        ),
    )
    projector = CanonicalEvidenceProjector(db=db, settings=settings)
    prepared = projector.prepare_model_influence(
        slots,
        scope=scope,
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        prepared_readiness_hmac=None,
        rendered_input_hmac='a' * 64,
    )
    fence = ProjectionFence(
        1,
        None,
        1,
        None,
        prepared_hidden_membership_hmac='b' * 64,
        current_hidden_membership_hmac='b' * 64,
    )
    dependencies = projector.finalize_model_influence_dependencies(
        prepared, ('E1',), scope=scope, fence=fence
    )
    evidence = projector.project_selected(
        slots, selected_slot_ids=('E1',), scope=scope, fence=fence
    )
    assert len(dependencies) == 2 and len(evidence.citations) == 1
    conversation = create_conversation(db, USERS['viewer'])
    parent = AgentRun(
        agent_name='rag_orchestrator_agent',
        prompt_version='rag-answer:v2',
        source_window='rag-v2:product:enforce:assistant:keyword',
        cache_key='rag-v2-final:' + 'e' * 64,
        model_name='fake',
        permission_level='internal',
        status='complete',
        run_contract_version='rag-run:v2',
        run_record_phase='final',
        total_charged_cost_usd=0,
        completed_at=datetime.now(UTC),
        metadata_={
            'rag_result_hmac': 'd' * 64,
            'outcome': 'supported',
            'hidden_match_count': 0,
        },
    )
    db.add(parent)
    db.flush()
    projection = AssistantMessageProjection(
        content='  원본 e\u0301 답변\n',
        metadata={'prompt_version': 'rag-answer:v2', 'status': 'supported'},
        evidence=evidence,
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        assembled_answer_hmac='c' * 64,
        canned_message_identity=None,
        result_hmac='d' * 64,
        model_influence=dependencies,
    )
    writer = AssistantEvidenceWriter(
        fingerprint_secret=settings.agent_runtime_fingerprint_secret.encode(),
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        settings=settings,
    )
    with sqlite_writer_insert(db, projection, settings):
        message = writer.append_final(
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
    children = list(
        db.scalars(
            select(AssistantMessageEvidenceDependency).order_by(
                AssistantMessageEvidenceDependency.candidate_ordinal
            )
        )
    )
    return db, conversation, message, parent, children, source, chunk, decision


def test_writer_reader_preserves_exact_bytes_and_only_selected_citation(written):
    db, _, message, _, children, *_ = written
    result = serialize_message(message, db=db, user=USERS['viewer'])
    assert result['content'].encode() == '  원본 e\u0301 답변\n'.encode()
    assert len(result['citations']) == 1
    assert len(children) == 2


@pytest.mark.parametrize(
    'field,value',
    (
        ('content', '  원본 é 답변\n'),
        ('content', '  원본 e\u0301 답변\n '),
        ('content_hmac_key_version', 'rotated'),
        ('content_hmac_key_material_verifier', 'f' * 64),
        ('assistant_message_content_hmac', 'f' * 64),
        ('content_origin_hmac', 'f' * 64),
        ('rag_result_hmac', 'f' * 64),
        ('agent_run_id', 999),
        ('dependency_set_hmac', 'f' * 64),
        ('model_influence_set_hmac', 'f' * 64),
        ('parent_selected_evidence_projection_hmac', 'f' * 64),
        ('permission_level', 'public'),
        ('content_origin', 'rag_canned'),
        ('content_write_mode', 'legacy_trimmed'),
        ('serving_dependency_count', 1),
        ('dependency_set_hmac_schema_version', None),
    ),
)
def test_parent_tamper_removes_entire_answer_from_every_consumer(written, field, value):
    db, conversation, message, *_ = written
    setattr(
        message, field, value
    )  # Read dirty corrupt row without constraint-driven collection failures.
    with db.no_autoflush:
        result = serialize_message(message, db=db, user=USERS['viewer'])
        assert result['content'] == UNAVAILABLE
        assert (
            result['citations']
            == result['source_ids']
            == result['source_links']
            == result['source_snippets']
            == []
        )
        assert result['permission_level'] is None and result['hidden_match_count'] == 0
        assert result['metadata'] == {
            'status': 'evidence_unavailable',
            'regeneration_required': True,
        }
        assert eligible_context_messages(db, USERS['viewer'], [message]) == []
        assert (
            serialize_conversation(conversation, db=db, user=USERS['viewer'])['summary']
            is None
        )
        assert not assistant_message_projection_is_live(
            db, user=USERS['viewer'], message=message
        )


@pytest.mark.parametrize(
    'field,value',
    (
        ('dependency_child_hmac', 'f' * 64),
        ('fingerprint_key_version', 'old'),
        ('candidate_ordinal', 4),
        ('dependency_role', 'selected_citation'),
        ('dependency_serving_scope', 'legacy_v1_only'),
        ('model_content_hmac', 'f' * 64),
        ('serving_identity_hmac', 'f' * 64),
        ('approval_provenance_hmac', 'f' * 64),
        ('dependency_set_hmac', 'f' * 64),
        ('support_mode', 'source_observation'),
    ),
)
def test_unselected_influence_tamper_also_redacts_whole_answer(written, field, value):
    db, _, message, _, children, *_ = written
    setattr(children[1], field, value)
    with db.no_autoflush:
        assert (
            serialize_message(message, db=db, user=USERS['viewer'])['content']
            == UNAVAILABLE
        )


@pytest.mark.parametrize(
    'change',
    (
        'source_signature',
        'chunk_text',
        'source_permission',
        'approval',
        'missing_provenance',
        'citation_byte',
        'score',
        'source_array',
        'parent_result',
        'matched_term_order',
        'selected_child_deleted',
        'source_version',
    ),
)
def test_current_evidence_or_public_projection_drift_cannot_reuse_stored_answer(
    written, change
):
    db, _, message, parent, _, source, chunk, decision = written
    if change == 'source_signature':
        source.server_content_signature = 'f' * 64
    elif change == 'chunk_text':
        chunk.text += ' '
    elif change == 'source_permission':
        source.permission_level = 'restricted'
    elif change == 'approval':
        db.get(ReviewItem, decision.source_review_item_id).status = 'rejected'
    elif change == 'missing_provenance':
        decision.source_review_item_id = None
    elif change in {'citation_byte', 'score'}:
        citation = dict(message.citations[0])
        citation[
            'source_snippet' if change == 'citation_byte' else 'relevance_score'
        ] = 'altered' if change == 'citation_byte' else 0.76
        message.citations = [citation]
    elif change == 'source_array':
        message.source_links = ['https://example.test/altered']
    elif change == 'matched_term_order':
        citation = dict(message.citations[0])
        assert len(citation['matched_terms']) == 2
        citation['matched_terms'] = list(reversed(citation['matched_terms']))
        message.citations = [citation]
    elif change == 'selected_child_deleted':
        db.delete(
            db.scalar(
                select(AssistantMessageEvidenceDependency).where(
                    AssistantMessageEvidenceDependency.assistant_message_id
                    == message.id,
                    AssistantMessageEvidenceDependency.candidate_ordinal == 0,
                )
            )
        )
    elif change == 'source_version':
        from backend.app.models import Document, DocumentVersion

        version = db.get(DocumentVersion, chunk.version_id)
        db.get(Document, version.document_id).current_document_version_id = None
    else:
        parent.metadata_ = {**parent.metadata_, 'rag_result_hmac': 'f' * 64}
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )


def test_public_metadata_drops_private_question_trace_and_exception(written):
    db, _, message, *_ = written
    message.metadata_ = {
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v2',
        'question': 'PRIVATE QUESTION',
        'graph_version': 'internal',
        'effective_backend': 'keyword',
        'fallback_category': 'raw failure',
        'parent_agent_run_id': 999,
        'exception': 'PRIVATE ERROR',
    }
    db.commit()
    result = serialize_message(message, db=db, user=USERS['viewer'])
    assert result['metadata'] == {
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v2',
    }


@pytest.mark.parametrize(
    'field,value',
    (('serving_document_id', 'chunk:999'), ('serving_content_hash', 'f' * 64)),
)
def test_dependency_storage_identity_and_legacy_content_authority_cannot_drift(
    written, field, value
):
    db, _, message, _, children, *_ = written
    setattr(children[0], field, value)
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )


@pytest.mark.parametrize(
    'field,value',
    (('hidden_match_count', 1), ('permission_notice', 'PRIVATE EXCEPTION')),
)
def test_public_hidden_fields_must_match_final_parent_projection(written, field, value):
    db, _, message, *_ = written
    setattr(message, field, value)
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )


@pytest.mark.parametrize('hidden_count', (0, 1, 20))
def test_zero_child_canned_exact_bytes_survive_context_and_guard(
    db_session, hidden_count
):
    conversation = create_conversation(db_session, USERS['viewer'])
    message = write_canned(
        db_session,
        conversation,
        '  canned e\u0301\n',
        hidden_count=hidden_count,
        outcome='hidden_only' if hidden_count else 'no_match',
    )
    result = serialize_message(message, db=db_session, user=USERS['viewer'])
    assert result['content'].encode() == b'  canned e\xcc\x81\n'
    assert result['hidden_match_count'] == hidden_count
    assert result['citations'] == []
    assert [
        row.id
        for row in eligible_context_messages(db_session, USERS['viewer'], [message])
    ] == [message.id]
    assert assistant_message_projection_is_live(
        db_session, user=USERS['viewer'], message=message
    )


@pytest.mark.parametrize(
    'mutation',
    ('content', 'key', 'origin', 'linked_parent', 'parent_phase', 'parent_result'),
)
def test_canned_integrity_is_checked_without_dependency_marker(db_session, mutation):
    conversation = create_conversation(db_session, USERS['viewer'])
    message = write_canned(db_session, conversation, 'canned bytes')
    parent = db_session.get(AgentRun, message.agent_run_id)
    if mutation == 'content':
        message.content += ' '
    elif mutation == 'key':
        message.content_hmac_key_version = 'old'
    elif mutation == 'origin':
        message.content_origin_hmac = 'f' * 64
    elif mutation == 'linked_parent':
        other = write_canned(db_session, conversation, 'other answer')
        message.agent_run_id = message.linked_agent_run_id = other.agent_run_id
    elif mutation == 'parent_phase':
        parent.run_record_phase = 'admission'
    else:
        parent.metadata_ = {**parent.metadata_, 'rag_result_hmac': 'f' * 64}
    with db_session.no_autoflush:
        assert (
            serialize_message(message, db=db_session, user=USERS['viewer'])['content']
            == UNAVAILABLE
        )


def test_runtime_rotation_revokes_old_signature_without_rewriting_answer(written):
    db, _, message, *_ = written
    runtime = db.scalar(select(AutoReviewRuntimeKeyState))
    runtime.fingerprint_key_version = 'rotated'
    runtime.fingerprint_key_material_verifier = fingerprint_key_material_verifier(
        'new-test-secret'
    )
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )
    assert message.content == '  원본 e\u0301 답변\n'


def test_hidden_notice_alone_is_not_historical_evidence(db_session):
    conversation = create_conversation(db_session, USERS['viewer'])
    from backend.app.assistant.service import append_assistant_message

    message = append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='  Legacy hidden  ',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=1,
        permission_notice='Some sources may be hidden by permissions.',
        agent_run_id=None,
        metadata={
            'agent_name': 'rag_orchestrator_agent',
            'prompt_version': 'rag-answer:v1',
        },
    )
    result = serialize_message(message, db=db_session, user=USERS['viewer'])
    assert result['content'] == 'Legacy hidden'
    assert result['hidden_match_count'] == 1
    assert (
        message.content_write_mode is None
        and message.dependency_set_hmac_schema_version is None
    )


def test_legacy_failure_triple_survives_but_exception_and_question_do_not(db_session):
    conversation = create_conversation(db_session, USERS['viewer'])
    from backend.app.assistant.service import append_assistant_message

    message = append_assistant_message(
        db_session,
        USERS['viewer'],
        conversation,
        content='Failure',
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={
            'agent_name': 'rag_orchestrator_agent',
            'prompt_version': 'rag-answer:v1',
            'status': 'failed',
            'failure_reason': 'rag_exception',
            'failure_class': 'RuntimeError',
            'question': 'PRIVATE',
            'exception': 'PRIVATE',
            'graph_version': 'PRIVATE',
            'retrieval_backend': 'PRIVATE',
            'internal_id': 'PRIVATE',
            'failure_message': 'PRIVATE',
        },
    )
    assert serialize_message(message, db=db_session, user=USERS['viewer'])[
        'metadata'
    ] == {
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v1',
        'status': 'failed',
        'failure_reason': 'rag_exception',
        'failure_class': 'RuntimeError',
    }


def test_zero_child_generation_failure_retains_signed_operational_message(db_session):
    conversation = create_conversation(db_session, USERS['viewer'])
    message = write_canned(
        db_session,
        conversation,
        'Safe generation failure',
        outcome='model_provider_failed',
        canned_identity='rag-canned-generation-failure:v1',
        parent_status='failed',
    )
    result = serialize_message(message, db=db_session, user=USERS['viewer'])
    assert result['content'] == 'Safe generation failure'
    assert result['citations'] == []


def test_external_source_change_cannot_hide_in_session_identity_map(written):
    from sqlalchemy.orm import Session

    db, _, message, _, _, source, *_ = written
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        != UNAVAILABLE
    )
    from backend.app.models import Source

    with Session(db.get_bind()) as other:
        other.get(Source, source.id).server_content_signature = 'f' * 64
        other.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )


@pytest.fixture
def legacy_keyed(db_session):
    """Future legacy snapshot, independently signed from the written §16.2 payload.

    Controller ruling: snapshot serving_version_fingerprint is the existing V1
    serving_content_hash. Child's D serving_version_fingerprint stays NULL.
    Task18 must add the missing production legacy writer; no envelope is invented.
    """
    import struct

    from backend.app.assistant.service import _serving_content_hash
    from backend.app.knowledge.serving_text import canonical_knowledge_text
    from backend.app.models import DecisionRecord
    from backend.app.rag.evidence_projection import (
        build_v1_selected_evidence_projection_hmac,
    )
    from backend.app.rag.serving_contracts import (
        build_canonical_citation_projection_hmac,
        build_model_content_hmac,
    )

    db = db_session
    settings = get_settings()
    ensure_fingerprint_runtime(db, settings)
    secret = settings.agent_runtime_fingerprint_secret.encode()

    def fp(value, schema, policy='assistant-evidence:v1'):
        return keyed_fingerprint(
            value, secret=secret, schema_version=schema, policy_version=policy
        )

    target = DecisionRecord(
        title='Legacy decision',
        decision_summary='Exact old fact',
        source_links=['https://example.test/legacy'],
        source_snippets=['Original snippet'],
        permission_level='internal',
        review_status='approved',
        confidence_score=1,
    )
    db.add(target)
    db.flush()
    source_id = f'decision_record:{target.id}'
    text = canonical_knowledge_text('decision_record', target)
    citation = {
        'source_id': source_id,
        'source_url': 'https://example.test/legacy',
        'source_type': 'decision_record',
        'permission_level': 'internal',
        'source_snippet': 'Original snippet',
        'relevance_score': 0.75,
        'matched_terms': ['old'],
    }
    citation_hmac = fp(
        {
            'source_id_bytes': exact_utf8_bytes(source_id),
            'source_url_bytes': exact_utf8_bytes(citation['source_url']),
            'source_type_bytes': exact_utf8_bytes('decision_record'),
            'permission_level': 'internal',
            'source_snippet_bytes': exact_utf8_bytes('Original snippet'),
            'relevance_score_binary64_be_hex': struct.pack('>d', 0.75).hex(),
            'matched_terms_bytes': [exact_utf8_bytes('old')],
        },
        'rag-v1-evidence-projection:citation:v1',
        'rag-v1-evidence-projection:v1',
    )
    selected_set = build_v1_selected_evidence_projection_hmac(
        citation_hmacs=(citation_hmac,),
        source_ids=(source_id,),
        source_links=tuple(target.source_links),
        source_snippets=tuple(target.source_snippets),
        settings=settings,
    )
    model_hmac = build_model_content_hmac(
        serving_kind='trusted_knowledge', model_content=text, settings=settings
    )
    canonical_hmac = build_canonical_citation_projection_hmac(
        public_source_id=source_id,
        public_source_type='decision_record',
        source_url=citation['source_url'],
        source_snippet=citation['source_snippet'],
        effective_permission='internal',
        settings=settings,
    )
    legacy_hash = _serving_content_hash(
        source_id=source_id,
        text=text,
        source_url=citation['source_url'],
        source_snippet=citation['source_snippet'],
        permission_level='internal',
    )
    legacy_identity = fp(
        {
            'canonical_citation_projection_hmac': canonical_hmac,
            'effective_permission': 'internal',
            'legacy_public_source_id_bytes': exact_utf8_bytes(source_id),
            'legacy_source_links_bytes': [exact_utf8_bytes(citation['source_url'])],
            'legacy_source_snippets_bytes': [
                exact_utf8_bytes(citation['source_snippet'])
            ],
            'model_content_hmac': model_hmac,
            'serving_version_fingerprint': legacy_hash,
        },
        'assistant-legacy-dependency-snapshot:v1',
    )
    origin = fp(
        {
            'agent_name_bytes': exact_utf8_bytes('rag_orchestrator_agent'),
            'content_bytes_hmac': fp(
                {'content_bytes': exact_utf8_bytes('Legacy answer')},
                'assistant-legacy-content-bytes:v1',
            ),
            'effective_backend': 'deterministic_lexical',
            'legacy_result_contract_version': 'rag-answer:v1',
            'output_permission': 'internal',
            'prompt_version_bytes': exact_utf8_bytes('rag-answer:v1'),
            'selected_evidence_projection_hmac': selected_set,
        },
        'assistant-legacy-evidence-origin:v1',
    )
    conversation = create_conversation(db, USERS['viewer'])
    message_id = (db.scalar(select(func.max(AssistantMessage.id))) or 0) + 1
    content_hmac = fp(
        {
            'agent_name_bytes': exact_utf8_bytes('rag_orchestrator_agent'),
            'assistant_message_id': message_id,
            'content_bytes': exact_utf8_bytes('Legacy answer'),
            'content_origin': 'legacy_evidence',
            'content_origin_hmac': origin,
            'content_write_mode': 'legacy_trimmed',
            'conversation_id': conversation.id,
            'linked_agent_run_id': None,
            'message_role': 'assistant',
            'prompt_version_bytes': exact_utf8_bytes('rag-answer:v1'),
            'rag_result_hmac': None,
        },
        'assistant-message-content-hmac:v1',
    )
    child_hmac = fp(
        {
            'approval_link_id': None,
            'approval_provenance_hmac': None,
            'candidate_ordinal': 0,
            'canonical_citation_projection_hmac': canonical_hmac,
            'dependency_kind': 'legacy_unbound',
            'dependency_role': 'selected_citation',
            'dependency_serving_scope': 'legacy_v1_only',
            'effective_permission': 'internal',
            'evidence_link_ids': [],
            'evidence_link_set_hmac': None,
            'legacy_dependency_identity_hmac': legacy_identity,
            'model_content_hmac': model_hmac,
            'raw_document_chunk_id': None,
            'selected_v1_citation_projection_hmac': citation_hmac,
            'serving_identity_hmac': None,
            'serving_version_fingerprint': None,
            'support_mode': None,
            'trusted_knowledge_id': None,
        },
        'assistant-dependency-child-hmac:v2',
    )
    set_hmac = fp(
        {
            'assistant_message_content_hmac': content_hmac,
            'content_origin': 'legacy_evidence',
            'content_origin_hmac': origin,
            'content_write_mode': 'legacy_trimmed',
            'dependencies': [
                {'candidate_ordinal': 0, 'dependency_child_hmac': child_hmac}
            ],
            'dependency_count': 1,
            'dependency_serving_scope': 'legacy_v1_only',
            'evidence_contract_version': 'assistant-evidence:v1',
            'linked_agent_run_id': None,
            'model_influence_set_hmac': None,
            'parent_selected_evidence_projection_hmac': selected_set,
            'rag_result_hmac': None,
        },
        'assistant-dependency-set-hmac:v2',
    )
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    message = AssistantMessage(
        id=message_id,
        conversation_id=conversation.id,
        role='assistant',
        content='Legacy answer',
        citations=[citation],
        source_ids=[source_id],
        source_links=target.source_links,
        source_snippets=target.source_snippets,
        permission_level='internal',
        hidden_match_count=0,
        evidence_contract_version='assistant-evidence:v1',
        serving_dependency_count=1,
        content_write_mode='legacy_trimmed',
        content_hmac_schema_version='assistant-message-content-hmac:v1',
        assistant_message_content_hmac=content_hmac,
        content_hmac_key_version=settings.agent_runtime_fingerprint_key_version,
        content_hmac_key_material_verifier=verifier,
        content_origin='legacy_evidence',
        content_origin_hmac=origin,
        dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v2',
        dependency_set_hmac=set_hmac,
        parent_selected_evidence_projection_hmac=selected_set,
        metadata_={
            'agent_name': 'rag_orchestrator_agent',
            'prompt_version': 'rag-answer:v1',
            'effective_backend': 'deterministic_lexical',
        },
    )
    db.add(message)
    db.flush()
    child = AssistantMessageEvidenceDependency(
        assistant_message_id=message.id,
        candidate_ordinal=0,
        serving_document_id=source_id,
        dependency_kind='legacy_unbound',
        dependency_set_hmac=set_hmac,
        serving_content_hash=legacy_hash,
        permission_level='internal',
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=verifier,
        dependency_serving_scope='legacy_v1_only',
        dependency_role='selected_citation',
        dependency_child_hmac=child_hmac,
        model_content_hmac=model_hmac,
        canonical_citation_projection_hmac=canonical_hmac,
        selected_v1_citation_projection_hmac=citation_hmac,
        legacy_dependency_identity_hmac=legacy_identity,
    )
    db.add(child)
    db.commit()
    return db, message, child, target


def test_valid_keyed_legacy_snapshot_reads_without_ever_granting_v2_eligibility(
    legacy_keyed,
):
    db, message, child, target = legacy_keyed
    result = serialize_message(message, db=db, user=USERS['viewer'])
    assert result['content'] == 'Legacy answer'
    assert result['citations'][0]['source_snippet'] == 'Original snippet'
    assert (
        child.serving_version_fingerprint is None
        and child.serving_identity_hmac is None
    )
    assert (
        TrustedServingEnvelopeResolver(
            db=db, settings=get_settings()
        ).resolve_for_index('decision_record', target.id)
        is None
    )


@pytest.mark.parametrize(
    'mutation',
    (
        'content',
        'permission',
        'identity',
        'arrays',
        'child_hmac',
        'set_hmac',
        'origin_hmac',
    ),
)
def test_keyed_legacy_snapshot_drift_redacts_without_promoting_missing_provenance(
    legacy_keyed, mutation
):
    db, message, child, target = legacy_keyed
    if mutation == 'content':
        target.decision_summary += ' altered'
    elif mutation == 'permission':
        target.permission_level = 'restricted'
    elif mutation == 'identity':
        child.serving_document_id = 'decision_record:99999'
    elif mutation == 'arrays':
        message.source_links = ['https://example.test/changed']
    elif mutation == 'child_hmac':
        child.dependency_child_hmac = 'f' * 64
    elif mutation == 'set_hmac':
        message.dependency_set_hmac = 'f' * 64
    else:
        message.content_origin_hmac = 'f' * 64
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )


@pytest.mark.parametrize(
    'mutation', ('none', 'revoke', 'ref_removed', 'ref_approval_swap', 'link_signature')
)
def test_explicit_approval_and_exact_evidence_link_refs_are_revalidated(
    db_session, monkeypatch, mutation
):
    from backend.app.models import (
        AssistantMessageKnowledgeEvidenceRef,
        TrustedKnowledgeEvidenceLink,
    )
    from backend.tests.test_rag_trusted_evidence import (
        _seed_item,
        _seed_link,
        _seed_source,
        _seed_target,
    )
    from backend.tests.test_rag_trusted_evidence import _settings as trusted_settings

    settings = trusted_settings().model_copy(
        update={'agent_runtime_security_scope_id': 'workspace-a'}
    )
    monkeypatch.setenv(
        'AGENT_RUNTIME_FINGERPRINT_SECRET', settings.agent_runtime_fingerprint_secret
    )
    monkeypatch.setenv(
        'AGENT_RUNTIME_FINGERPRINT_KEY_VERSION',
        settings.agent_runtime_fingerprint_key_version,
    )
    monkeypatch.setenv('AGENT_RUNTIME_SECURITY_SCOPE_ID', 'workspace-a')
    get_settings.cache_clear()
    db = db_session
    source, chunk = _seed_source(db, ordinal=1)
    item = _seed_item(db, source=source, chunk=chunk, resolution_source='human')
    target = _seed_target(db, source_review_item_id=item.id)
    target.source_links, target.source_snippets = (
        list(item.source_links),
        list(item.source_snippets),
    )
    approval = _seed_link(
        db, target=target, item=item, source=source, resolution_source='human'
    )
    db.commit()
    envelope = TrustedServingEnvelopeResolver(
        db=db, settings=settings
    ).resolve_for_index('history_event', target.id)
    assert envelope is not None
    scope = _scope(
        allowed_permission_levels=('public', 'internal'),
        workspace_scope_id='workspace-a',
    )
    slots = (EvidenceSlot('E1', 'trusted_fact', envelope.evidence, 0.75, ('history',)),)
    projector = CanonicalEvidenceProjector(db=db, settings=settings)
    prepared = projector.prepare_model_influence(
        slots,
        scope=scope,
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        prepared_readiness_hmac=None,
        rendered_input_hmac='a' * 64,
    )
    fence = ProjectionFence(
        1,
        None,
        1,
        None,
        prepared_hidden_membership_hmac='b' * 64,
        current_hidden_membership_hmac='b' * 64,
    )
    dependencies = projector.finalize_model_influence_dependencies(
        prepared, ('E1',), scope=scope, fence=fence
    )
    evidence = projector.project_selected(
        slots, selected_slot_ids=('E1',), scope=scope, fence=fence
    )
    conversation = create_conversation(db, USERS['viewer'])
    canned = write_canned(db, conversation, 'temporary', settings=settings)
    parent = db.get(AgentRun, canned.agent_run_id)
    parent.metadata_ = {**parent.metadata_, 'outcome': 'supported'}
    projection = AssistantMessageProjection(
        content='Approved answer',
        metadata={'status': 'supported'},
        evidence=evidence,
        permission_level='internal',
        hidden_match_count=0,
        permission_notice=None,
        assembled_answer_hmac='c' * 64,
        canned_message_identity=None,
        result_hmac='d' * 64,
        model_influence=dependencies,
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
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == 'Approved answer'
    )
    ref = db.scalar(
        select(AssistantMessageKnowledgeEvidenceRef).where(
            AssistantMessageKnowledgeEvidenceRef.assistant_message_id == message.id
        )
    )
    if mutation == 'revoke':
        approval.active = False
    elif mutation == 'ref_removed':
        db.delete(ref)
    elif mutation == 'ref_approval_swap':
        ref.approval_link_id = 999
    elif mutation == 'link_signature':
        db.get(
            TrustedKnowledgeEvidenceLink, ref.trusted_knowledge_evidence_link_id
        ).canonical_version_or_signature = 'f' * 64
    db.commit()
    assert serialize_message(message, db=db, user=USERS['viewer'])['content'] == (
        'Approved answer' if mutation == 'none' else UNAVAILABLE
    )


def test_null_parent_marker_cannot_downgrade_v2_children_into_historical_authority(
    written,
):
    db, _, message, *_ = written
    for field in (
        'content_write_mode',
        'content_hmac_schema_version',
        'assistant_message_content_hmac',
        'content_hmac_key_version',
        'content_hmac_key_material_verifier',
        'content_origin',
        'content_origin_hmac',
        'rag_result_hmac',
        'linked_agent_run_id',
        'dependency_set_hmac_schema_version',
        'dependency_set_hmac',
        'parent_selected_evidence_projection_hmac',
        'model_influence_set_hmac',
    ):
        setattr(message, field, None)
    db.commit()
    assert (
        serialize_message(message, db=db, user=USERS['viewer'])['content']
        == UNAVAILABLE
    )
