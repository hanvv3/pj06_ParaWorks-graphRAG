from dataclasses import replace
from inspect import signature

from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.auto_review_eligibility import (
    AutoReviewEligibilityContext,
    AutoReviewEligibilityService,
    AutoReviewEvidenceMessage,
    build_candidate_validation_request,
)
from backend.app.agent_runtime.auto_review_input_safety import (
    scan_auto_review_plaintext,
)
from backend.app.agent_runtime.canonical_sources import (
    BoundWorkflowEvidenceResolution,
    ResolvedSourceVersion,
    build_keyed_fingerprint,
    classify_bound_workflow_evidence,
)
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.core.config import Settings
from backend.app.knowledge.trusted_fingerprint_projection import (
    find_visible_trusted_collisions,
    hidden_or_legacy_collision_exists,
)
from backend.app.knowledge.trusted_provenance import (
    build_trusted_candidate_fingerprint,
)
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    ReviewItem,
    ReviewItemEvidenceRef,
    TrustedKnowledgeFingerprint,
)


def _timeline_item(**payload: object) -> ReviewItem:
    return ReviewItem(
        item_type='timeline_event',
        payload={
            'title': '배포 완료',
            'summary': '검색 인덱스 배포 요약',
            'result_summary': '검색 인덱스 배포가 완료되었습니다.',
        }
        | payload,
        source_links=['https://example.test/1', 'https://example.test/2'],
        source_snippets=['배포가 완료됨', '인덱스 상태 정상'],
        confidence_score=0.99,
        permission_level='internal',
        status='pending_review',
    )


def test_request_uses_only_local_slots_and_normalized_promotion_fields() -> None:
    request = build_candidate_validation_request(
        _timeline_item(),
        evidence_texts=('배포가 완료됨', '인덱스 상태 정상'),
    )

    assert request is not None
    assert request.candidate_slot_id == 'C01'
    assert request.item_type == 'timeline_event'
    assert [(claim.field_key, claim.text) for claim in request.claims] == [
        ('title', '배포 완료'),
        ('result_summary', '검색 인덱스 배포가 완료되었습니다.'),
    ]
    assert [(slot.slot_id, slot.text) for slot in request.evidence_slots] == [
        ('E01', '배포가 완료됨'),
        ('E02', '인덱스 상태 정상'),
    ]
    rendered = repr(request)
    assert 'https://example.test' not in rendered
    assert 'internal' not in rendered


def test_credential_match_discards_request_and_never_redacts_then_validates() -> None:
    item = _timeline_item(
        result_summary='OPENAI_API_KEY=sk-proj-' + ('A' * 32)
    )

    assert build_candidate_validation_request(
        item,
        evidence_texts=tuple(item.source_snippets),
    ) is None


def test_generic_credential_context_entropy_discards_request() -> None:
    for evidence in (
        'oauth_client_token=A8d91Kp2Lm4Nq6Rs8Tu0Vw3Xy5Za7Bc9',
        'database_secret=qP7!vN2@xR9#cL4$wT8%zK6&mH3',
        'cookie_secret=7f4c9a2e1b8d6f3a5c0e9b7d2a4f8c1e',
    ):
        assert build_candidate_validation_request(
            _timeline_item(),
            evidence_texts=(evidence,),
        ) is None


def test_versioned_scanner_detects_assignments_and_credential_context_entropy() -> None:
    detected = (
        'password = correct-horse-battery-staple',
        'connector_token: A8d91Kp2Lm4Nq6Rs8Tu0Vw3Xy5Za7Bc9',
        'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature',
        'WEBHOOK_SECRET=whsec_' + ('A1b2' * 8),
        'github_token=ghp_' + ('A1b2' * 10),
        'xoxb-' + ('1' * 24),
        '-----BEGIN PRIVATE KEY-----',
        'oauth_client_token=A8d91Kp2Lm4Nq6Rs8Tu0Vw3Xy5Za7Bc9',
        'database_secret=qP7!vN2@xR9#cL4$wT8%zK6&mH3',
        'cookie_secret=7f4c9a2e1b8d6f3a5c0e9b7d2a4f8c1e',
        'OPENAI_API_KEY=sk-proj-example-' + ('A1b2' * 10),
    )
    for text in detected:
        decision = scan_auto_review_plaintext(text)
        assert decision.allowed is False
        assert decision.reason_code == 'sensitive_input_detected'
        assert decision.scanner_version == 'credential-scan:v1'


def test_scanner_allows_safe_prose_and_documented_fake_values() -> None:
    safe = (
        '비밀번호 교체 절차를 문서화했습니다.',
        'token budget is 6000',
        'OPENAI_API_KEY=example-not-a-real-key',
        'sk-proj-example-placeholder-not-real',
        'cookie_secret=placeholder',
        'token_estimator_version=openai-o200k-chat:v1',
        'secretary_notes=QuarterlyPlanning2026_Final',
    )
    assert all(scan_auto_review_plaintext(text).allowed for text in safe)
    assert not scan_auto_review_plaintext(
        'OPENAI_API_KEY=sk-proj-AAAexampleBBB' + ('C' * 24)
    ).allowed


def test_missing_or_duplicate_evidence_text_is_not_rendered() -> None:
    item = _timeline_item()

    assert build_candidate_validation_request(item, evidence_texts=()) is None
    assert build_candidate_validation_request(
        item, evidence_texts=('same', 'same')
    ) is None


def test_history_uses_title_and_reason_while_other_types_are_ineligible() -> None:
    history = ReviewItem(
        item_type='history_event',
        payload={'title': '범위 변경', 'reason': '고급 diff는 다음 단계로 이동했습니다.'},
        source_links=['https://example.test/1'],
        source_snippets=['회의에서 범위 변경을 확인함'],
        confidence_score=0.99,
        permission_level='public',
        status='pending_review',
    )
    todo = ReviewItem(
        item_type='todo',
        payload={'title': '담당자 확인', 'priority': 'high', 'priority_reason': '필수'},
        source_links=['https://example.test/1'],
        source_snippets=['담당자 확인 필요'],
        confidence_score=0.99,
        permission_level='internal',
        status='pending_review',
    )

    request = build_candidate_validation_request(
        history, evidence_texts=tuple(history.source_snippets)
    )
    assert request is not None
    assert [claim.field_key for claim in request.claims] == ['title', 'reason']
    assert build_candidate_validation_request(
        todo, evidence_texts=tuple(todo.source_snippets)
    ) is None


def _context(**overrides: object) -> AutoReviewEligibilityContext:
    values: dict[str, object] = {
        'evidence_state': 'exact',
        'evidence_texts': ('배포가 완료됨', '인덱스 상태 정상'),
        'claim_fingerprint': 'a' * 64,
    }
    values.update(overrides)
    return AutoReviewEligibilityContext(**values)  # type: ignore[arg-type]


def test_service_returns_request_only_for_deterministically_eligible_input() -> None:
    result = AutoReviewEligibilityService().evaluate(
        _timeline_item(), context=_context()
    )

    assert result.decision == 'eligible'
    assert result.reason_codes == ('eligible',)
    assert result.claim_fingerprint == 'a' * 64
    assert result.validation_request is not None


def test_service_reports_sensitive_input_without_returning_plaintext_request() -> None:
    item = _timeline_item(
        result_summary='password = correct-horse-battery-staple'
    )

    result = AutoReviewEligibilityService().evaluate(item, context=_context())

    assert result.decision == 'human_review'
    assert result.reason_codes == ('sensitive_input_detected',)
    assert result.validation_request is None
    assert 'correct-horse' not in repr(result)


def test_service_forces_proposal_or_conditional_language_to_human_review() -> None:
    for summary in (
        '다음 주에 배포할 예정입니다.',
        'If the audit passes, we will deploy.',
    ):
        result = AutoReviewEligibilityService().evaluate(
            _timeline_item(result_summary=summary), context=_context()
        )
        assert result.decision == 'human_review'
        assert result.reason_codes == ('high_risk_language',)
        assert result.validation_request is None


def test_service_preserves_exact_duplicate_target_but_still_builds_validation() -> None:
    from backend.app.agent_runtime.auto_review_policy import TrustedTargetRef

    target = TrustedTargetRef('timeline_event', 9, 'a' * 64)
    result = AutoReviewEligibilityService().evaluate(
        _timeline_item(), context=_context(visible_targets=(target,))
    )

    assert result.decision == 'reuse_trusted'
    assert result.duplicate_target == target
    assert result.validation_request is not None


def _workflow_ref(**overrides: object) -> AgentWorkflowEvidenceRef:
    values: dict[str, object] = {
        'workflow_thread_id': 'thread-1',
        'ordinal': 1,
        'canonical_source_type': 'drive',
        'canonical_table': 'sources',
        'canonical_row_id': 7,
        'document_version_id': 11,
        'external_revision': 'r1',
        'content_signature': 'a' * 64,
        'permission_level_snapshot': 'internal',
        'content_fingerprint': 'b' * 64,
    }
    values.update(overrides)
    return AgentWorkflowEvidenceRef(**values)


def _resolved(**overrides: object) -> ResolvedSourceVersion:
    values: dict[str, object] = {
        'source_type': 'drive',
        'canonical_table': 'sources',
        'canonical_row_id': 7,
        'document_version_id': 11,
        'external_revision': 'r1',
        'content_signature': 'a' * 64,
        'permission_level': 'internal',
        'content_fingerprint': 'b' * 64,
    }
    values.update(overrides)
    return ResolvedSourceVersion(**values)


def test_bound_workflow_evidence_classification_separates_identity_and_drift() -> None:
    exact = classify_bound_workflow_evidence(
        ref=_workflow_ref(), resolved=_resolved()
    )
    identity_mismatch = classify_bound_workflow_evidence(
        ref=_workflow_ref(), resolved=_resolved(canonical_row_id=8)
    )
    permission_drift = classify_bound_workflow_evidence(
        ref=_workflow_ref(), resolved=_resolved(permission_level='restricted')
    )

    assert exact.state == 'exact'
    assert exact.resolved == _resolved()
    assert identity_mismatch.state == 'mismatch'
    assert identity_mismatch.resolved is None
    assert permission_drift.state == 'changed'
    assert permission_drift.resolved is None


def test_candidate_fingerprint_builds_exact_claim_project_and_title_buckets() -> None:
    settings = Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='x' * 32,
        agent_runtime_fingerprint_key_version='v1',
    )
    item = _timeline_item(project_key='project-alpha')

    first = build_trusted_candidate_fingerprint(
        item=item, security_scope_id='scope-a', settings=settings
    )
    second = build_trusted_candidate_fingerprint(
        item=item, security_scope_id='scope-a', settings=settings
    )

    assert first == second
    assert first.knowledge_type == 'timeline_event'
    assert len(first.project_scope_hmac) == 64
    assert len(first.normalized_title_bucket_hmac) == 64
    assert len(first.normalized_claim_fingerprint) == 64


def test_legacy_unknown_collision_is_guarded_even_when_its_permission_is_visible(
    db_session: Session,
) -> None:
    common = {
        'knowledge_type': 'timeline_event',
        'project_scope_hmac': 'p' * 64,
        'normalized_title_bucket_hmac': 't' * 64,
        'fingerprint_key_version': 'v1',
        'fingerprint_key_material_verifier': 'v' * 64,
        'permission_level': 'internal',
        'review_status': 'approved',
    }
    db_session.add_all(
        [
            TrustedKnowledgeFingerprint(
                knowledge_id=1,
                scope_resolution='exact',
                security_scope_id='scope-a',
                normalized_claim_fingerprint='a' * 64,
                **common,
            ),
            TrustedKnowledgeFingerprint(
                knowledge_id=2,
                scope_resolution='legacy_unknown',
                security_scope_id=None,
                normalized_claim_fingerprint=None,
                **common,
            ),
        ]
    )
    db_session.flush()

    visible = find_visible_trusted_collisions(
        db_session,
        knowledge_type='timeline_event',
        security_scope_id='scope-a',
        project_scope_hmac='p' * 64,
        normalized_title_bucket_hmac='t' * 64,
        visible_permission_levels=('public', 'internal'),
        candidate_permission_level='internal',
    )
    guarded = hidden_or_legacy_collision_exists(
        db_session,
        knowledge_type='timeline_event',
        security_scope_id='scope-a',
        project_scope_hmac='p' * 64,
        normalized_title_bucket_hmac='t' * 64,
        visible_permission_levels=('public', 'internal'),
        candidate_permission_level='internal',
    )

    assert [(row.knowledge_id, row.normalized_claim_fingerprint) for row in visible] == [
        (1, 'a' * 64)
    ]
    assert guarded is True


def test_different_permission_collision_cannot_be_reused_even_when_actor_sees_it(
    db_session: Session,
) -> None:
    db_session.add(
        TrustedKnowledgeFingerprint(
            knowledge_type='timeline_event',
            knowledge_id=3,
            scope_resolution='exact',
            security_scope_id='scope-a',
            project_scope_hmac='p' * 64,
            normalized_title_bucket_hmac='t' * 64,
            normalized_claim_fingerprint='a' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='v' * 64,
            permission_level='public',
            review_status='approved',
        )
    )
    db_session.flush()

    visible = find_visible_trusted_collisions(
        db_session,
        knowledge_type='timeline_event',
        security_scope_id='scope-a',
        project_scope_hmac='p' * 64,
        normalized_title_bucket_hmac='t' * 64,
        visible_permission_levels=('public', 'internal'),
        candidate_permission_level='internal',
    )
    guarded = hidden_or_legacy_collision_exists(
        db_session,
        knowledge_type='timeline_event',
        security_scope_id='scope-a',
        project_scope_hmac='p' * 64,
        normalized_title_bucket_hmac='t' * 64,
        visible_permission_levels=('public', 'internal'),
        candidate_permission_level='internal',
    )

    assert visible == ()
    assert guarded is True


def _seed_database_candidate(
    db_session: Session,
    monkeypatch,
) -> tuple[
    ReviewItem,
    Settings,
    AgentRegistry,
    tuple[AutoReviewEvidenceMessage, ...],
]:
    settings = Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='x' * 32,
        agent_runtime_fingerprint_key_version='v1',
    )
    thread = AgentWorkflowThread(
        thread_id='task7-thread',
        workflow_name='company-memory-review',
        graph_version='company-memory-review-v2.1-auto-review',
        checkpoint_thread_id='checkpoint:task7-thread',
        checkpoint_store='memory',
        owner_subject_id='owner-1',
        security_scope_id='scope-a',
        input_hash='1' * 64,
        evidence_version_hash='2' * 64,
        status='awaiting_review',
    )
    run = AgentRun(
        agent_name='timeline_agent',
        prompt_version='timeline-extraction:c5-v1',
        status='complete',
        source_window='bounded',
        cache_key='task7-cache',
        model_name='gpt-5.4-mini-2026-03-17',
        generation_provider='openai',
        generation_reasoning_effort='none',
        generation_route_version='auto-review-extraction-route:v1',
        generation_output_contract_version='timeline-candidate:c5-v1',
        permission_level='internal',
        workflow_thread_id=thread.thread_id,
        effect_key='task7-effect',
    )
    db_session.add_all([thread, run])
    db_session.flush()
    item = _timeline_item()
    item.workflow_thread_id = thread.thread_id
    item.candidate_contract_version = 'c5-v1'
    item.agent_run_id = run.id
    item.payload.update(
        {
            'agent_name': run.agent_name,
            'agent_run_id': run.id,
            'prompt_version': run.prompt_version,
            'cache_key': run.effect_key,
            'estimated_cost_usd': 0.01,
            'token_usage': {
                'input_tokens': 10,
                'output_tokens': 5,
                'total_tokens': 15,
            },
            'uncertainty_reason': None,
            'source_ids': ['drive:task7'],
            'source_types': ['drive'],
            'source_authors': [],
        }
    )
    item.candidate_key = build_keyed_fingerprint(
        {
            'effect_key': run.effect_key,
            'candidate': {
                'item_type': item.item_type,
                'title': item.payload['title'],
                'summary': item.payload['summary'],
                'source_links': list(item.source_links),
                'source_snippets': list(item.source_snippets),
                'confidence_score': item.confidence_score,
                'permission_level': item.permission_level,
                'uncertainty_reason': item.payload['uncertainty_reason'],
                'payload_fields': {
                    'result_summary': item.payload['result_summary']
                },
            },
        },
        settings=settings,
        schema_version='review-agent-candidate:v1',
        policy_version='review-agent-candidate-key:v1',
    )
    ref = _workflow_ref(workflow_thread_id=thread.thread_id)
    db_session.add_all([item, ref])
    db_session.flush()
    secret, key_version = fingerprint_secret_bytes(settings)
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    messages = (
        AutoReviewEvidenceMessage(
            workflow_evidence_ref_id=ref.id,
            stable_message_identity='message-1',
            text='배포가 완료됨',
            permission_level='internal',
        ),
        AutoReviewEvidenceMessage(
            workflow_evidence_ref_id=ref.id,
            stable_message_identity='message-2',
            text='인덱스 상태 정상',
            permission_level='internal',
        ),
    )
    message_set_hmac = build_keyed_fingerprint(
        {
            'canonical_source_kind': ref.canonical_source_type,
            'canonical_source_id': ref.canonical_row_id,
            'canonical_version_or_signature': (
                ref.external_revision or ref.content_signature
            ),
            'content_fingerprint': ref.content_fingerprint,
            'permission_level': 'internal',
            'fingerprint_key_version': key_version,
            'fingerprint_key_material_verifier': verifier,
            'messages': [
                {
                    'stable_message_identity': message.stable_message_identity,
                    'text_fingerprint': build_keyed_fingerprint(
                        message.text,
                        settings=settings,
                        schema_version='candidate-message-content:v1',
                        policy_version='candidate-message-content:v1',
                    ),
                }
                for message in messages
            ],
        },
        settings=settings,
        schema_version='candidate-message-set:v1',
        policy_version='candidate-message-set:v1',
    )
    db_session.add(
        ReviewItemEvidenceRef(
            review_item_id=item.id,
            workflow_thread_id=thread.thread_id,
            workflow_evidence_ref_id=ref.id,
            candidate_slot_ordinal=1,
            message_content_fingerprint=message_set_hmac,
            fingerprint_key_version=key_version,
            fingerprint_key_material_verifier=verifier,
        )
    )
    db_session.flush()
    monkeypatch.setattr(
        'backend.app.agent_runtime.auto_review_eligibility.resolve_bound_workflow_evidence',
        lambda *args, **kwargs: BoundWorkflowEvidenceResolution(
            'exact', _resolved()
        ),
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.auto_review_eligibility.trusted_fingerprint_projection_ready',
        lambda *args, **kwargs: True,
    )
    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            name='timeline_agent',
            owner='Developer C',
            input_contract='EvidencePacket',
            output_contract='AgentRunResult',
            prompt_versions=('timeline-extraction:c5-v1',),
            supported_permissions=('public', 'internal'),
            capabilities=('timeline_extraction',),
        )
    )
    return item, settings, registry, messages


def test_database_service_revalidates_bound_evidence_and_generation_identity(
    db_session: Session,
    monkeypatch,
) -> None:
    item, settings, registry, messages = _seed_database_candidate(
        db_session, monkeypatch
    )

    result = AutoReviewEligibilityService().evaluate_from_database(
        db_session,
        item,
        settings=settings,
        registry=registry,
        visible_permission_levels=('public', 'internal'),
        evidence_messages=messages,
        budget_available=True,
    )

    assert result.decision == 'eligible'
    assert result.validation_request is not None
    assert result.reason_codes == ('eligible',)


def test_database_service_fails_closed_when_registry_identity_is_missing(
    db_session: Session,
    monkeypatch,
) -> None:
    item, settings, _, messages = _seed_database_candidate(db_session, monkeypatch)

    result = AutoReviewEligibilityService().evaluate_from_database(
        db_session,
        item,
        settings=settings,
        registry=AgentRegistry(),
        visible_permission_levels=('public', 'internal'),
        evidence_messages=messages,
        budget_available=True,
    )

    assert result.decision == 'human_review'
    assert result.reason_codes == ('registry_unavailable',)
    assert result.validation_request is None


def test_database_service_rejects_wrong_manifest_output_contract(
    db_session: Session,
    monkeypatch,
) -> None:
    item, settings, _, messages = _seed_database_candidate(db_session, monkeypatch)
    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            name='timeline_agent',
            owner='Developer C',
            input_contract='EvidencePacket',
            output_contract='unreviewed-local-shape',
            prompt_versions=('timeline-extraction:v1',),
            supported_permissions=('public', 'internal'),
            capabilities=('timeline_extraction',),
        )
    )

    result = AutoReviewEligibilityService().evaluate_from_database(
        db_session,
        item,
        settings=settings,
        registry=registry,
        visible_permission_levels=('public', 'internal'),
        evidence_messages=messages,
        budget_available=True,
    )

    assert result.decision == 'human_review'
    assert result.reason_codes == ('registry_unavailable',)


def test_database_service_rejects_tampered_plaintext_before_validation(
    db_session: Session,
    monkeypatch,
) -> None:
    item, settings, registry, messages = _seed_database_candidate(
        db_session, monkeypatch
    )
    tampered = (replace(messages[0], text='변조된 메시지'), messages[1])

    result = AutoReviewEligibilityService().evaluate_from_database(
        db_session,
        item,
        settings=settings,
        registry=registry,
        visible_permission_levels=('public', 'internal'),
        evidence_messages=tampered,
        budget_available=True,
    )

    assert result.decision == 'human_review'
    assert result.reason_codes == ('evidence_binding_mismatch',)
    assert result.validation_request is None


def test_database_service_rejects_candidate_payload_drift_before_validation(
    db_session: Session,
    monkeypatch,
) -> None:
    item, settings, registry, messages = _seed_database_candidate(
        db_session, monkeypatch
    )
    item.payload['result_summary'] = '사후에 바뀐 주장'
    db_session.flush()

    result = AutoReviewEligibilityService().evaluate_from_database(
        db_session,
        item,
        settings=settings,
        registry=registry,
        visible_permission_levels=('public', 'internal'),
        evidence_messages=messages,
        budget_available=True,
    )

    assert result.decision == 'human_review'
    assert result.reason_codes == ('evidence_binding_mismatch',)
    assert result.validation_request is None


def test_database_service_has_no_caller_scope_or_projection_override() -> None:
    parameters = signature(
        AutoReviewEligibilityService.evaluate_from_database
    ).parameters

    assert 'security_scope_id' not in parameters
    assert 'fingerprint_projection_ready' not in parameters


def test_database_service_derives_scope_from_stored_workflow(
    db_session: Session,
    monkeypatch,
) -> None:
    item, settings, registry, messages = _seed_database_candidate(
        db_session, monkeypatch
    )
    observed: dict[str, str] = {}
    original = build_trusted_candidate_fingerprint

    def record_scope(*, item, security_scope_id, settings):
        observed['security_scope_id'] = security_scope_id
        return original(
            item=item,
            security_scope_id=security_scope_id,
            settings=settings,
        )

    monkeypatch.setattr(
        'backend.app.agent_runtime.auto_review_eligibility.build_trusted_candidate_fingerprint',
        record_scope,
    )

    result = AutoReviewEligibilityService().evaluate_from_database(
        db_session,
        item,
        settings=settings,
        registry=registry,
        visible_permission_levels=('public', 'internal'),
        evidence_messages=messages,
        budget_available=True,
    )

    assert result.decision == 'eligible'
    assert observed == {'security_scope_id': 'scope-a'}


def test_database_service_fails_closed_when_projection_is_not_server_ready(
    db_session: Session,
    monkeypatch,
) -> None:
    item, settings, registry, messages = _seed_database_candidate(
        db_session, monkeypatch
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.auto_review_eligibility.trusted_fingerprint_projection_ready',
        lambda *args, **kwargs: False,
    )

    result = AutoReviewEligibilityService().evaluate_from_database(
        db_session,
        item,
        settings=settings,
        registry=registry,
        visible_permission_levels=('public', 'internal'),
        evidence_messages=messages,
        budget_available=True,
    )

    assert result.decision == 'human_review'
    assert result.reason_codes == ('trusted_lookup_unavailable',)
    assert result.validation_request is None
