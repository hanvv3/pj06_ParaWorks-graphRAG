from dataclasses import replace

import pytest

from backend.app.agent_runtime.contracts import (
    AgentRunCost,
    AgentRunResult,
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
    ReviewCandidate,
    TokenUsage,
)
from backend.app.agent_runtime.review_v2_drafting import (
    CandidateEvidenceBindingError,
    GenerationIdentity,
    build_candidate_evidence_bindings,
    build_candidate_generation_fingerprint,
    generation_identity_is_auto_eligible,
)
from backend.app.core.config import Settings


def _settings() -> Settings:
    return Settings(
        agent_runtime_fingerprint_secret='candidate-binding-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='binding-v1',
    )


def _message(
    index: int, *, snippet: str | None = None, permission: str = 'internal'
) -> EvidenceMessage:
    text = snippet or f'evidence-{index}'
    return EvidenceMessage(
        source_id=f'source-{index}',
        source_url=f'https://evidence.test/{index}',
        text=text,
        author='author',
        timestamp=f'2026-08-28T00:00:0{index}Z',
        permission_level=permission,
        source_snippet_override=text,
        metadata={
            'workflow_evidence_ref_id': index,
            'canonical_source_kind': 'gmail',
            'canonical_source_id': index,
            'canonical_version_or_signature': f'version-{index}',
            'content_fingerprint': f'{index}' * 64,
            'stable_message_identity': f'message-{index}',
        },
    )


def _packet(*messages: EvidenceMessage) -> EvidencePacket:
    return EvidencePacket(
        source_type='company_memory',
        source_window='window',
        messages=list(messages),
        permission_context=PermissionContext('owner', 'employee'),
    )


def _candidate(*messages: EvidenceMessage) -> ReviewCandidate:
    return ReviewCandidate(
        item_type='history_event',
        title='title',
        summary='summary',
        source_links=[message.source_url for message in messages],
        source_snippets=[message.source_snippet for message in messages],
        confidence_score=0.9,
        permission_level='internal',
        payload_fields={'reason': 'reason'},
    )


def _bindings(candidate=None, packet=None):
    messages = (_message(1), _message(2))
    return build_candidate_evidence_bindings(
        candidate=candidate or _candidate(*messages),
        packet=packet or _packet(*messages),
        workflow_execution_hmac='a' * 64,
        security_scope_hmac='b' * 64,
        candidate_key='candidate-key',
        settings=_settings(),
        fingerprint_key_material_verifier='c' * 64,
    )


def test_v20_and_v21_drafts_link_each_candidate_to_exact_workflow_evidence_refs():
    result = _bindings()
    assert [row.workflow_evidence_ref_id for row in result.refs] == [1, 2]
    assert [row.ordinal for row in result.refs] == [1, 2]


def test_v20_and_v21_drafts_link_each_candidate_to_same_workflow_agent_run():
    result = _bindings()
    assert result.candidate_contract_version == 'c5-v1'
    assert result.evidence_version_hash


def test_cross_workflow_agent_run_reference_is_rejected_by_database():
    with pytest.raises(CandidateEvidenceBindingError, match='same workflow'):
        _bindings().assert_same_workflow_agent_run(
            workflow_thread_id='workflow-a',
            agent_run_workflow_thread_id='workflow-b',
        )


def test_cross_workflow_evidence_ref_reference_is_rejected_by_database():
    message = replace(
        _message(1),
        metadata={**_message(1).metadata, 'workflow_thread_id': 'workflow-b'},
    )
    with pytest.raises(CandidateEvidenceBindingError, match='same workflow'):
        build_candidate_evidence_bindings(
            candidate=_candidate(message),
            packet=_packet(message),
            workflow_execution_hmac='a' * 64,
            security_scope_hmac='b' * 64,
            candidate_key='candidate-key',
            settings=_settings(),
            fingerprint_key_material_verifier='c' * 64,
            workflow_thread_id='workflow-a',
        )


def test_agent_run_relational_replay_mismatch_is_invalid_not_payload_repair():
    with pytest.raises(CandidateEvidenceBindingError, match='immutable'):
        _bindings().verify_replay(
            agent_run_id=2, stored_agent_run_id=3, stored_refs=(1, 2)
        )


def test_candidate_and_evidence_refs_rollback_together():
    assert _bindings().atomic_rows()[0].candidate_contract_version == 'c5-v1'


def test_candidate_replay_neither_duplicates_nor_repairs_ambiguous_refs_silently():
    with pytest.raises(CandidateEvidenceBindingError, match='immutable'):
        _bindings().verify_replay(
            agent_run_id=2, stored_agent_run_id=2, stored_refs=(1,)
        )


def test_v20_binding_remains_auto_review_ineligible_by_graph_version():
    assert not _bindings().is_auto_review_eligible('company-memory-review-v2.0')


def test_two_messages_for_one_source_ref_use_one_sorted_aggregate_fingerprint():
    first = _message(1)
    second = replace(
        _message(2),
        metadata={
            **first.metadata,
            'stable_message_identity': 'message-2',
        },
    )
    result = _bindings(_candidate(first, second), _packet(second, first))
    assert len(result.refs) == 1
    assert len(result.refs[0].message_set_hmac) == 64


def test_candidate_evidence_state_hash_is_ref_order_invariant():
    first, second = _message(1), _message(2)
    assert (
        _bindings(
            _candidate(first, second), _packet(first, second)
        ).evidence_version_hash
        == _bindings(
            _candidate(second, first), _packet(second, first)
        ).evidence_version_hash
    )


@pytest.mark.parametrize(
    ('metadata_key', 'value'),
    (
        ('canonical_version_or_signature', 'changed-version'),
        ('content_fingerprint', 'f' * 64),
        ('stable_message_identity', 'changed-message'),
    ),
)
def test_candidate_evidence_state_hash_changes_for_message_version_signature_permission_or_key_identity(
    metadata_key, value
):
    base = _message(1)
    changed = replace(base, metadata={**base.metadata, metadata_key: value})
    assert (
        _bindings(_candidate(base), _packet(base)).evidence_version_hash
        != _bindings(_candidate(changed), _packet(changed)).evidence_version_hash
    )


def test_duplicate_snippet_mapping_is_ambiguous_and_never_guessed():
    first = _message(1, snippet='same')
    second = replace(_message(2, snippet='same'), source_url=first.source_url)
    with pytest.raises(CandidateEvidenceBindingError, match='ambiguous'):
        _bindings(_candidate(first), _packet(first, second))


def test_ambiguous_candidate_mapping_persists_no_review_item_or_partial_binding():
    first = _message(1, snippet='same')
    second = replace(_message(2, snippet='same'), source_url=first.source_url)
    writes = []
    with pytest.raises(CandidateEvidenceBindingError):
        build_candidate_evidence_bindings(
            candidate=_candidate(first),
            packet=_packet(first, second),
            workflow_execution_hmac='a' * 64,
            security_scope_hmac='b' * 64,
            candidate_key='candidate-key',
            settings=_settings(),
            fingerprint_key_material_verifier='c' * 64,
            on_persist=writes.append,
        )
    assert writes == []


def _identity() -> GenerationIdentity:
    return GenerationIdentity(
        agent_name='history_agent',
        provider='openai',
        model='generator-model',
        reasoning_effort='none',
        prompt_version='history-extraction:c5-v1',
        route_version='auto-review-extraction-route:v1',
        output_contract_version='history-candidate:c5-v1',
    )


def test_generation_identity_is_stable_and_contains_no_raw_evidence():
    value = build_candidate_generation_fingerprint(
        _identity(), settings=_settings(), fingerprint_key_material_verifier='c' * 64
    )
    assert len(value) == 64
    assert 'evidence' not in value


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('agent_name', 'timeline_agent'),
        ('provider', 'gemini'),
        ('model', 'other'),
        ('reasoning_effort', 'medium'),
        ('prompt_version', 'other:v1'),
        ('route_version', 'other-route:v1'),
        ('output_contract_version', 'other:v1'),
    ),
)
def test_generation_fingerprint_changes_for_agent_provider_model_reasoning_prompt_route_or_output_contract(
    field, value
):
    baseline = build_candidate_generation_fingerprint(
        _identity(), settings=_settings(), fingerprint_key_material_verifier='c' * 64
    )
    changed = build_candidate_generation_fingerprint(
        replace(_identity(), **{field: value}),
        settings=_settings(),
        fingerprint_key_material_verifier='c' * 64,
    )
    assert changed != baseline


def test_v21_null_or_wrong_generation_reasoning_is_ineligible_before_validation():
    assert not generation_identity_is_auto_eligible(
        replace(_identity(), reasoning_effort=None),
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
    )
    assert not generation_identity_is_auto_eligible(
        replace(_identity(), reasoning_effort='medium'),
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
    )


def test_missing_generation_provider_stays_valid_for_v20_but_auto_ineligible():
    result = AgentRunResult(
        agent_name='history_agent',
        prompt_version='v1',
        candidates=[],
        cost=AgentRunCost('model', TokenUsage(1, 1), 0, False),
        cache_key='key',
    )
    assert result.model_provider is None
    assert not generation_identity_is_auto_eligible(
        GenerationIdentity('history_agent', None, 'model', None, 'v1', None, None),
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
    )
