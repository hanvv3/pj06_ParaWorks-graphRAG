from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from backend.app.agent_runtime.rag_advisory_locks import (
    acquire_advisory_lock,
    load_registered_advisory_capability,
    rag_projection_owner_lock_id,
    register_advisory_identity_db,
    release_advisory_lock,
    try_acquire_advisory_lock,
)
from backend.app.agent_runtime.rag_finalization import (
    AssistantProjectionTarget,
    PreparedRagFinalization,
    RagFinalizationError,
    RagFinalizationService,
    RagProjectionPending,
    SqlAlchemyRagFinalizationBoundary,
    _assemble_projection_owner_recovery_authority,
    _pre_projection_cost_snapshot_hmac,
)
from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    security_scope_fingerprint,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    prepare_direct_request_text,
)
from backend.app.core.config import Settings
from backend.app.rag.evidence_projection import V1EvidenceProjection
from backend.app.rag.retrieval import RetrievalResult, SanitizedRetrievalTrace


def _prepared() -> PreparedRagFinalization:
    text = prepare_direct_request_text('  민감한 질의  ', key=b'k')
    scope = SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='private-principal',
        workspace_scope_id='scope',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=('public', 'internal'),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )
    retrieval = RetrievalResult(
        configured_backend='keyword',
        effective_backend='deterministic_lexical',
        visible=(),
        hidden_match_count=2,
        hidden_count_capped=False,
        top_candidate_window_hmac='1' * 64,
        query_embedding_receipt=None,
        trace=SanitizedRetrievalTrace(2, 0, 2, 0, 1, None),
    )
    return PreparedRagFinalization(
        product_kind='answer',
        tentative_outcome='hidden_only',
        prepared_text=text,
        security_scope=scope,
        query_embedding_result=None,
        retrieval_result=retrieval,
        evidence_slots=(),
        model_influence_observations=(),
        selected_slot_ids=(),
        validated_answer=None,
        canned_message_identity='rag-canned-no-evidence:v1',
    )


def _pending() -> RagProjectionPending:
    settings = Settings(
        _env_file=None, agent_runtime_fingerprint_secret='secret'
    )
    return RagProjectionPending(
        parent_agent_run_id=3,
        projection_owner_fence_hmac='2' * 64,
        security_scope_fingerprint=security_scope_fingerprint(
            _prepared().security_scope, settings=settings
        ),
        prepared_corpus_generation=5,
        prepared_vector_index_generation=None,
        terminal_cost_snapshot_hmac='4' * 64,
    )


class _Boundary:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.retrievals = 0
        self.final_parent = None
        self.fail_commit = False

    def begin(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.rollbacks += 1
        return False

    def validate_pending(self, pending):
        assert pending.parent_agent_run_id == 3

    def current_generations(self):
        return 5, None

    def retrieve_fresh(self, request):
        self.retrievals += 1
        assert request.retrieval_query_text == '  민감한 질의  '
        assert request.security_scope.principal_subject == 'private-principal'
        return _prepared().retrieval_result

    def project(self, prepared, fresh, *, drifted):
        assert drifted is False
        return SimpleNamespace(
            outcome='hidden_only', answer_text='근거를 찾지 못했습니다.',
            evidence=V1EvidenceProjection((), (), (), (), (), '5' * 64),
            model_influence=(), hidden_match_count=2,
            effective_backend='deterministic_lexical', result_hmac='6' * 64,
        )

    def finalize_parent(self, pending, projection):
        self.final_parent = (pending, projection)

    def commit(self):
        if self.fail_commit:
            raise ConnectionError('commit ack missing')
        self.commits += 1


def test_prepared_carrier_is_frozen_and_request_local_complete() -> None:
    prepared = _prepared()
    assert prepared.prepared_text.caller_text == '  민감한 질의  '
    assert prepared.security_scope.principal_subject == 'private-principal'
    assert prepared.retrieval_result.hidden_match_count == 2
    with pytest.raises(FrozenInstanceError):
        prepared.tentative_outcome = 'no_match'  # type: ignore[misc]


def test_direct_result_is_returned_only_after_fresh_projection_commit() -> None:
    boundary = _Boundary()
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    result = service.finalize_provider_free_safe(_pending(), _prepared())

    assert result.outcome == 'hidden_only'
    assert boundary.retrievals == 1
    assert boundary.final_parent is not None
    assert boundary.commits == 1


def test_commit_failure_returns_no_projection_and_does_not_retry() -> None:
    boundary = _Boundary()
    boundary.fail_commit = True
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )
    with pytest.raises(RagFinalizationError, match='commit'):
        service.finalize_provider_free_safe(_pending(), _prepared())
    assert boundary.retrievals == 1
    assert boundary.commits == 0


def test_pending_validation_recomputes_exact_terminal_cost_snapshot() -> None:
    def child(component: str, ordinal: int) -> SimpleNamespace:
        return SimpleNamespace(
            actual_input_tokens=None,
            actual_output_tokens=None,
            attempted=False,
            authorized_cost_policy_version='rag-cost-policy:v2',
            authorized_model_config_snapshot_hmac='7' * 64,
            authorized_model_config_version='model-config:v1',
            authorized_policy_snapshot_hmac='8' * 64,
            authorized_token_estimator_version='token-estimator:v1',
            charge_basis='zero',
            charged_cost_usd=Decimal('0.000000'),
            component=component,
            component_ordinal=ordinal,
            dispatch_count=0,
            dispatch_fence_hmac=None,
            dispatch_state='terminal',
            model='none',
            overrun=False,
            process_instance_hmac=None,
            provider='none',
            reserved_cost_usd=Decimal('0.000000'),
            reserved_input_tokens=0,
            reserved_output_tokens=0,
        )

    children = [child('query_embedding', 0), child('answer_generation', 1)]
    secret = b'secret'
    snapshot = _pre_projection_cost_snapshot_hmac(
        agent_run_id=3, children=tuple(children), secret=secret
    )
    pending = replace(_pending(), terminal_cost_snapshot_hmac=snapshot)
    parent = SimpleNamespace(
        id=3,
        run_contract_version='rag-run:v2',
        status='running',
        run_record_phase='cost_finalized_pending_projection',
        completed_at=None,
        projection_owner_fence_hmac=pending.projection_owner_fence_hmac,
        total_charged_cost_usd=Decimal('0.000000'),
        metadata_={
            'security_scope_fingerprint': pending.security_scope_fingerprint,
            'runtime_cost_snapshot_hmac': snapshot,
        },
    )
    db = SimpleNamespace(
        scalar=lambda _statement: parent,
        scalars=lambda _statement: tuple(children),
    )
    boundary = object.__new__(SqlAlchemyRagFinalizationBoundary)
    boundary._db = db
    boundary._secret = secret
    boundary._prefix_generations = (5, None)

    boundary.validate_pending(pending)
    children[0].attempted = True

    with pytest.raises(RagFinalizationError, match='pending RAG projection changed'):
        boundary.validate_pending(pending)


def test_assistant_target_uses_existing_string_owner_identity() -> None:
    target = AssistantProjectionTarget(7, 8, 'owner-string')
    assert target.owner_user_id == 'owner-string'
    with pytest.raises(ValueError):
        replace(target, owner_user_id='')


def test_assistant_finalization_rejects_search_product_before_transaction() -> None:
    boundary = _Boundary()
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    with pytest.raises(RagFinalizationError, match='answer product'):
        service.finalize_assistant(
            _pending(),
            replace(_prepared(), product_kind='search'),
            AssistantProjectionTarget(7, 8, 'owner-string'),
        )

    assert boundary.retrievals == 0
    assert boundary.commits == 0


def test_projection_recovery_has_only_exact_lock_reacquire_authority() -> None:
    seen: list[int] = []
    authority = _assemble_projection_owner_recovery_authority(
        lambda run_id: seen.append(run_id) or None
    )
    boundary = object.__new__(SqlAlchemyRagFinalizationBoundary)
    boundary._recovery_authority = authority
    assert boundary.recover_dead_projection_owner(17) is None
    assert seen == [17]
    import inspect

    assert tuple(
        inspect.signature(boundary.recover_dead_projection_owner).parameters
    ) == ('run_id',)


@pytest.mark.skipif(
    not os.environ.get('PARAWORKS_TEST_POSTGRES_URL'),
    reason='disposable PostgreSQL projection-owner gate is not configured',
)
def test_postgres_live_projection_owner_blocks_recovery_until_exact_reacquire() -> None:
    engine = create_engine(os.environ['PARAWORKS_TEST_POSTGRES_URL'])
    run_id = (uuid4().int % (2**31 - 1)) + 1
    identity = rag_projection_owner_lock_id(run_id)
    with engine.begin() as connection:
        register_advisory_identity_db(
            connection, identity, identity_namespace='dynamic'
        )
    owner = engine.connect()
    recovery = engine.connect()
    try:
        capability = load_registered_advisory_capability(
            owner, identity, identity_namespace='dynamic'
        )
        owner.rollback()
        acquire_advisory_lock(owner, capability, shared=False)
        assert try_acquire_advisory_lock(
            recovery, capability, shared=False
        ) is False
        release_advisory_lock(owner, capability, shared=False)
        assert try_acquire_advisory_lock(
            recovery, capability, shared=False
        ) is True
        release_advisory_lock(recovery, capability, shared=False)
    finally:
        owner.close()
        recovery.close()
        engine.dispose()
