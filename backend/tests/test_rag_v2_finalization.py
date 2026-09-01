from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine

from backend.app.agent_runtime.provider_send_fence import (
    _assemble_rag_evidence_barrier,
)
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
    acquire_advisory_lock,
    load_registered_advisory_capability,
    rag_projection_owner_lock_id,
    register_advisory_identity_db,
    release_advisory_lock,
    try_acquire_advisory_lock,
)
from backend.app.agent_runtime.rag_cost_ledger import (
    PendingProjectionRecoverySnapshot,
    RagCostLedger,
)
from backend.app.agent_runtime.rag_finalization import (
    AssistantProjectionTarget,
    CanonicalRagProjection,
    PreparedRagFinalization,
    RagFinalizationError,
    RagFinalizationService,
    RagProjectionPending,
    SqlAlchemyRagFinalizationBoundary,
    _assemble_paid_rag_phase2_authority,
    _assemble_projection_owner_recovery_authority,
    _assemble_provider_free_rag_phase2_authority,
    _pre_projection_cost_snapshot_hmac,
)
from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService
from backend.app.agent_runtime.rag_runtime_contracts import RagProviderSafetyBinding
from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    security_scope_fingerprint,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ValidatedAnswerBlocks,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    prepare_direct_request_text,
)
from backend.app.core.config import Settings
from backend.app.models.rag_runtime import RagAdvisoryLockKey
from backend.app.rag.evidence_projection import (
    PreparedModelInfluenceObservation,
    PreparedModelInfluenceSet,
    V1EvidenceProjection,
)
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import RetrievalResult, SanitizedRetrievalTrace
from backend.app.rag.serving_locks import ServingProjectionReadCoordinator
from backend.tests.test_rag_v2_costs import _snapshot
from backend.tests.test_rag_v2_pgvector_retriever import _embedding_result


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


def _cost_child(component: str, ordinal: int) -> SimpleNamespace:
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
        terminal_outcome=None,
    )


def _pending_boundary(
    children: list[SimpleNamespace],
) -> tuple[SqlAlchemyRagFinalizationBoundary, RagProjectionPending]:
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
        total_charged_cost_usd=sum(
            (value.charged_cost_usd for value in children), Decimal('0.000000')
        ),
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
    return boundary, pending


def _owner_capability(run_id: int):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    RagAdvisoryLockKey.__table__.create(engine)
    identity = rag_projection_owner_lock_id(run_id)
    with engine.begin() as connection:
        register_advisory_identity_db(
            connection, identity, identity_namespace='dynamic'
        )
    with engine.connect() as connection:
        capability = load_registered_advisory_capability(
            connection, identity, identity_namespace='dynamic'
        )
        connection.rollback()
    engine.dispose()
    return capability


def _evidence_capability():
    engine = create_engine('sqlite+pysqlite:///:memory:')
    RagAdvisoryLockKey.__table__.create(engine)
    with engine.begin() as connection:
        register_advisory_identity_db(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
    with engine.connect() as connection:
        capability = load_registered_advisory_capability(
            connection,
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            identity_namespace='static',
        )
        connection.rollback()
    engine.dispose()
    return capability


class _ClosableConnection:
    def __init__(self, engine: object | None = None) -> None:
        self.engine = engine
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _readiness(*, snapshot: str = '4' * 64) -> RagServingIndexReadiness:
    return RagServingIndexReadiness(
        ready=True,
        corpus_generation=1,
        vector_index_generation=1,
        expected_document_count=1,
        live_vector_count=1,
        tombstone_count=0,
        mismatch_count_capped_at_20=0,
        embedding_model='test-embedding',
        embedding_dimensions=3,
        index_policy_version='rag-index-policy:v2',
        readiness_snapshot_hmac=snapshot,
    )


def _safety_requirement(component: str):
    snapshot = _snapshot(component, None)
    return snapshot, RagProviderSafetyBinding(
        policy_snapshot=snapshot,
        authority_uuid=UUID('00000000-0000-0000-0000-000000000001'),
        designated_environment_id='test',
        envelope_digest='1' * 64,
        fingerprint_key_version='fingerprint-key:v1',
        fingerprint_key_material_verifier='2' * 64,
        global_safety_generation=1,
        family_state='ready',
        family_state_version=1,
        family_safety_generation=1,
        provider_safety_snapshot_hmac='3' * 64,
    )


def _post_generation_insufficient() -> PreparedRagFinalization:
    observation = PreparedModelInfluenceObservation(
        ordinal=0,
        slot_id='E1',
        support_mode='trusted_fact',
        lookup_identity=SimpleNamespace(),
        effective_permission='internal',
        serving_identity_hmac='5' * 64,
        serving_version_fingerprint='6' * 64,
        model_content_hmac='7' * 64,
        canonical_citation_projection_hmac='8' * 64,
        approval_provenance_hmac=None,
        evidence_link_set_hmac=None,
        observation_hmac='9' * 64,
    )
    influence = PreparedModelInfluenceSet(
        observations=(observation,),
        prepared_corpus_generation=5,
        prepared_index_generation=None,
        prepared_readiness_hmac=None,
        rendered_input_hmac='a' * 64,
        aggregate_observation_hmac='b' * 64,
    )
    return replace(
        _prepared(),
        tentative_outcome='insufficient_evidence',
        evidence_slots=(SimpleNamespace(slot_id='E1'),),
        model_influence_observations=(observation,),
        selected_slot_ids=(),
        validated_answer=ValidatedAnswerBlocks(
            blocks=(),
            insufficient_reason='raw provider reason must never persist',
            selected_slot_ids=(),
            assembled_answer='',
            assembled_answer_hmac=None,
            answer_block_audit_set_hmac=None,
        ),
        canned_message_identity='rag-canned-no-evidence:v1',
        prepared_model_influence=influence,
        rendered_input_hmac=influence.rendered_input_hmac,
        answer_model_config_snapshot_hmac='c' * 64,
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

    def acquire_phase2(self, pending, prepared, *, branch):
        del pending, prepared, branch
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.rollbacks += 1
        return False

    def validate_pending(self, pending):
        assert pending.parent_agent_run_id == 3

    def validate_branch_costs(self, prepared, *, branch):
        del prepared
        assert branch in {'provider_free', 'paid_embedding_only', 'paid_prepared'}

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


class _OrderedBoundary(_Boundary):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def acquire_phase2(self, pending, prepared, *, branch):
        boundary = self

        class _Gate:
            def __enter__(self):
                boundary.events.append(f'phase2:{branch}')

            def __exit__(self, exc_type, exc, tb):
                boundary.events.append('phase2:release')

        return _Gate()

    def acquire_projection_prefix(self):
        self.events.append('prefix')

    def current_generations(self):
        self.events.append('generations')
        return super().current_generations()

    def retrieve_fresh(self, request):
        self.events.append('retrieve')
        return super().retrieve_fresh(request)

    def project(self, prepared, fresh, *, drifted):
        self.events.append('canonical')
        return super().project(prepared, fresh, drifted=drifted)

    def validate_pending(self, pending):
        self.events.append('run_cost_tail')
        return super().validate_pending(pending)

    def validate_prepared(self, pending, prepared):
        self.events.append('prepared_binding')

    def finalize_parent(self, pending, projection):
        self.events.append('parent_final')
        return super().finalize_parent(pending, projection)

    def commit(self):
        self.events.append('commit')
        return super().commit()


def test_prepared_carrier_is_frozen_and_request_local_complete() -> None:
    prepared = _prepared()
    assert prepared.prepared_text.caller_text == '  민감한 질의  '
    assert prepared.security_scope.principal_subject == 'private-principal'
    assert prepared.retrieval_result.hidden_match_count == 2
    with pytest.raises(FrozenInstanceError):
        prepared.tentative_outcome = 'no_match'  # type: ignore[misc]


def test_prepared_carrier_rejects_terminal_error_outcome() -> None:
    with pytest.raises(ValueError, match='product outcome'):
        replace(
            _prepared(),
            tentative_outcome='budget_exceeded',
            canned_message_identity='rag-canned-budget-failure:v1',
        )


def test_prepared_carrier_rejects_mismatched_canned_identity() -> None:
    with pytest.raises(ValueError, match='canned identity'):
        replace(
            _prepared(),
            canned_message_identity='rag-canned-evidence-unavailable:v1',
        )


def test_provider_free_rejects_insufficient_evidence_before_transaction() -> None:
    boundary = _Boundary()
    prepared = _prepared()
    object.__setattr__(prepared, 'tentative_outcome', 'insufficient_evidence')
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    with pytest.raises(RagFinalizationError, match='post-generation'):
        service.finalize_provider_free_safe(_pending(), prepared)

    assert boundary.retrievals == 0 and boundary.commits == 0


def test_paid_embedding_only_rejects_insufficient_before_vector_validation() -> None:
    boundary = _Boundary()
    prepared = _prepared()
    object.__setattr__(prepared, 'tentative_outcome', 'insufficient_evidence')
    object.__setattr__(
        prepared,
        'query_embedding_result',
        SimpleNamespace(prepared=SimpleNamespace()),
    )
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    with pytest.raises(RagFinalizationError, match='post-generation'):
        service.finalize_paid_embedding_only_safe(_pending(), prepared)

    assert boundary.retrievals == 0 and boundary.commits == 0


def test_post_generation_insufficient_requires_actual_answer_cost_and_discards_reason() -> None:
    prepared = _post_generation_insufficient()
    snapshot, binding = _safety_requirement('answer_generation')
    snapshot = replace(
        snapshot,
        authorized_model_config_snapshot_hmac=(
            prepared.answer_model_config_snapshot_hmac
        ),
    )
    binding = replace(binding, policy_snapshot=snapshot)
    query = _cost_child('query_embedding', 0)
    answer = _cost_child('answer_generation', 1)
    answer.attempted = True
    answer.dispatch_count = 1
    answer.actual_input_tokens = 21
    answer.actual_output_tokens = 4
    answer.charged_cost_usd = Decimal('0.000123')
    answer.charge_basis = 'actual'
    answer.dispatch_fence_hmac = 'd' * 64
    answer.process_instance_hmac = 'e' * 64
    answer.terminal_outcome = 'component_succeeded'
    answer.provider = snapshot.provider
    answer.model = snapshot.model
    answer.authorized_model_config_version = (
        snapshot.authorized_model_config_version
    )
    answer.authorized_model_config_snapshot_hmac = (
        snapshot.authorized_model_config_snapshot_hmac
    )
    answer.authorized_cost_policy_version = (
        snapshot.authorized_cost_policy_version
    )
    answer.authorized_token_estimator_version = (
        snapshot.authorized_token_estimator_version
    )
    answer.authorized_policy_snapshot_hmac = (
        snapshot.authorized_policy_snapshot_hmac
    )
    boundary, pending = _pending_boundary([query, answer])
    boundary._phase2_authority = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=_ClosableConnection,
        safety_requirements=((snapshot, binding),),
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=_readiness,
    )
    boundary.validate_pending(pending)
    boundary.validate_branch_costs(prepared, branch='paid_prepared')
    boundary._settings = Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='secret',
    )
    boundary._pending = pending
    boundary._projection_coordinator = SimpleNamespace(
        lock_canonical_tail=lambda document_ids: object(),
        validate_tail_context=lambda context: None,
    )
    projection = boundary.project(
        prepared,
        prepared.retrieval_result,
        drifted=False,
    )

    assert projection.outcome == 'insufficient_evidence'
    assert projection.model_influence == ()
    assert projection.answer_text == '질문에 답할 수 있는 근거를 찾지 못했습니다.'
    assert 'raw provider reason' not in projection.answer_text
    assert answer.actual_input_tokens == 21
    assert answer.actual_output_tokens == 4
    assert answer.charged_cost_usd == Decimal('0.000123')


def test_paid_prepared_rejects_unattempted_terminal_zero_answer() -> None:
    prepared = _post_generation_insufficient()
    children = [
        _cost_child('query_embedding', 0),
        _cost_child('answer_generation', 1),
    ]
    boundary, pending = _pending_boundary(children)
    snapshot, binding = _safety_requirement('answer_generation')
    boundary._phase2_authority = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=_ClosableConnection,
        safety_requirements=((snapshot, binding),),
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=_readiness,
    )
    boundary.validate_pending(pending)

    with pytest.raises(RagFinalizationError, match='attempted components'):
        boundary.validate_branch_costs(prepared, branch='paid_prepared')


def test_prepared_influence_requires_exact_authenticated_aggregate() -> None:
    observation = PreparedModelInfluenceObservation(
        ordinal=0,
        slot_id='E1',
        support_mode='trusted_fact',
        lookup_identity=SimpleNamespace(),
        effective_permission='internal',
        serving_identity_hmac='5' * 64,
        serving_version_fingerprint='6' * 64,
        model_content_hmac='7' * 64,
        canonical_citation_projection_hmac='8' * 64,
        approval_provenance_hmac=None,
        evidence_link_set_hmac=None,
        observation_hmac='9' * 64,
    )
    with pytest.raises(ValueError, match='influence authority'):
        replace(
            _prepared(),
            evidence_slots=(SimpleNamespace(slot_id='E1'),),
            model_influence_observations=(observation,),
            rendered_input_hmac='a' * 64,
            answer_model_config_snapshot_hmac='b' * 64,
        )


@pytest.mark.parametrize(
    ('metadata_key', 'replacement'),
    (
        ('rendered_input_hmac', 'c' * 64),
        ('answer_model_config_snapshot_hmac', 'd' * 64),
        ('prepared_model_influence_observation_hmac', 'e' * 64),
    ),
)
def test_phase1_committed_model_influence_authority_rejects_substitution(
    metadata_key: str,
    replacement: str,
) -> None:
    observation = PreparedModelInfluenceObservation(
        ordinal=0,
        slot_id='E1',
        support_mode='trusted_fact',
        lookup_identity=SimpleNamespace(),
        effective_permission='internal',
        serving_identity_hmac='5' * 64,
        serving_version_fingerprint='6' * 64,
        model_content_hmac='7' * 64,
        canonical_citation_projection_hmac='8' * 64,
        approval_provenance_hmac=None,
        evidence_link_set_hmac=None,
        observation_hmac='9' * 64,
    )
    full = PreparedModelInfluenceSet(
        observations=(observation,),
        prepared_corpus_generation=5,
        prepared_index_generation=None,
        prepared_readiness_hmac=None,
        rendered_input_hmac='a' * 64,
        aggregate_observation_hmac='b' * 64,
    )
    prepared = replace(
        _prepared(),
        evidence_slots=(SimpleNamespace(slot_id='E1'),),
        model_influence_observations=(observation,),
        prepared_model_influence=full,
        rendered_input_hmac='a' * 64,
        answer_model_config_snapshot_hmac='f' * 64,
    )
    pending = _pending()
    parent = SimpleNamespace(
        id=pending.parent_agent_run_id,
        metadata_={
            'current_text_hmac': prepared.prepared_text.current_text_hmac,
            'retrieval_query_hmac': prepared.prepared_text.retrieval_query_hmac,
            'configured_backend': prepared.retrieval_result.configured_backend,
            'surface': 'ask',
            'rendered_input_hmac': prepared.rendered_input_hmac,
            'answer_model_config_snapshot_hmac': (
                prepared.answer_model_config_snapshot_hmac
            ),
            'prepared_model_influence_observation_hmac': (
                full.aggregate_observation_hmac
            ),
        },
    )
    boundary = object.__new__(SqlAlchemyRagFinalizationBoundary)
    boundary._pending_parent = parent
    boundary.validate_prepared(pending, prepared)
    parent.metadata_[metadata_key] = replacement

    with pytest.raises(RagFinalizationError, match='carrier changed'):
        boundary.validate_prepared(pending, prepared)


def test_provider_free_phase2_refuses_live_owner_without_entering_barrier(
    monkeypatch,
) -> None:
    connection = _ClosableConnection()
    barrier = _assemble_rag_evidence_barrier(
        load_current_identity=lambda: 'a' * 64
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_finalization.try_acquire_advisory_lock',
        lambda *args, **kwargs: False,
    )
    authority = _assemble_provider_free_rag_phase2_authority(
        owner_connection_factory=lambda: connection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=barrier,
    )

    with pytest.raises(RagFinalizationError, match='still live'), authority.acquire(
        _pending(), _prepared(), branch='provider_free'
    ):
        raise AssertionError('unreachable')

    assert connection.closed is True


def test_provider_free_phase2_refuses_projection_fence_substitution(
    monkeypatch,
) -> None:
    connection = _ClosableConnection()
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_finalization.try_acquire_advisory_lock',
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_finalization.release_advisory_lock',
        lambda *args, **kwargs: None,
    )
    authority = _assemble_provider_free_rag_phase2_authority(
        owner_connection_factory=lambda: connection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: 'f' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
    )

    with (
        pytest.raises(RagFinalizationError, match='fence changed'),
        authority.acquire(_pending(), _prepared(), branch='provider_free'),
    ):
        raise AssertionError('unreachable')


def test_paid_phase2_refuses_fresh_readiness_drift_after_safety_and_owner(
    monkeypatch,
) -> None:
    safety_connection = _ClosableConnection()
    owner_connection = _ClosableConnection()
    safety = object.__new__(RagProviderSafetyService)

    @contextmanager
    def admitted_safety(*args, **kwargs):
        del args, kwargs
        yield

    monkeypatch.setattr(
        RagProviderSafetyService, 'finalization_barrier', admitted_safety
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_finalization.try_acquire_advisory_lock',
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_finalization.release_advisory_lock',
        lambda *args, **kwargs: None,
    )
    authority = _assemble_paid_rag_phase2_authority(
        provider_safety=safety,
        safety_connection_factory=lambda: safety_connection,
        safety_requirements=(_safety_requirement('query_embedding'),),
        owner_connection_factory=lambda: owner_connection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=lambda: _readiness(snapshot='5' * 64),
    )
    embedding = SimpleNamespace(
        prepared=SimpleNamespace(
            corpus_generation=1,
            vector_index_generation=1,
            readiness_snapshot_hmac='4' * 64,
        )
    )
    prepared = replace(_prepared(), query_embedding_result=embedding)

    with (
        pytest.raises(RagFinalizationError, match='readiness changed'),
        authority.acquire(
            _pending(), prepared, branch='paid_embedding_only'
        ),
    ):
        raise AssertionError('unreachable')

    assert safety_connection.closed is True
    assert owner_connection.closed is True


def test_paid_phase2_rejects_wrong_component_safety_requirement_before_locking() -> None:
    authority = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=_ClosableConnection,
        safety_requirements=(_safety_requirement('answer_generation'),),
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=_readiness,
    )
    prepared = replace(
        _prepared(),
        query_embedding_result=SimpleNamespace(prepared=SimpleNamespace()),
    )

    with (
        pytest.raises(RagFinalizationError, match='safety requirements'),
        authority.acquire(_pending(), prepared, branch='paid_embedding_only'),
    ):
        raise AssertionError('unreachable')


def test_paid_prepared_phase2_rejects_incomplete_safety_requirement_set() -> None:
    prepared = replace(
        _post_generation_insufficient(),
        query_embedding_result=SimpleNamespace(prepared=SimpleNamespace()),
    )
    authority = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=_ClosableConnection,
        safety_requirements=(_safety_requirement('answer_generation'),),
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=_readiness,
    )

    with (
        pytest.raises(RagFinalizationError, match='safety requirements'),
        authority.acquire(_pending(), prepared, branch='paid_prepared'),
    ):
        raise AssertionError('unreachable')


def test_postgresql_boundary_rejects_mutex_only_evidence_barrier() -> None:
    phase2 = _assemble_provider_free_rag_phase2_authority(
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
    )
    db = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(
            dialect=SimpleNamespace(name='postgresql')
        )
    )

    with pytest.raises(TypeError, match='PostgreSQL evidence barrier'):
        SqlAlchemyRagFinalizationBoundary(
            db=db,
            settings=Settings(
                _env_file=None,
                agent_runtime_fingerprint_secret='secret',
            ),
            retriever=SimpleNamespace(invoke=lambda request: request),
            phase2_authority=phase2,
        )


def _recovery_bundle(
    session: object,
    *,
    provider_free_barrier,
    paid_barrier,
    owner_connection_factory=_ClosableConnection,
    safety_connection_factory=_ClosableConnection,
):
    ledger = object.__new__(RagCostLedger)
    ledger._session = session
    projection_read = object.__new__(ServingProjectionReadCoordinator)
    projection_read._db = session
    provider_free = _assemble_provider_free_rag_phase2_authority(
        owner_connection_factory=owner_connection_factory,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=provider_free_barrier,
    )
    paid = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=safety_connection_factory,
        safety_requirements=(),
        owner_connection_factory=owner_connection_factory,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=paid_barrier,
        load_current_readiness=_readiness,
    )
    return (
        _assemble_projection_owner_recovery_authority(
            ledger=ledger,
            provider_free=provider_free,
            paid=paid,
            projection_read=projection_read,
        ),
        provider_free,
        paid,
    )


def _postgres_barrier(*, engine: object):
    return _assemble_rag_evidence_barrier(
        load_current_identity=lambda: 'a' * 64,
        connection_factory=lambda: _ClosableConnection(engine),
        registered_lock=_evidence_capability(),
    )


def _fake_postgres_session(*, engine: object):
    return SimpleNamespace(
        get_bind=lambda: SimpleNamespace(
            dialect=SimpleNamespace(name='postgresql'),
            engine=engine,
        )
    )


def _construct_boundary(*, db, phase2, recovery):
    return SqlAlchemyRagFinalizationBoundary(
        db=db,
        settings=Settings(
            _env_file=None,
            agent_runtime_fingerprint_secret='secret',
        ),
        retriever=SimpleNamespace(invoke=lambda request: request),
        phase2_authority=phase2,
        recovery_authority=recovery,
    )


def test_postgresql_boundary_rejects_cross_session_recovery_authority() -> None:
    engine = object()
    boundary_db = _fake_postgres_session(engine=engine)
    recovery_db = _fake_postgres_session(engine=engine)
    barrier = _postgres_barrier(engine=engine)
    recovery, phase2, _ = _recovery_bundle(
        recovery_db,
        provider_free_barrier=barrier,
        paid_barrier=barrier,
    )

    with pytest.raises(TypeError, match='recovery database authority'):
        _construct_boundary(db=boundary_db, phase2=phase2, recovery=recovery)


@pytest.mark.parametrize('mutex_side', ('provider_free', 'paid'))
def test_postgresql_boundary_rejects_mutex_recovery_evidence_barrier(
    mutex_side: str,
) -> None:
    engine = object()
    db = _fake_postgres_session(engine=engine)
    postgres = _postgres_barrier(engine=engine)
    mutex = _assemble_rag_evidence_barrier(
        load_current_identity=lambda: 'a' * 64
    )
    recovery, provider_free, paid = _recovery_bundle(
        db,
        provider_free_barrier=(mutex if mutex_side == 'provider_free' else postgres),
        paid_barrier=(mutex if mutex_side == 'paid' else postgres),
    )

    ordinary = paid if mutex_side == 'provider_free' else provider_free
    with pytest.raises(TypeError, match='recovery database authority'):
        _construct_boundary(db=db, phase2=ordinary, recovery=recovery)


def test_postgresql_boundary_rejects_cross_database_recovery_connections() -> None:
    boundary_engine = object()
    other_engine = object()
    db = _fake_postgres_session(engine=boundary_engine)
    barrier = _postgres_barrier(engine=other_engine)
    recovery, phase2, _ = _recovery_bundle(
        db,
        provider_free_barrier=barrier,
        paid_barrier=barrier,
        owner_connection_factory=lambda: _ClosableConnection(other_engine),
        safety_connection_factory=lambda: _ClosableConnection(other_engine),
    )

    with pytest.raises(TypeError, match='recovery database authority'):
        _construct_boundary(db=db, phase2=phase2, recovery=recovery)


def test_evidence_barrier_rejects_wrong_registered_postgresql_capability() -> None:
    with pytest.raises(ValueError, match='evidence advisory'):
        _assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64,
            connection_factory=_ClosableConnection,
            registered_lock=_owner_capability(3),
        )


def test_paid_embedding_only_rejects_unvalidated_vector_carrier() -> None:
    boundary = _Boundary()
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(
            _env_file=None,
            agent_runtime_fingerprint_secret='secret',
        ),
    )
    prepared = replace(
        _prepared(),
        query_embedding_result=SimpleNamespace(
            prepared=SimpleNamespace(
                corpus_generation=1,
                vector_index_generation=1,
                readiness_snapshot_hmac='4' * 64,
            )
        ),
    )

    with pytest.raises(RagFinalizationError, match='embedding carrier'):
        service.finalize_paid_embedding_only_safe(_pending(), prepared)

    assert boundary.retrievals == 0
    assert boundary.commits == 0


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


def test_provider_free_safe_rejects_corrupted_failure_outcome_before_transaction() -> None:
    boundary = _Boundary()
    prepared = _prepared()
    object.__setattr__(prepared, 'tentative_outcome', 'retriever_unavailable')
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    with pytest.raises(RagFinalizationError, match='safe product outcome'):
        service.finalize_provider_free_safe(_pending(), prepared)

    assert boundary.retrievals == 0 and boundary.commits == 0


def test_paid_embedding_safe_rejects_corrupted_failure_outcome_before_validation() -> None:
    boundary = _Boundary()
    prepared = replace(
        _prepared(),
        query_embedding_result=SimpleNamespace(prepared=SimpleNamespace()),
    )
    object.__setattr__(prepared, 'tentative_outcome', 'budget_exceeded')
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    with pytest.raises(RagFinalizationError, match='safe product outcome'):
        service.finalize_paid_embedding_only_safe(_pending(), prepared)

    assert boundary.retrievals == 0 and boundary.commits == 0


def test_evidence_changed_safe_rejects_failure_outcome_before_transaction() -> None:
    boundary = _Boundary()
    prepared = replace(
        _prepared(),
        tentative_outcome='evidence_unavailable',
        canned_message_identity='rag-canned-evidence-unavailable:v1',
    )
    object.__setattr__(prepared, 'tentative_outcome', 'model_provider_failed')
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    with pytest.raises(
        RagFinalizationError,
        match='safe product outcome|evidence-changed',
    ):
        service.finalize_pre_generation_evidence_changed(_pending(), prepared)

    assert boundary.retrievals == 0 and boundary.commits == 0


def test_phase2_and_canonical_locks_precede_agent_run_cost_tail() -> None:
    boundary = _OrderedBoundary()
    service = RagFinalizationService(
        transaction_boundary=boundary,
        settings=Settings(_env_file=None, agent_runtime_fingerprint_secret='secret'),
    )

    service.finalize_provider_free_safe(_pending(), _prepared())

    assert boundary.events == [
        'phase2:provider_free',
        'prefix',
        'generations',
        'retrieve',
        'canonical',
        'run_cost_tail',
        'prepared_binding',
        'parent_final',
        'commit',
        'phase2:release',
    ]


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


def test_pending_success_rejects_abandoned_unknown_child() -> None:
    children = [
        _cost_child('query_embedding', 0),
        _cost_child('answer_generation', 1),
    ]
    children[0].dispatch_state = 'abandoned_unknown'
    children[0].terminal_outcome = 'abandoned_unknown'
    boundary, pending = _pending_boundary(children)

    with pytest.raises(RagFinalizationError, match='pending RAG projection changed'):
        boundary.validate_pending(pending)


def test_parent_finalizer_refuses_terminal_failure_projection() -> None:
    children = [
        _cost_child('query_embedding', 0),
        _cost_child('answer_generation', 1),
    ]
    boundary, pending = _pending_boundary(children)
    boundary.validate_pending(pending)
    projection = CanonicalRagProjection(
        outcome='budget_exceeded',
        answer_text='실패',
        evidence=V1EvidenceProjection((), (), (), (), (), '5' * 64),
        model_influence=(),
        hidden_match_count=0,
        effective_backend='deterministic_lexical',
        result_hmac='6' * 64,
    )

    with pytest.raises(RagFinalizationError, match='terminal failure'):
        boundary.finalize_parent(pending, projection)


def test_provider_free_branch_requires_both_exact_terminal_zero_children() -> None:
    children = [
        _cost_child('query_embedding', 0),
        _cost_child('answer_generation', 1),
    ]
    children[0].reserved_input_tokens = 1
    children[0].reserved_cost_usd = Decimal('0.000001')
    boundary, pending = _pending_boundary(children)
    boundary.validate_pending(pending)

    with pytest.raises(RagFinalizationError, match='provider-free cost shape'):
        boundary.validate_branch_costs(_prepared(), branch='provider_free')


def test_paid_embedding_only_requires_exact_attempt_vector_receipt_and_fence() -> None:
    result = _embedding_result(query='  민감한 질의  ')
    snapshot, binding = _safety_requirement('query_embedding')
    snapshot = replace(
        snapshot,
        authorized_model_config_snapshot_hmac=(
            result.prepared.model_config_snapshot_hmac
        ),
        authorized_policy_snapshot_hmac=(
            result.prepared.provider_policy_snapshot_hmac
        ),
    )
    binding = replace(binding, policy_snapshot=snapshot)
    query = _cost_child('query_embedding', 0)
    query.attempted = True
    query.dispatch_count = 1
    query.actual_input_tokens = result.validated_input_tokens
    query.actual_output_tokens = 0
    query.charged_cost_usd = result.actual_cost_usd
    query.charge_basis = 'actual'
    query.dispatch_fence_hmac = result.prepared.attempt_fence_hmac
    query.process_instance_hmac = 'c' * 64
    query.terminal_outcome = 'component_succeeded'
    query.provider = snapshot.provider
    query.model = snapshot.model
    query.authorized_model_config_version = snapshot.authorized_model_config_version
    query.authorized_model_config_snapshot_hmac = (
        snapshot.authorized_model_config_snapshot_hmac
    )
    query.authorized_cost_policy_version = snapshot.authorized_cost_policy_version
    query.authorized_token_estimator_version = (
        snapshot.authorized_token_estimator_version
    )
    query.authorized_policy_snapshot_hmac = (
        snapshot.authorized_policy_snapshot_hmac
    )
    children = [query, _cost_child('answer_generation', 1)]
    boundary, pending = _pending_boundary(children)
    boundary._phase2_authority = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=_ClosableConnection,
        safety_requirements=((snapshot, binding),),
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=_readiness,
    )
    boundary.validate_pending(pending)
    prepared = replace(_prepared(), query_embedding_result=result)
    boundary.validate_branch_costs(prepared, branch='paid_embedding_only')
    query.dispatch_fence_hmac = 'd' * 64

    with pytest.raises(RagFinalizationError, match='paid embedding cost shape'):
        boundary.validate_branch_costs(prepared, branch='paid_embedding_only')


def test_paid_phase2_safety_hmacs_must_match_locked_cost_rows() -> None:
    result = _embedding_result(query='  민감한 질의  ')
    snapshot, binding = _safety_requirement('query_embedding')
    snapshot = replace(
        snapshot,
        authorized_model_config_snapshot_hmac=(
            result.prepared.model_config_snapshot_hmac
        ),
        authorized_policy_snapshot_hmac=(
            result.prepared.provider_policy_snapshot_hmac
        ),
    )
    binding = replace(binding, policy_snapshot=snapshot)
    query = _cost_child('query_embedding', 0)
    query.attempted = True
    query.dispatch_count = 1
    query.actual_input_tokens = result.validated_input_tokens
    query.actual_output_tokens = 0
    query.charged_cost_usd = result.actual_cost_usd
    query.charge_basis = 'actual'
    query.dispatch_fence_hmac = result.prepared.attempt_fence_hmac
    query.process_instance_hmac = 'c' * 64
    query.terminal_outcome = 'component_succeeded'
    query.provider = snapshot.provider
    query.model = snapshot.model
    query.authorized_model_config_version = snapshot.authorized_model_config_version
    query.authorized_model_config_snapshot_hmac = (
        snapshot.authorized_model_config_snapshot_hmac
    )
    query.authorized_cost_policy_version = snapshot.authorized_cost_policy_version
    query.authorized_token_estimator_version = (
        snapshot.authorized_token_estimator_version
    )
    query.authorized_policy_snapshot_hmac = snapshot.authorized_policy_snapshot_hmac
    children = [query, _cost_child('answer_generation', 1)]
    boundary, pending = _pending_boundary(children)
    boundary._phase2_authority = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=_ClosableConnection,
        safety_requirements=((snapshot, binding),),
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=_readiness,
    )
    boundary.validate_pending(pending)
    prepared = replace(_prepared(), query_embedding_result=result)
    boundary.validate_branch_costs(prepared, branch='paid_embedding_only')
    query.authorized_policy_snapshot_hmac = 'd' * 64

    with pytest.raises(RagFinalizationError, match='committed cost'):
        boundary.validate_branch_costs(prepared, branch='paid_embedding_only')


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
            replace(
                _prepared(),
                product_kind='search',
                tentative_outcome='search_projected',
                canned_message_identity=None,
            ),
            AssistantProjectionTarget(7, 8, 'owner-string'),
        )

    assert boundary.retrievals == 0
    assert boundary.commits == 0


def test_projection_recovery_uses_provider_free_owner_fence_and_cost_cas(
    monkeypatch,
) -> None:
    seen: list[tuple[object, ...]] = []
    snapshot = PendingProjectionRecoverySnapshot(
        agent_run_id=17,
        projection_owner_fence_hmac='2' * 64,
        runtime_cost_snapshot_hmac='4' * 64,
        paid_work_performed=False,
    )
    ledger = object.__new__(RagCostLedger)
    ledger._session = SimpleNamespace()
    projection_read = object.__new__(ServingProjectionReadCoordinator)
    projection_read._db = ledger._session

    @contextmanager
    def acquired_prefix(_self):
        yield (5, 0)

    monkeypatch.setattr(
        ServingProjectionReadCoordinator,
        'acquire',
        acquired_prefix,
    )
    monkeypatch.setattr(
        ServingProjectionReadCoordinator,
        'lock_canonical_tail',
        lambda self, document_ids: seen.append(
            ('c5_tail', tuple(document_ids))
        ) or object(),
    )
    monkeypatch.setattr(
        ServingProjectionReadCoordinator,
        'validate_tail_context',
        lambda self, context: seen.append(('c5_tail_validated', context)),
    )
    monkeypatch.setattr(
        RagCostLedger,
        'pending_projection_recovery_snapshot',
        lambda self, run_id: snapshot,
    )
    monkeypatch.setattr(
        RagCostLedger,
        'recover_incomplete_run',
        lambda self, **kwargs: seen.append(tuple(sorted(kwargs.items()))) or None,
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_finalization.try_acquire_advisory_lock',
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_finalization.release_advisory_lock',
        lambda *args, **kwargs: None,
    )
    provider_free = _assemble_provider_free_rag_phase2_authority(
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
    )
    paid = _assemble_paid_rag_phase2_authority(
        provider_safety=object.__new__(RagProviderSafetyService),
        safety_connection_factory=_ClosableConnection,
        safety_requirements=(),
        owner_connection_factory=_ClosableConnection,
        owner_capability_factory=_owner_capability,
        load_current_owner_fence=lambda _run_id: '2' * 64,
        evidence_barrier=_assemble_rag_evidence_barrier(
            load_current_identity=lambda: 'a' * 64
        ),
        load_current_readiness=_readiness,
    )
    authority = _assemble_projection_owner_recovery_authority(
        ledger=ledger,
        provider_free=provider_free,
        paid=paid,
        projection_read=projection_read,
    )
    boundary = object.__new__(SqlAlchemyRagFinalizationBoundary)
    boundary._recovery_authority = authority
    assert boundary.recover_dead_projection_owner(17) is None
    assert seen[0][0] == 'c5_tail'
    assert seen[0][1] == ()
    assert seen[1][0] == 'c5_tail_validated'
    assert seen[2:] == [(
            ('expected_runtime_cost_snapshot_hmac', '4' * 64),
            ('projection_owner_fence_hmac', '2' * 64),
            ('run_id', 17),
        )]
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
