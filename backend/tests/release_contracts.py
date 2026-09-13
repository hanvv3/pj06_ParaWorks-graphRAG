from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

ReleaseProfileName = Literal['settings-diagnostic', 'postgres', 'compatibility', 'non-slack', 'full']
ReleaseOutcomeCode = Literal['preflight_refused', 'environment_refused', 'collection_mismatch', 'evidence_refused', 'verification_failed', 'cleanup_failed', 'baseline_mismatch', 'passed']
ChildKind = Literal['canonical_collection', 'verification']
_ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,62}$', re.ASCII)


@dataclass(frozen=True, slots=True)
class ReleaseChild:
    child_id: str
    kind: ChildKind
    pytest_argv: tuple[str, ...]
    timeout_seconds: int
    allowed_pytest_exit_codes: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.child_id) or self.timeout_seconds < 1:
            raise ValueError('invalid release child')
        if not self.pytest_argv or any('\x00' in value for value in self.pytest_argv):
            raise ValueError('invalid release selector')


@dataclass(frozen=True, slots=True)
class ReleaseProfile:
    name: ReleaseProfileName
    children: tuple[ReleaseChild, ...]
    timeout_seconds: int
    expected_deselected_nodeids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReleaseInvocationIdentity:
    run_id: str
    profile: ReleaseProfileName
    child_id: str
    invocation_hash: str


@dataclass(frozen=True, slots=True)
class ReleaseEvent:
    nodeid: str
    phase: Literal['setup', 'call', 'teardown']
    outcome: Literal['passed', 'failed', 'skipped']
    wasxfail: bool
    collection_index: int
    failure_reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class LeaseLifecycleEvent:
    lease_id_sha256: str
    state: Literal['created', 'dropped']


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceSidecar:
    schema_version: int
    identity: ReleaseInvocationIdentity
    collection_nodeids: tuple[str, ...]
    collection_sha256: str
    events: tuple[ReleaseEvent, ...]
    lease_events: tuple[LeaseLifecycleEvent, ...]
    pytest_exitstatus: int
    events_sha256: str
    complete: Literal[True]


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    controller_schema_version: int
    sidecar_schema_version: int
    explicit_pytest_plugins: tuple[str, ...]
    deferred_slack_failure_nodeids: tuple[str, ...]
    postgres_release_critical_modules: tuple[str, ...]
    safe_parameter_case_counts: Mapping[str, int]
    profiles: Mapping[ReleaseProfileName, ReleaseProfile]


SETTINGS_NODES = (
    'backend/tests/test_agent_runtime_checkpointing.py::test_enabled_checkpoint_mode_rejects_unsupported_backend_without_secret',
    'backend/tests/test_agent_runtime_checkpointing.py::test_production_rejects_the_local_default_fingerprint_secret',
    'backend/tests/test_auto_review_contracts.py::test_non_disabled_mode_rejects_local_default_fingerprint_secret',
    'backend/tests/test_auto_review_contracts.py::test_disabled_sqlite_smoke_may_use_process_local_placeholder_without_durable_ready_state',
    'backend/tests/test_auto_review_contracts.py::test_process_local_sqlite_smoke_is_limited_to_disabled_in_memory_url[sqlite-memory-disabled]',
    'backend/tests/test_auto_review_contracts.py::test_file_backed_sqlite_placeholder_refuses_every_c5_bound_durable_write',
)
POSTGRES_MODULES = (
    'backend/tests/test_postgres_isolation.py',
    'backend/tests/test_agent_runtime_postgres_checkpoint.py',
    'backend/tests/test_auto_review_migration.py',
    'backend/tests/test_auto_review_provenance.py',
    'backend/tests/test_review_transition_postgres.py',
    'backend/tests/test_review_v21_extraction_postgres.py',
    'backend/tests/test_review_v2_postgres.py',
    'backend/tests/test_pgvector_integration.py',
    'backend/tests/test_auto_review_postgres.py',
    'backend/tests/test_rag_release_authority_postgres.py',
)
POSTGRES_IDS = (
    'postgres-isolation-contract', 'postgres-agent-runtime-checkpoint',
    'postgres-auto-review-migration', 'postgres-auto-review-provenance',
    'postgres-review-transition', 'postgres-review-v21-extraction',
    'postgres-review-v2', 'postgres-pgvector', 'postgres-auto-review-c5',
    'postgres-rag-release-authority',
)
SLACK_TEN = (
    'backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_runs_real_agent_services',
    'backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_skips_agents_that_exceed_cost_budget',
    'backend/tests/test_company_memory_orchestration_service.py::test_company_memory_orchestration_uses_cache_when_evidence_is_unchanged',
    'backend/tests/test_oauth_pkce.py::test_slack_oauth_pkce_generation',
    'backend/tests/test_oauth_pkce.py::test_slack_callback_with_custom_redirect_uri_and_pkce',
    'backend/tests/test_oauth_pkce.py::test_api_endpoints_support_redirect_uri',
    'backend/tests/test_orchestration_api.py::test_company_memory_orchestration_api_runs_agent_services',
    'backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_company_memory_emits_review_checkpoint_without_paid_calls',
    'backend/tests/test_quality_permission_regression_suite.py::test_quality_suite_cache_hit_does_not_duplicate_agent_runs_or_review_items',
    'backend/tests/test_slack_oauth.py::test_slack_sync_endpoint_uses_installed_connection_token_without_exposing_it',
)
COMPATIBILITY_GROUPS = (
    ('compatibility-contracts-provenance', (
        'backend/tests/test_auto_review_contracts.py', 'backend/tests/test_auto_review_cost_policy.py', 'backend/tests/test_auto_review_migration.py', 'backend/tests/test_assistant_models.py', 'backend/tests/test_keyed_mutation_guard.py', 'backend/tests/test_auto_review_key_bootstrap.py', 'backend/tests/test_provider_send_fence.py', 'backend/tests/test_review_v21_preflight.py', 'backend/tests/test_review_v21_drafting.py', 'backend/tests/test_review_v21_extraction.py', 'backend/tests/test_review_resolution_actors.py', 'backend/tests/test_auto_review_provenance.py',
    )),
    ('compatibility-serving-revoke', (
        'backend/tests/test_auto_review_revocation.py', 'backend/tests/test_auto_review_quality_revoke.py', 'backend/tests/test_auto_review_source_reconciliation.py', 'backend/tests/test_auto_review_source_reconciliation_admin.py', 'backend/tests/test_source_content_signature.py', 'backend/tests/test_review_evidence_visibility.py', 'backend/tests/test_google_connector.py', 'backend/tests/test_connector_ingestion_contract.py', 'backend/tests/test_document_ingestion_service.py', 'backend/tests/test_rag_indexing.py', 'backend/tests/test_pgvector_store.py', 'backend/tests/test_pgvector_integration.py', 'backend/tests/test_knowledge_api.py', 'backend/tests/test_dashboard_api.py', 'backend/tests/test_review.py', 'backend/tests/test_todos_api.py', 'backend/tests/test_notifications_api.py', 'backend/tests/test_mock_sync.py', 'backend/tests/test_integration_runtime_status.py', 'backend/tests/test_search_permissions.py', 'backend/tests/test_search_retrieval_backend.py', 'backend/tests/test_ask_api.py', 'backend/tests/test_assistant_api.py', 'backend/tests/test_assistant_service.py', 'backend/tests/test_assistant_email_agent.py', 'backend/tests/test_company_memory_orchestration_service.py', 'backend/tests/test_orchestration_api.py', 'backend/tests/test_project_memory_api.py', 'backend/tests/test_rag_orchestrator_agent.py', 'backend/tests/test_rag_orchestrator_service.py',
    )),
    ('compatibility-policy-validation', (
        'backend/tests/test_auto_review_eligibility.py', 'backend/tests/test_auto_review_policy.py', 'backend/tests/test_auto_review_validator.py', 'backend/tests/test_auto_review_model_router.py', 'backend/tests/test_auto_review_validation_store.py', 'backend/tests/test_auto_review_orchestrator.py', 'backend/tests/test_auto_review_rollout.py', 'backend/tests/test_auto_review_audit.py', 'backend/tests/test_auto_review_rollout_admin.py', 'backend/tests/test_auto_review_launch_confirmation.py', 'backend/tests/test_review_workflow_facade.py',
    )),
    ('compatibility-v21-release', (
        'backend/tests/test_review_v21_state.py', 'backend/tests/test_review_v21_graph.py', 'backend/tests/test_review_v21_service.py', 'backend/tests/test_review_v21_api.py', 'backend/tests/test_auto_review_call_recovery.py', 'backend/tests/test_auto_review_api.py', 'backend/tests/test_auto_review_golden.py', 'backend/tests/test_auto_review_evaluation_cli.py', 'backend/tests/test_auto_review_extraction_compatibility_cli.py', 'backend/tests/test_auto_review_postgres.py', 'backend/tests/test_auto_review_smoke.py',
    )),
    ('compatibility-review-v2', (
        'backend/tests/test_review_v2_schemas.py', 'backend/tests/test_review_v2_preflight.py', 'backend/tests/test_review_v2_drafting.py', 'backend/tests/test_review_transitions.py', 'backend/tests/test_review_v2_graph.py', 'backend/tests/test_review_v2_service.py', 'backend/tests/test_review_v2_api.py', 'backend/tests/test_review_v2_postgres.py', 'backend/tests/test_review_transition_postgres.py', 'backend/tests/test_review_knowledge_promotion.py', 'backend/tests/test_review_rbac.py',
    )),
    ('compatibility-agent-runtime', (
        'backend/tests/test_agent_preflight.py', 'backend/tests/test_agent_runtime_state.py', 'backend/tests/test_agent_runtime_fingerprints.py', 'backend/tests/test_agent_workflow_models.py', 'backend/tests/test_agent_runtime_migration.py', 'backend/tests/test_agent_runtime_checkpointing.py', 'backend/tests/test_agent_runtime_lifespan.py', 'backend/tests/test_agent_runtime_bootstrap.py', 'backend/tests/test_agent_runtime_retention.py', 'backend/tests/test_agent_runtime_graph_versions.py', 'backend/tests/test_agent_runtime_checkpoint_execution.py', 'backend/tests/test_agent_runtime_postgres_checkpoint.py', 'backend/tests/test_db_init.py', 'backend/tests/test_db_schema_operations.py', 'backend/tests/test_data_reset.py', 'backend/tests/test_langchain_langgraph_dependency_compat.py',
    )),
)


def _child(child_id: str, selectors: tuple[str, ...], *, collect: bool = False, exits: tuple[int, ...] = (0,)) -> ReleaseChild:
    argv = selectors + (('--collect-only',) if collect else ()) + ('-q',)
    return ReleaseChild(child_id, 'canonical_collection' if collect else 'verification', argv, 1200, exits)


def build_manifest() -> ReleaseManifest:
    settings = ReleaseProfile('settings-diagnostic', (_child('settings-collection', SETTINGS_NODES, collect=True), _child('settings-contracts', SETTINGS_NODES)), 1200)
    postgres_children = [_child('postgres-collection', POSTGRES_MODULES, collect=True)]
    for child_id, module in zip(POSTGRES_IDS, POSTGRES_MODULES, strict=True):
        argv = (module, '-vv', '--tb=long') if child_id == 'postgres-review-v2' else (module, '-q')
        postgres_children.append(ReleaseChild(child_id, 'verification', argv, 1800))
    postgres = ReleaseProfile('postgres', tuple(postgres_children), 3600)
    deselect = tuple(f'--deselect={node}' for node in SLACK_TEN)
    compat_selectors = tuple(dict.fromkeys(node for _, group in COMPATIBILITY_GROUPS for node in group)) + deselect
    compatibility_slack = tuple(
        node
        for node in SLACK_TEN
        if node.split('::', 1)[0]
        in {path for _, group in COMPATIBILITY_GROUPS for path in group}
    )
    compatibility = ReleaseProfile('compatibility', (_child('compatibility-collection', compat_selectors, collect=True), *(_child(name, group + deselect) for name, group in COMPATIBILITY_GROUPS)), 3600, compatibility_slack)
    non_slack_args = ('backend/tests',) + deselect
    non_slack = ReleaseProfile('non-slack', (_child('non-slack-collection', non_slack_args, collect=True), _child('non-slack-backend', non_slack_args)), 3600, SLACK_TEN)
    full = ReleaseProfile('full', (_child('full-collection', ('backend/tests',), collect=True), _child('full-backend', ('backend/tests',), exits=(0, 1))), 3600)
    return ReleaseManifest(1, 1, ('backend.tests.release_evidence_plugin',), SLACK_TEN, POSTGRES_MODULES, {}, {p.name: p for p in (settings, postgres, compatibility, non_slack, full)})


def invocation_hash(*, profile: str, child: ReleaseChild, commit_sha: str) -> str:
    payload = {'profile': profile, 'child_id': child.child_id, 'kind': child.kind, 'argv': child.pytest_argv, 'timeout': child.timeout_seconds, 'commit_sha': commit_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def sidecar_from_json(value: dict) -> ReleaseEvidenceSidecar:
    identity = ReleaseInvocationIdentity(**value['identity'])
    return ReleaseEvidenceSidecar(
        schema_version=value['schema_version'], identity=identity,
        collection_nodeids=tuple(value['collection_nodeids']),
        collection_sha256=value['collection_sha256'],
        events=tuple(ReleaseEvent(**event) for event in value['events']),
        lease_events=tuple(LeaseLifecycleEvent(**event) for event in value['lease_events']),
        pytest_exitstatus=value['pytest_exitstatus'], events_sha256=value['events_sha256'], complete=value['complete'],
    )


def to_json_dict(value: object) -> dict:
    return asdict(value)
