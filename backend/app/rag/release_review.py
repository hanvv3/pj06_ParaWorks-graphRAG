"""Task24's case-claim projection boundary; no preview or authorization issuer.

The manifest preimage is checked against the immutable, already reviewed
authorization row. Signing arbitrary submitted SQL values cannot authorize them.
Runtime ids/clock are allocated by this module, never taken from SQL literals.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID
from weakref import WeakKeyDictionary

from sqlalchemy import Connection, func, select

from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
    admission_source_window,
)
from backend.app.agent_runtime.rag_safety_identity import (
    admission_identity,
    rag_identity_hmac,
    require_lower_hmac,
    runtime_cost_identity,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent

_ORDER = ('query_embedding', 'answer_generation')
_ZERO = Decimal('0.000000')
_VERSION = 'rag-live-case-claim-manifest:v1'
_ASSEMBLY = 'rag-live-case-claim-assembly:v1'


@dataclass(frozen=True, slots=True)
class FrozenCaseClaimComponent:
    policy: AuthorizedProviderPolicySnapshot
    reserved_input_tokens: int
    reserved_output_tokens: int
    reserved_cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class FrozenCaseClaimCase:
    ordinal: int
    case_id_hmac: str
    surface: str
    configured_backend: str
    current_text_hmac: str
    retrieval_query_hmac: str
    security_scope_fingerprint: str
    components: tuple[FrozenCaseClaimComponent, FrozenCaseClaimComponent]


@dataclass(frozen=True, slots=True)
class FrozenCaseClaimManifest:
    """Approved execution preimage; full 30-case quality fixtures remain Task24.

    The outer frozen quality manifest must retain this complete preimage when
    its eventual review/preview producer is implemented. It is not a projection
    assembled from an AgentRun, SQL plan, or submitted row-mutation HMAC.
    """

    cases: tuple[FrozenCaseClaimCase, ...]
    contract_version: str = _VERSION
    fixture_manifest_version: str = 'rag-live-quality-30:v1'
    runtime_contract_version: str = 'rag-run:v2'
    assembly_version: str = _ASSEMBLY


def _refuse():
    from backend.app.rag.release_ledger import RagReleaseLedgerError

    raise RagReleaseLedgerError('approved case claim projection is unavailable')


def _manifest_payload(manifest):
    if (
        type(manifest) is not FrozenCaseClaimManifest
        or any(
            type(getattr(manifest, name)) is not str
            for name in (
                'contract_version',
                'fixture_manifest_version',
                'runtime_contract_version',
                'assembly_version',
            )
        )
        or manifest.contract_version != _VERSION
        or manifest.fixture_manifest_version != 'rag-live-quality-30:v1'
        or manifest.runtime_contract_version != 'rag-run:v2'
        or manifest.assembly_version != _ASSEMBLY
        or type(manifest.cases) is not tuple
        or not 1 <= len(manifest.cases) <= 30
    ):
        _refuse()
    seen = set()
    for ordinal, case in enumerate(manifest.cases):
        if (
            type(case) is not FrozenCaseClaimCase
            or any(
                type(getattr(case, name)) is not str
                for name in (
                    'case_id_hmac',
                    'surface',
                    'configured_backend',
                    'current_text_hmac',
                    'retrieval_query_hmac',
                    'security_scope_fingerprint',
                )
            )
            or type(case.ordinal) is not int
            or case.ordinal != ordinal
            or case.case_id_hmac in seen
            or case.surface not in {'ask', 'assistant'}
            or case.configured_backend not in {'keyword', 'pgvector'}
            or type(case.components) is not tuple
            or len(case.components) != 2
        ):
            _refuse()
        for value in (
            case.case_id_hmac,
            case.current_text_hmac,
            case.retrieval_query_hmac,
            case.security_scope_fingerprint,
        ):
            try:
                require_lower_hmac(value)
            except ValueError:
                _refuse()
        seen.add(case.case_id_hmac)
        for index, component in enumerate(case.components):
            if (
                type(component) is not FrozenCaseClaimComponent
                or type(component.policy) is not AuthorizedProviderPolicySnapshot
                or component.policy.component != _ORDER[index]
                or type(component.reserved_input_tokens) is not int
                or type(component.reserved_output_tokens) is not int
                or min(
                    component.reserved_input_tokens, component.reserved_output_tokens
                )
                < 0
                or type(component.reserved_cost_usd) is not Decimal
                or not component.reserved_cost_usd.is_finite()
                or not _ZERO <= component.reserved_cost_usd <= Decimal('0.012000')
                or component.reserved_cost_usd.as_tuple().exponent < -6
            ):
                _refuse()
            # Check original fields before asdict/deepcopy can invoke a
            # subclass hook and hide a noncanonical input behind a plain str.
            for field in fields(AuthorizedProviderPolicySnapshot):
                key = field.name
                value = getattr(component.policy, key)
                if type(value) is not str or not value:
                    _refuse()
                if key.endswith('_hmac') or key == 'fingerprint_key_material_verifier':
                    try:
                        require_lower_hmac(value)
                    except ValueError:
                        _refuse()
        if (
            sum(item.reserved_cost_usd for item in case.components)
            > Decimal('0.012000')
            or case.components[0].reserved_output_tokens != 0
            or (
                case.configured_backend == 'keyword'
                and (
                    case.components[0].reserved_input_tokens != 0
                    or case.components[0].reserved_cost_usd != _ZERO
                )
            )
        ):
            _refuse()
    payload = asdict(manifest)
    payload['cases'] = list(payload['cases'])
    for case in payload['cases']:
        case['components'] = list(case['components'])
        for component in case['components']:
            component['reserved_cost_usd'] = format(
                component['reserved_cost_usd'], '.6f'
            )
    return payload


def case_claim_manifest_hmac(
    manifest: FrozenCaseClaimManifest, *, identity_secret: bytes
) -> str:
    return rag_identity_hmac(
        _manifest_payload(manifest),
        secret=identity_secret,
        schema_version=_VERSION,
        policy_version='rag-live-gate:v1',
    )


_BINDING_TYPES = {
    'ledger_uuid': str,
    'ledger_epoch': int,
    'approval_id_hmac': str,
    'approval_hmac': str,
    'approved_corpus_snapshot_hmac': str,
    'approved_provider_safety_snapshot_hmac': str,
    'provider_safety_envelope_digest': str,
    'validation_database_identity_hmac': str,
    'from_generation': int,
    'execution_process_instance_hmac': str,
    'execution_runner_fence_hmac': str,
    'case_id_hmac': str,
}


def _claim_binding(payload):
    if (
        type(payload) is not dict
        or any(type(key) is not str for key in payload)
        or any(
            type(payload.get(key)) is not kind for key, kind in _BINDING_TYPES.items()
        )
    ):
        _refuse()
    binding = {key: payload[key] for key in _BINDING_TYPES}
    try:
        if str(UUID(binding['ledger_uuid'])) != binding['ledger_uuid']:
            _refuse()
        for key, value in binding.items():
            if key not in {'ledger_uuid', 'ledger_epoch', 'from_generation'}:
                require_lower_hmac(value)
    except ValueError:
        _refuse()
    if binding['ledger_epoch'] < 1 or binding['from_generation'] < 0:
        _refuse()
    return binding


def _assert_exact_image_types(actual, expected):
    """Literal SQL keys/scalars and nested JSON must keep their exact types."""
    if type(actual) is not type(expected):
        _refuse()
    if type(expected) is dict:
        if any(type(key) is not str for key in actual) or set(actual) != set(expected):
            _refuse()
        for key, value in expected.items():
            _assert_exact_image_types(actual[key], value)
    elif type(expected) is list:
        if len(actual) != len(expected):
            _refuse()
        for item, value in zip(actual, expected, strict=True):
            _assert_exact_image_types(item, value)


def _runtime_images(case, *, run_id, child_ids, now, secret):
    """Literal rag-run:v2 admission defaults from RagCostLedger.create_admission.

    Permission starts restricted until final evidence projection; generation
    identity, workflow ownership, usage and completion fields are unset at
    admission. Route-inapplicable embedding is the approved terminal-zero case.
    """
    children = []
    for ordinal, value in enumerate(case.components):
        policy = value.policy
        children.append(
            {
                'id': child_ids[ordinal],
                'agent_run_id': run_id,
                'component': _ORDER[ordinal],
                'component_ordinal': ordinal,
                'dispatch_state': 'terminal'
                if ordinal == 0 and case.configured_backend == 'keyword'
                else 'not_attempted',
                'dispatch_fence_hmac': None,
                'process_instance_hmac': None,
                'attempted': False,
                'dispatch_count': 0,
                'reserved_input_tokens': value.reserved_input_tokens,
                'reserved_output_tokens': value.reserved_output_tokens,
                'actual_input_tokens': None,
                'actual_output_tokens': None,
                'reserved_cost_usd': value.reserved_cost_usd,
                'charged_cost_usd': _ZERO,
                'charge_basis': 'zero',
                'overrun': False,
                'provider': policy.provider,
                'model': policy.model,
                'authorized_model_config_version': policy.authorized_model_config_version,
                'authorized_model_config_snapshot_hmac': policy.authorized_model_config_snapshot_hmac,
                'authorized_cost_policy_version': policy.authorized_cost_policy_version,
                'authorized_token_estimator_version': policy.authorized_token_estimator_version,
                'authorized_policy_snapshot_hmac': policy.authorized_policy_snapshot_hmac,
                'terminal_outcome': None,
                'created_at': now,
                'updated_at': now,
            }
        )
    context = (
        'assistant-context:v1' if case.surface == 'assistant' else 'direct-query:v1'
    )
    total = sum((row['reserved_cost_usd'] for row in children), _ZERO)
    admission = admission_identity(
        {
            'answer_provider_policy_snapshot_hmac': case.components[
                1
            ].policy.authorized_policy_snapshot_hmac,
            'configured_backend': case.configured_backend,
            'current_text_hmac': case.current_text_hmac,
            'cutover_stage': case.surface,
            'graph_version': 'company-memory-rag-answer-v2.0',
            'mode': 'enforce',
            'query_context_version_bytes': exact_utf8_bytes(context),
            'query_embedding_provider_policy_snapshot_hmac': case.components[
                0
            ].policy.authorized_policy_snapshot_hmac
            if case.configured_backend == 'pgvector'
            else None,
            'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
            'retrieval_query_hmac': case.retrieval_query_hmac,
            'security_scope_fingerprint': case.security_scope_fingerprint,
            'surface': case.surface,
        },
        secret=secret,
    )
    runtime_components = []
    for row in children:
        projected = {
            key: row[key]
            for key in (
                'actual_input_tokens',
                'actual_output_tokens',
                'attempted',
                'authorized_model_config_snapshot_hmac',
                'authorized_policy_snapshot_hmac',
                'charge_basis',
                'component',
                'dispatch_count',
                'dispatch_fence_hmac',
                'dispatch_state',
                'overrun',
                'process_instance_hmac',
                'reserved_input_tokens',
                'reserved_output_tokens',
            )
        }
        for key in (
            'provider',
            'model',
            'authorized_model_config_version',
            'authorized_cost_policy_version',
            'authorized_token_estimator_version',
        ):
            projected[key + '_bytes'] = exact_utf8_bytes(row[key])
        for key in ('reserved_cost_usd', 'charged_cost_usd'):
            projected[key] = format(row[key], '.6f')
        runtime_components.append(projected)
    runtime_hmac = runtime_cost_identity(
        {
            'agent_run_id': run_id,
            'components': runtime_components,
            'parent_outcome': None,
            'parent_run_record_phase': 'admission',
            'parent_status': 'running',
            'run_contract_version': 'rag-run:v2',
            'snapshot_stage': 'pre_projection',
            'total_charged_cost_usd': '0.000000',
            'total_reserved_cost_usd': format(total, '.6f'),
        },
        secret=secret,
    )
    parent = {
        'id': run_id,
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v2',
        'status': 'running',
        'source_window': admission_source_window(
            mode='enforce', surface=case.surface, backend=case.configured_backend
        ),
        'cache_key': 'rag-v2-admission:' + admission,
        'model_name': 'rag-v2-admission',
        'generation_provider': None,
        'generation_reasoning_effort': None,
        'generation_route_version': None,
        'generation_output_contract_version': None,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'estimated_cost_usd': float(total),
        'permission_level': 'restricted',
        'metadata': {
            'configured_backend': case.configured_backend,
            'current_text_hmac': case.current_text_hmac,
            'cutover_stage': case.surface,
            'mode': 'enforce',
            'query_context_version': context,
            'retrieval_query_hmac': case.retrieval_query_hmac,
            'runtime_cost_snapshot_hmac': runtime_hmac,
            'security_scope_fingerprint': case.security_scope_fingerprint,
            'surface': case.surface,
        },
        'workflow_thread_id': None,
        'effect_key': None,
        'run_contract_version': 'rag-run:v2',
        'run_record_phase': 'admission',
        'total_charged_cost_usd': _ZERO,
        'projection_owner_fence_hmac': None,
        'started_at': now,
        'completed_at': None,
    }
    return parent, *children


def _projection_boundary():
    # The caller cannot forge an issued projection with a valid HMAC or by
    # mutating dataclass fields: the authority state is private to this closure.
    issued = WeakKeyDictionary()

    class ApprovedCaseClaimProjection:
        __slots__ = ('__weakref__',)

        def __init__(self):
            _refuse()

        @property
        def runtime_images(self):
            if type(self) is not ApprovedCaseClaimProjection or self not in issued:
                _refuse()
            return deepcopy(issued[self]['images'])

        def __repr__(self):
            return '<ApprovedCaseClaimProjection opaque>'

    def prepare(connection: Connection, *, manifest, payload, identity_secret):
        from backend.app.rag.release_ledger import (
            RagReleaseMutationSet,
            ReleaseRowPrimaryKey,
        )

        binding = _claim_binding(payload)
        manifest_hmac = case_claim_manifest_hmac(
            manifest, identity_secret=identity_secret
        )
        key = ReleaseRowPrimaryKey(
            'authorization',
            {
                name: payload[name]
                for name in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
            },
        )
        auth = RagReleaseMutationSet._snapshot(connection, key)
        if auth is None or auth['manifest_hmac'] != manifest_hmac:
            _refuse()
        case = next(
            (
                case
                for case in manifest.cases
                if case.case_id_hmac == payload['case_id_hmac']
            ),
            None,
        )
        if case is None or case.ordinal != auth['case_claim_count']:
            _refuse()
        # Versioned local allocation: next unoccupied positive ids, one UTC
        # clock sample. This is not a SQL sequence write or a caller row value.
        run_id = connection.scalar(select(func.coalesce(func.max(AgentRun.id), 0))) + 1
        child_id = (
            connection.scalar(
                select(func.coalesce(func.max(AgentRunCostComponent.id), 0))
            )
            + 1
        )
        images = _runtime_images(
            case,
            run_id=run_id,
            child_ids=(child_id, child_id + 1),
            now=datetime.now(UTC),
            secret=identity_secret,
        )
        if any(
            binding[name] != auth[name]
            for name in binding
            if name in auth and not name.startswith('execution_')
        ):
            _refuse()
        for name in ('execution_process_instance_hmac', 'execution_runner_fence_hmac'):
            require_lower_hmac(binding[name])
        from backend.app.rag.release_ledger import _runtime_projection

        canonical = canonical_json_bytes(
            {
                'assembly_version': _ASSEMBLY,
                'manifest_hmac': manifest_hmac,
                'binding': binding,
                'images': [
                    _runtime_projection(kind, image)
                    for kind, image in zip(
                        ('agent_run', 'cost_component', 'cost_component'),
                        images,
                        strict=True,
                    )
                ],
            }
        )
        projection = object.__new__(ApprovedCaseClaimProjection)
        issued[projection] = {
            'engine': connection.engine,
            'manifest': deepcopy(manifest),
            'manifest_hmac': manifest_hmac,
            'authorization': deepcopy(auth),
            'case': deepcopy(case),
            'binding': binding,
            'images': images,
            'canonical': canonical,
            'hmac': rag_identity_hmac(
                exact_utf8_bytes(canonical.decode('utf-8')),
                secret=identity_secret,
                schema_version='rag-live-case-claim-projection:v1',
                policy_version=_ASSEMBLY,
            ),
        }
        return projection

    def validate(
        projection,
        *,
        connection,
        payload,
        runtime_rows,
        case_rows,
        after_execution,
        observations,
        identity_secret,
    ):
        from backend.app.rag.release_ledger import (
            RagReleaseMutationSet,
            ReleaseRowPrimaryKey,
            _runtime_projection,
        )

        if (
            type(projection) is not ApprovedCaseClaimProjection
            or projection not in issued
        ):
            _refuse()
        state = issued[projection]
        binding = _claim_binding(payload)
        if (
            connection.engine is not state['engine']
            or binding != state['binding']
            or case_claim_manifest_hmac(
                state['manifest'], identity_secret=identity_secret
            )
            != state['manifest_hmac']
        ):
            _refuse()
        auth_key = ReleaseRowPrimaryKey(
            'authorization',
            {
                key: payload[key]
                for key in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
            },
        )
        auth = RagReleaseMutationSet._snapshot(connection, auth_key, for_update=True)
        expected_auth = state['authorization']
        if auth is None or any(
            auth[key] != expected_auth[key]
            for key in (
                'manifest_hmac',
                'approval_hmac',
                'base_generation',
                'baseline_hmac',
                'reviewer_roster_hmac',
                'approved_corpus_snapshot_hmac',
                'approved_provider_safety_snapshot_hmac',
                'provider_safety_envelope_digest',
                'validation_database_identity_hmac',
            )
        ):
            _refuse()
        case = state['case']
        run_hmac = rag_identity_hmac(
            {'agent_run_id': state['images'][0]['id']},
            secret=identity_secret,
            schema_version='rag-runtime-agent-run-id:v1',
            policy_version='rag-run:v2',
        )
        expected_case = {
            key: payload[key]
            for key in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
        } | {
            'case_id_hmac': case.case_id_hmac,
            'manifest_ordinal': case.ordinal,
            'state': 'claimed',
            'case_projection_hmac': None,
            'runtime_agent_run_id_hmac': run_hmac,
            'embedding_reserved_cost_usd': case.components[0].reserved_cost_usd,
            'generation_reserved_cost_usd': case.components[1].reserved_cost_usd,
            'total_reserved_cost_usd': sum(
                item.reserved_cost_usd for item in case.components
            ),
        }
        _assert_exact_image_types(case_rows, [expected_case])
        if (
            case_rows != [expected_case]
            or payload['runtime_agent_run_id_hmac'] != run_hmac
        ):
            _refuse()
        if any(
            Decimal(str(payload[f'case_{part}_reserved_cost_usd'])) != reserve
            for part, reserve in (
                ('embedding', case.components[0].reserved_cost_usd),
                ('generation', case.components[1].reserved_cost_usd),
                ('total', sum(item.reserved_cost_usd for item in case.components)),
            )
        ):
            _refuse()
        readiness = [
            row
            for kind, row in observations
            if kind == 'provider_readiness' and row['active']
        ]
        authority = [
            row for kind, row in observations if kind == 'provider_safety_authority'
        ]
        if (
            len(authority) != 1
            or len(readiness) != 2
            or authority[0]['envelope_digest']
            != expected_auth['provider_safety_envelope_digest']
        ):
            _refuse()
        for component in case.components:
            policy = asdict(component.policy)
            current = [
                row for row in readiness if row['component'] == policy['component']
            ]
            if (
                len(current) != 1
                or current[0]['state'] != 'ready'
                or current[0]['authority_id'] != authority[0]['id']
            ):
                _refuse()
            for name, value in policy.items():
                column = (
                    'authorized_' + name if name.startswith('fingerprint_') else name
                )
                if current[0][column] != value:
                    _refuse()
        expected_kinds = ('agent_run', 'cost_component', 'cost_component')
        if tuple(kind for kind, _ in runtime_rows) != expected_kinds:
            _refuse()
        for (_, actual_row), expected_row in zip(
            runtime_rows, state['images'], strict=True
        ):
            _assert_exact_image_types(actual_row, expected_row)
            if set(actual_row) != set(expected_row):
                _refuse()
            for key, value in expected_row.items():
                actual_value = actual_row[key]
                if isinstance(value, Decimal) and (
                    type(actual_value) is not Decimal or actual_value != value
                ):
                    _refuse()
                if (
                    key in {'started_at', 'created_at', 'updated_at'}
                    and not after_execution
                    and (
                        type(actual_value) is not type(value)
                        or actual_value.tzinfo is not UTC
                        or actual_value.fold != value.fold
                        or actual_value != value
                    )
                ):
                    _refuse()
        actual = [_runtime_projection(kind, row) for kind, row in runtime_rows]
        expected = [
            _runtime_projection(kind, row)
            for kind, row in zip(expected_kinds, state['images'], strict=True)
        ]
        if actual != expected:
            _refuse()
        canonical = canonical_json_bytes(
            {
                'assembly_version': _ASSEMBLY,
                'manifest_hmac': state['manifest_hmac'],
                'binding': state['binding'],
                'images': actual,
            }
        )
        if (
            canonical != state['canonical']
            or rag_identity_hmac(
                exact_utf8_bytes(canonical.decode('utf-8')),
                secret=identity_secret,
                schema_version='rag-live-case-claim-projection:v1',
                policy_version=_ASSEMBLY,
            )
            != state['hmac']
        ):
            _refuse()

    return ApprovedCaseClaimProjection, prepare, validate


(
    ApprovedCaseClaimProjection,
    prepare_case_claim_projection,
    validate_case_claim_projection,
) = _projection_boundary()
