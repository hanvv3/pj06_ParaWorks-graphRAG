from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from backend.app.agent_runtime.fingerprints import keyed_fingerprint


class RagIdentityError(ValueError):
    pass


def _require_exact_mapping(
    value: object,
    keys: set[str],
    domain: str,
) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise RagIdentityError(f'{domain} identity shape is invalid')
    return value


def require_lower_hmac(value: object, field: str = 'identity') -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in '0123456789abcdef' for char in value)
    ):
        raise RagIdentityError(f'{field} must be a lowercase SHA-256 HMAC')
    return value


def rag_identity_hmac(
    value: object,
    *,
    secret: bytes,
    schema_version: str,
    policy_version: str = 'rag-answer:v2',
) -> str:
    if type(secret) is not bytes or not secret:
        raise RagIdentityError('identity signer is unavailable')
    return keyed_fingerprint(
        value,
        secret=secret,
        schema_version=schema_version,
        policy_version=policy_version,
    )


def identities_match(left: object, right: object) -> bool:
    try:
        return hmac.compare_digest(
            require_lower_hmac(left), require_lower_hmac(right)
        )
    except (TypeError, ValueError):
        return False


def admission_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'answer_provider_policy_snapshot_hmac', 'configured_backend',
        'current_text_hmac', 'cutover_stage', 'graph_version', 'mode',
        'query_context_version_bytes',
        'query_embedding_provider_policy_snapshot_hmac',
        'retrieval_policy_version', 'retrieval_query_hmac',
        'security_scope_fingerprint', 'surface',
    }, 'admission')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-admission-identity:v1',
        policy_version='rag-run:v2',
    )


def runtime_cost_identity(value: object, *, secret: bytes) -> str:
    payload = _require_exact_mapping(value, {
        'agent_run_id', 'components', 'parent_outcome',
        'parent_run_record_phase', 'parent_status', 'run_contract_version',
        'snapshot_stage', 'total_charged_cost_usd', 'total_reserved_cost_usd',
    }, 'runtime cost')
    components = payload['components']
    component_keys = {
        'actual_input_tokens', 'actual_output_tokens', 'attempted',
        'authorized_cost_policy_version_bytes',
        'authorized_model_config_snapshot_hmac',
        'authorized_model_config_version_bytes',
        'authorized_policy_snapshot_hmac',
        'authorized_token_estimator_version_bytes', 'charge_basis',
        'charged_cost_usd', 'component', 'dispatch_count',
        'dispatch_fence_hmac', 'dispatch_state', 'model_bytes', 'overrun',
        'process_instance_hmac', 'provider_bytes', 'reserved_cost_usd',
        'reserved_input_tokens', 'reserved_output_tokens',
    }
    if (
        type(components) is not list
        or len(components) != 2
        or any(type(item) is not dict or set(item) != component_keys for item in components)
        or [item['component'] for item in components]
        != ['query_embedding', 'answer_generation']
    ):
        raise RagIdentityError('runtime cost component identity shape is invalid')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-runtime-cost-snapshot:v1',
        policy_version='rag-run:v2',
    )


def dispatch_fence_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'agent_run_id', 'approval_hmac', 'approval_id_hmac', 'case_id_hmac',
        'component', 'dispatch_count', 'dispatch_nonce_hex',
        'prepared_input_hmac', 'process_instance_hmac',
        'provider_safety_snapshot_hmac', 'release_to_generation',
        'retrieval_query_hmac',
    }, 'dispatch fence')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-dispatch-fence:v1',
        policy_version='rag-run:v2',
    )


def process_instance_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'designated_environment_id_bytes', 'designated_host_id_bytes',
        'process_boot_nonce_hex', 'process_role',
    }, 'process instance')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-process-instance:v1',
        policy_version='rag-run:v2',
    )


def dead_process_attestation_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'agent_run_id', 'dead_process_instance_hmac', 'recovery_outcome',
    }, 'dead process attestation')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-dead-process-attestation:v1',
        policy_version='rag-run:v2',
    )


def provider_safety_snapshot_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'active_family', 'authority_uuid', 'envelope_digest',
        'global_safety_generation',
    }, 'provider safety')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-provider-safety-snapshot:v1',
        policy_version='rag-provider-safety:v1',
    )


def projection_owner_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'advisory_lock_identity_digest', 'agent_run_id', 'lock_backend',
        'lock_namespace', 'owner_nonce_hex', 'owner_phase',
        'process_instance_hmac',
    }, 'projection owner')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-projection-owner-fence:v1',
        policy_version='rag-run:v2',
    )


@dataclass(frozen=True, slots=True)
class FinalProductIdentityInput:
    admission_cache_identity_hmac: str
    rag_result_hmac: str
    surface: Literal['ask', 'search', 'assistant']

    def __post_init__(self) -> None:
        require_lower_hmac(
            self.admission_cache_identity_hmac,
            'admission_cache_identity_hmac',
        )
        require_lower_hmac(self.rag_result_hmac, 'rag_result_hmac')
        if self.surface not in {'ask', 'search', 'assistant'}:
            raise RagIdentityError('final product surface is invalid')


def final_product_identity(
    value: FinalProductIdentityInput,
    *,
    secret: bytes,
) -> str:
    if type(value) is not FinalProductIdentityInput:
        raise TypeError('typed final product identity is required')
    return rag_identity_hmac(
        {
            'admission_cache_identity_hmac': value.admission_cache_identity_hmac,
            'rag_result_hmac': value.rag_result_hmac,
            'surface': value.surface,
        }, secret=secret, schema_version='rag-final-product-identity:v1',
        policy_version='rag-answer-cache-key:v2',
    )


def final_shadow_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'admission_cache_identity_hmac', 'outcome',
        'runtime_cost_snapshot_hmac', 'surface',
    }, 'final shadow')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-final-shadow-identity:v1',
        policy_version='rag-shadow:v2',
    )


def final_error_identity(value: object, *, secret: bytes) -> str:
    _require_exact_mapping(value, {
        'admission_cache_identity_hmac', 'answer_question_hmac',
        'configured_backend', 'current_text_hmac', 'fallback_category',
        'graph_version', 'outcome',
        'prepared_model_influence_observation_hmac', 'retrieval_query_hmac',
        'runtime_cost_snapshot_hmac', 'security_scope_fingerprint', 'surface',
    }, 'final error')
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-final-error-identity:v1',
        policy_version='rag-run:v2',
    )


def implementation_plan_reference_identity(value: object, *, secret: bytes) -> str:
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-implementation-plan-reference:v1',
        policy_version='rag-run:v2',
    )
