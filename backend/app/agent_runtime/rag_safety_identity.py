from __future__ import annotations

import hmac

from backend.app.agent_runtime.fingerprints import keyed_fingerprint


class RagIdentityError(ValueError):
    pass


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
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-admission-cache-identity:v1'
    )


def runtime_cost_identity(value: object, *, secret: bytes) -> str:
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-runtime-cost-snapshot:v1',
        policy_version='rag-run:v2',
    )


def dispatch_fence_identity(value: object, *, secret: bytes) -> str:
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-dispatch-fence:v1',
        policy_version='rag-run:v2',
    )


def process_instance_identity(value: object, *, secret: bytes) -> str:
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-process-instance:v1',
        policy_version='rag-run:v2',
    )


def provider_safety_snapshot_identity(value: object, *, secret: bytes) -> str:
    return rag_identity_hmac(
        value, secret=secret, schema_version='rag-provider-safety-snapshot:v1',
        policy_version='rag-provider-safety:v1',
    )
