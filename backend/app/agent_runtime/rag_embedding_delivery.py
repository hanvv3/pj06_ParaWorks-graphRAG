from __future__ import annotations

from backend.app.agent_runtime.rag_safety_identity import rag_identity_hmac
from backend.app.rag.retrieval import QueryEmbeddingCallResult


def embedding_dispatch_receipt_hmac(
    result: QueryEmbeddingCallResult,
    *,
    agent_run_id: int,
    dispatch_fence_hmac: str,
    process_instance_hmac: str,
    secret: bytes,
) -> str:
    """Authenticate transient validated output to its exact committed dispatch."""
    return rag_identity_hmac(
        {
            'agent_run_id': agent_run_id,
            'dispatch_fence_hmac': dispatch_fence_hmac,
            'process_instance_hmac': process_instance_hmac,
            'preparation_attempt_hmac': result.prepared.attempt_fence_hmac,
            'vector_float32_sha256': result.vector.canonical_big_endian_float32_sha256,
            'validated_input_tokens': result.validated_input_tokens,
            'actual_cost_usd': format(result.actual_cost_usd, '.6f'),
        },
        secret=secret,
        schema_version='rag-embedding-committed-delivery:v1',
        policy_version='rag-run:v2',
    )
