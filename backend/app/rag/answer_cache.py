"""Internal cache identity/eligibility, not permission or execution authority.

C2 must obtain fresh slots/preparation and perform E's canonical revalidation
again before publication. No cache result carries an old provider receipt.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import asdict, dataclass
from time import time
from typing import Protocol

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    exact_utf8_bytes,
    security_scope_fingerprint,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    RagAnswerOutputValidator,
    ValidatedAnswerBlocks,
)
from backend.app.core.config import Settings
from backend.app.rag.evidence_projection import (
    PreparedModelInfluenceSet,
    ProjectionFence,
    _valid_prepared_influence_set,
    verify_answer_model_influence,
)
from backend.app.rag.retrieval import EvidenceSlot
from backend.app.rag.serving_contracts import (
    require_exact_nonblank,
    require_lower_hex_64,
)

ANSWER_CACHE_TTL_SECONDS = 3600


def exact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )


def cache_hmac(value: object, *, settings: Settings, domain: str) -> str:
    secret, version = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {'key_version': version, 'bytes': exact_utf8_bytes(exact_json(value))},
        secret=secret,
        schema_version=domain,
        policy_version='rag-answer-cache:v1',
    )


@dataclass(frozen=True, slots=True)
class AnswerCacheVersions:
    prompt_hmac: str
    model_hmac: str
    output_hmac: str
    policy_hmac: str
    retrieval_policy: str
    graph_policy: str
    seed_backend: str
    effective_backend: str

    def __post_init__(self) -> None:
        for value in (
            self.prompt_hmac,
            self.model_hmac,
            self.output_hmac,
            self.policy_hmac,
        ):
            require_lower_hex_64(value)
        for value in (
            self.retrieval_policy,
            self.graph_policy,
            self.seed_backend,
            self.effective_backend,
        ):
            require_exact_nonblank(value)


@dataclass(frozen=True, slots=True)
class AnswerCacheKey:
    key_hmac: str
    scope_hmac: str
    dependencies_json: str
    versions: AnswerCacheVersions
    scope: SecurityScope
    prepared: PreparedModelInfluenceSet


@dataclass(frozen=True, slots=True)
class AnswerCacheHit:
    """Authenticated storage result; C2 must still revalidate live authority."""

    answer: ValidatedAnswerBlocks
    key_hmac: str
    scope_hmac: str
    value_hmac: str
    created_at: int
    expires_at: int


def build_answer_cache_key(
    *,
    scope: SecurityScope,
    prepared: PreparedModelInfluenceSet,
    versions: AnswerCacheVersions,
    settings: Settings,
) -> AnswerCacheKey:
    if (
        type(scope) is not SecurityScope
        or type(prepared) is not PreparedModelInfluenceSet
        or type(versions) is not AnswerCacheVersions
    ):
        raise ValueError('invalid cache identity')
    scope.__post_init__()
    versions.__post_init__()
    fence = ProjectionFence(
        prepared.prepared_corpus_generation,
        prepared.prepared_index_generation,
        prepared.prepared_corpus_generation,
        prepared.prepared_index_generation,
        prepared.prepared_readiness_hmac,
        prepared.prepared_readiness_hmac,
    )
    # Reuse the exact E v3 authentication domain, including all ordered paths.
    if not _valid_prepared_influence_set(prepared, fence=fence, settings=settings):
        raise ValueError('unauthenticated prepared cache influence')
    scope_hmac = security_scope_fingerprint(scope, settings=settings)
    if prepared.graph_paths:
        if prepared.graph_scope != scope:
            raise ValueError('cache graph scope mismatch')
        for path in prepared.graph_paths:
            path.__post_init__()
            if any(node.scope_id != scope_hmac for node in path.nodes):
                raise ValueError('cache graph scope mismatch')
    dependencies = exact_json(
        {
            'contract': 'rag-prepared-model-influence-set:v3',
            'observations': [asdict(item) for item in prepared.observations],
            'graph_paths': [asdict(path) for path in prepared.graph_paths],
            'prepared_corpus_generation': prepared.prepared_corpus_generation,
            'prepared_index_generation': prepared.prepared_index_generation,
            'prepared_readiness_hmac': prepared.prepared_readiness_hmac,
            'rendered_input_hmac': prepared.rendered_input_hmac,
            'aggregate_observation_hmac': prepared.aggregate_observation_hmac,
        }
    )
    digest = cache_hmac(
        {
            'scope_hmac': scope_hmac,
            'dependencies': dependencies,
            'versions': asdict(versions),
        },
        settings=settings,
        domain='rag-answer-cache-key:v1',
    )
    return AnswerCacheKey(digest, scope_hmac, dependencies, versions, scope, prepared)


def validate_cache_key(
    key: AnswerCacheKey, *, slots: tuple[EvidenceSlot, ...], settings: Settings
) -> None:
    if type(key) is not AnswerCacheKey or key != build_answer_cache_key(
        scope=key.scope, prepared=key.prepared, versions=key.versions, settings=settings
    ):
        raise ValueError('cache key authentication failed')
    verify_answer_model_influence(slots, key.prepared.observations, settings=settings)


def eligible_answer_payload(
    answer: object,
    *,
    slots: tuple[EvidenceSlot, ...],
    validator: RagAnswerOutputValidator,
) -> dict | None:
    if (
        type(answer) is not ValidatedAnswerBlocks
        or not answer.blocks
        or answer.insufficient_reason is not None
    ):
        return None
    try:
        payload = {
            'answer_blocks': [
                {
                    'text': block.text,
                    'evidence_slot_ids': list(block.evidence_slot_ids),
                    'support_mode': block.support_mode,
                }
                for block in answer.blocks
            ],
            'insufficient_evidence_reason': None,
        }
        # A dataclass alone is not validation: recompute all block/assembled HMACs.
        if validator.validate(payload, slots=slots) != answer:
            return None
        return payload
    except (ValueError, TypeError, AttributeError):
        return None


class AnswerCache(Protocol):
    def get(
        self, key: AnswerCacheKey, *, slots: tuple[EvidenceSlot, ...]
    ) -> AnswerCacheHit | None: ...
    def put(
        self,
        key: AnswerCacheKey,
        *,
        answer: ValidatedAnswerBlocks,
        slots: tuple[EvidenceSlot, ...],
    ) -> bool: ...
    def cleanup(self, *, limit: int = 100, strict: bool = False) -> int: ...


def runtime_answer_cache_key(*, scope, prepared, text, retrieval, settings):
    """Bind the current runtime policy and complete Assistant retrieval context."""
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
    from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
        build_answer_output_schema_hmac,
        build_answer_prompt_renderer_hmac,
    )

    prompt = build_answer_prompt_renderer_hmac(settings)
    output = build_answer_output_schema_hmac(settings)
    policy = RagCostPolicy(
        settings=settings,
        answer_output_schema_hmac=output,
        answer_prompt_renderer_hmac=prompt,
    )
    versions = AnswerCacheVersions(
        prompt_hmac=prompt,
        model_hmac=policy.answer_model_config_snapshot_hmac,
        output_hmac=output,
        policy_hmac=cache_hmac(
            {
                'policy': policy.authorized_policy_snapshot_hmac('answer_generation'),
                'retrieval_query': text.retrieval_query_hmac,
                'context_version': text.query_context_version,
                'question': text.answer_question_hmac,
            },
            settings=settings,
            domain='rag-answer-cache-context-policy:v1',
        ),
        retrieval_policy='rag-retrieval-policy:v2.0',
        graph_policy='company-memory-rag-answer-v2.0:'
        + (retrieval.graph_policy_version or 'graph-disabled'),
        seed_backend=retrieval.configured_backend,
        effective_backend=retrieval.effective_backend,
    )
    return build_answer_cache_key(
        scope=scope, prepared=prepared, versions=versions, settings=settings
    )


def validate_answer_cache_hit(hit, key, *, slots, settings, now=None):
    """Reauthenticate the exact reusable value, including expiry at publication."""
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy

    validate_cache_key(key, slots=slots, settings=settings)
    if type(hit) is not AnswerCacheHit:
        raise ValueError('cache hit carrier is invalid')
    policy = RagCostPolicy(
        settings=settings,
        answer_output_schema_hmac=key.versions.output_hmac,
        answer_prompt_renderer_hmac=key.versions.prompt_hmac,
    )
    payload = eligible_answer_payload(
        hit.answer,
        slots=slots,
        validator=RagAnswerOutputValidator(signer=policy.sign_answer_artifact),
    )
    instant = time() if now is None else now
    if (
        payload is None
        or hit.key_hmac != key.key_hmac
        or hit.scope_hmac != key.scope_hmac
        or type(hit.created_at) is not int
        or type(hit.expires_at) is not int
        or not 0 <= hit.created_at <= instant < hit.expires_at
        or not 0 < hit.expires_at - hit.created_at <= ANSWER_CACHE_TTL_SECONDS
    ):
        raise ValueError('cache hit identity or expiry changed')
    expected = cache_hmac(
        {
            'key_hmac': key.key_hmac,
            'scope_hmac': key.scope_hmac,
            'dependencies_json': key.dependencies_json,
            'answer_json': exact_json(payload),
            'created_at': hit.created_at,
            'expires_at': hit.expires_at,
        },
        settings=settings,
        domain='rag-answer-cache-value:v1',
    )
    if not hmac.compare_digest(hit.value_hmac, expected):
        raise ValueError('cache hit signature changed')
    return {
        'answer_finalization_mode': 'answer-cache-hit:v1',
        'answer_cache_key_hmac': key.key_hmac,
        'answer_cache_value_hmac': hit.value_hmac,
        'answer_cache_expires_at': hit.expires_at,
    }
