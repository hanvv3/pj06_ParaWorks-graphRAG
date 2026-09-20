"""Internal cache identity/eligibility, not permission or execution authority.

C2 must obtain fresh slots/preparation and perform E's canonical revalidation
again before publication. No cache result carries an old provider receipt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    def cleanup(self, *, limit: int = 100) -> int: ...
