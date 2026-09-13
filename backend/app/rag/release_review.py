"""Frozen live-gate preview and case projection; no execution authorization issuer.

The manifest preimage is checked against the immutable, already reviewed
authorization row. Signing arbitrary submitted SQL values cannot authorize them.
Runtime ids/clock are allocated by this module, never taken from SQL literals.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal
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
from backend.app.rag.release_authority import RagReleaseSnapshot

if TYPE_CHECKING:
    from backend.app.agent_runtime.rag_v2_identity import SecurityScope
    from backend.app.agents.rag_orchestrator_agent.v2_input import (
        AssistantContextMessage,
    )
    from backend.app.rag.serving_contracts import SupportMode

_ORDER = ('query_embedding', 'answer_generation')
_ZERO = Decimal('0.000000')
_VERSION = 'rag-live-case-claim-manifest:v1'
_ASSEMBLY = 'rag-live-case-claim-assembly:v1'

LIVE_FIXTURE_PATH = 'backend/tests/fixtures/rag_v2_live_gate_30.json'
LIVE_EVALUATOR_PATH = 'backend/app/rag/release_quality.py'
LIVE_FIXTURE_VERSION = 'rag-live-quality-30:v1'
LIVE_RUBRIC_VERSION = 'rag-live-quality-rubric:v1'
LIVE_RETRIEVER_SOURCE_PATHS = (
    'backend/app/agent_runtime/rag_v2_identity.py',
    'backend/app/agents/rag_orchestrator_agent/service.py',
    'backend/app/agents/rag_orchestrator_agent/v2_input.py',
    'backend/app/rag/evidence_projection.py',
    'backend/app/rag/keyword_retriever.py',
    'backend/app/rag/lexical_projection.py',
    'backend/app/rag/pgvector_retriever.py',
    'backend/app/rag/pgvector_store.py',
    'backend/app/rag/retrieval.py',
    'backend/app/rag/search_store.py',
    'backend/app/rag/shadow.py',
    'backend/app/rag/source_observations.py',
    'backend/app/rag/trusted_evidence.py',
)
_LIVE_POLICY = 'rag-live-gate:v1'
_DISTRIBUTION = {
    'ask_keyword': 10,
    'ask_pgvector': 5,
    'assistant_keyword': 10,
    'assistant_pgvector': 5,
}
_CONTEXT_DISTRIBUTION = {
    'keyword_with_prior_context': 5,
    'keyword_without_prior_context': 5,
    'pgvector_with_prior_context': 3,
    'pgvector_without_prior_context': 2,
}
_CASE_KEYS = frozenset(
    {
        'ordinal',
        'case_id',
        'case_kind',
        'surface',
        'configured_backend',
        'question_fixture_id',
        'security_scope_fixture_id',
        'prior_context_fixture_id',
        'expected_no_answer',
        'allowed_support_modes',
        'allowed_slot_ids',
        'relevant_serving_fixture_ids',
        'required_serving_fixture_ids',
        'query_embedding_required',
        'answer_generation_required',
        'query_embedding_reserved_input_tokens',
        'answer_generation_reserved_input_tokens',
        'answer_generation_reserved_output_tokens',
        'query_embedding_reserved_cost_usd',
        'answer_generation_reserved_cost_usd',
        'case_total_reserved_cost_usd',
    }
)


class LiveGatePreviewError(ValueError):
    """Bounded reason codes; never include fixture content or secret values."""

    def __init__(self, code='manifest_invalid'):
        self.code = code
        super().__init__(code)


def _live_require(condition, code='manifest_invalid'):
    if not condition:
        raise LiveGatePreviewError(code)


def _fixture_id(value):
    _live_require(
        type(value) is str and re.fullmatch(r'[a-z][a-z0-9-]{0,63}', value) is not None
    )
    return value


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        _live_require(key not in result)
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class FrozenCorpusMember:
    ordinal: int
    serving_identity_hmac: str
    serving_version_fingerprint: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    effective_permission: Literal['public', 'internal', 'restricted']
    support_mode: SupportMode
    vector_index_state_hmac: str | None


@dataclass(frozen=True, slots=True)
class FrozenCorpusSnapshot:
    corpus_generation: int
    vector_index_generation: int
    embedding_model_bytes: bytes
    index_policy_version_bytes: bytes
    pgvector_cosine_policy_version: str
    members: tuple[FrozenCorpusMember, ...]
    corpus_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class LiveGateLimits:
    case_claims: Literal[30] = 30
    answer_generation_dispatches: Literal[30] = 30
    query_embedding_dispatches: Literal[10] = 10
    total_dispatches: Literal[40] = 40
    case_max_cost_usd: Decimal = Decimal('0.012000')
    total_max_cost_usd: Decimal = Decimal('0.360000')

    def __post_init__(self):
        for actual, expected in (
            (self.case_claims, 30),
            (self.answer_generation_dispatches, 30),
            (self.query_embedding_dispatches, 10),
            (self.total_dispatches, 40),
            (self.case_max_cost_usd, Decimal('0.012000')),
            (self.total_max_cost_usd, Decimal('0.360000')),
        ):
            _live_require(
                type(actual) is type(expected) and actual == expected, 'limits_invalid'
            )


@dataclass(frozen=True, slots=True)
class FrozenLiveManifestCase:
    ordinal: int
    case_id_hmac: str
    case_kind: Literal['positive', 'hard_negative']
    surface: Literal['ask', 'assistant']
    configured_backend: Literal['keyword', 'pgvector']
    question_fixture_id: str
    security_scope_fixture_id: str
    prior_context_fixture_id: str | None
    query_bytes_hmac: str
    expected_no_answer: bool
    allowed_support_modes: tuple[SupportMode, ...]
    allowed_slot_ids: tuple[str, ...]
    relevant_serving_identity_hmacs: tuple[str, ...]
    required_serving_identity_hmacs: tuple[str, ...]
    query_embedding_required: bool
    answer_generation_required: Literal[True]
    query_embedding_reserved_cost_usd: Decimal
    answer_generation_reserved_cost_usd: Decimal
    case_total_reserved_cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class FrozenLiveManifestSnapshot:
    live_gate_contract_version: str
    fixture_manifest_version: str
    fixture_manifest_path: str
    fixture_manifest_sha256: str
    fixture_manifest_hmac: str
    manifest_hmac: str
    clean_git_commit: str
    rubric_version: str
    cases: tuple[FrozenLiveManifestCase, ...]
    executable: FrozenCaseClaimManifest
    ask_keyword_count: int = 10
    ask_pgvector_count: int = 5
    assistant_keyword_count: int = 10
    assistant_pgvector_count: int = 5
    keyword_with_prior_context_count: int = 5
    keyword_without_prior_context_count: int = 5
    pgvector_with_prior_context_count: int = 3
    pgvector_without_prior_context_count: int = 2
    limits: LiveGateLimits = LiveGateLimits()


@dataclass(frozen=True, slots=True, repr=False)
class LiveGatePreviewInputs:
    """In-memory data read from an approved snapshot under its owner's locks.

    The reader owns corpus completeness, current scope and authority validation.
    This value is neither a permission capability nor execution authorization.
    Raw synthetic request/context inputs are never retained in the preview.
    """

    corpus: FrozenCorpusSnapshot
    questions: tuple[tuple[str, str], ...]
    security_scopes: tuple[tuple[str, SecurityScope], ...]
    prior_contexts: tuple[tuple[str, tuple[AssistantContextMessage, ...]], ...]
    serving_references: tuple[tuple[str, str], ...]
    provider_snapshot_bytes: bytes
    release: RagReleaseSnapshot
    approved_provider_safety_snapshot_hmac: str
    reviewer_roster_hmac: str
    implementation_plan_reference_hmac: str
    release_row_counts: tuple[tuple[str, int], ...]

    def __repr__(self):
        return '<LiveGatePreviewInputs redacted>'


@dataclass(frozen=True, slots=True)
class HardNegativeOracleRequest:
    fixture_manifest_hmac: str
    case_id_hmac: str
    query_bytes_hmac: str
    security_scope_fingerprint: str
    corpus_snapshot_hmac: str
    configured_backend: Literal['keyword', 'pgvector']


@dataclass(frozen=True, slots=True)
class FrozenOracleVisibleCandidate:
    slot_id: str
    serving_identity_hmac: str
    entailment: Literal['not_entailed', 'entailed', 'ambiguous']


@dataclass(frozen=True, slots=True)
class FrozenHardNegativeOracleResult:
    """Reader-owned evaluation of frozen, permission-filtered candidates.

    The locked adapter must derive this from the requested corpus/query/scope
    and its frozen non-entailment oracle, never a declaration or caller flag.
    This result is provenance evidence, not an authorization capability.
    """

    request: HardNegativeOracleRequest
    oracle_definition_hmac: str
    visible_candidates: tuple[FrozenOracleVisibleCandidate, ...]
    hidden_match_count: int


@dataclass(frozen=True, slots=True, repr=False)
class LiveGatePreview:
    manifest: FrozenLiveManifestSnapshot
    corpus: FrozenCorpusSnapshot
    limits: LiveGateLimits
    total_reserved_cost_usd: Decimal
    baseline_definition_bytes: bytes
    baseline_hmac: str
    canonical_bytes: bytes
    preview_hmac: str
    source_binding: VerifiedLiveManifestSource | None = None

    def sanitized_payload(self):
        value = json.loads(self.canonical_bytes)
        # Complete execution preimages are retained in memory for B's approved
        # verifier. Operator output needs aggregate identities, never SQL images.
        value.pop('executable_preimage')
        return value | {
            'preview_hmac': self.preview_hmac,
            'provider_dispatch_count': 0,
            'authorization_issued': False,
        }

    def __repr__(self):
        return '<LiveGatePreview provider-free; no execution authority>'


def _live_hmac(value, schema, secret):
    return rag_identity_hmac(
        value, secret=secret, schema_version=schema, policy_version=_LIVE_POLICY
    )


def _live_digest(value):
    try:
        return require_lower_hmac(value)
    except ValueError:
        raise LiveGatePreviewError('snapshot_invalid') from None


def _live_number(value, minimum=0):
    _live_require(
        type(value) is int and minimum <= value <= 2**63 - 1, 'snapshot_invalid'
    )


def _corpus_payload(value):
    _live_require(type(value) is FrozenCorpusSnapshot, 'corpus_invalid')
    _live_number(value.corpus_generation)
    _live_number(value.vector_index_generation)
    _live_require(
        type(value.embedding_model_bytes) is bytes
        and value.embedding_model_bytes == b'text-embedding-3-small',
        'corpus_invalid',
    )
    _live_require(
        type(value.index_policy_version_bytes) is bytes
        and value.index_policy_version_bytes == b'rag-v2-serving-index:v1',
        'corpus_invalid',
    )
    _live_require(
        type(value.pgvector_cosine_policy_version) is str
        and value.pgvector_cosine_policy_version == 'pgvector-cosine-indexable:v1',
        'corpus_invalid',
    )
    _live_require(
        type(value.members) is tuple and len(value.members) > 0, 'corpus_invalid'
    )
    identities, members = [], []
    for ordinal, member in enumerate(value.members):
        _live_require(type(member) is FrozenCorpusMember, 'corpus_invalid')
        _live_number(member.ordinal)
        _live_require(member.ordinal == ordinal, 'corpus_invalid')
        for name in (
            'serving_identity_hmac',
            'serving_version_fingerprint',
            'model_content_hmac',
            'canonical_citation_projection_hmac',
        ):
            _live_digest(getattr(member, name))
        if member.vector_index_state_hmac is not None:
            _live_digest(member.vector_index_state_hmac)
        _live_require(
            type(member.effective_permission) is str
            and member.effective_permission in ('public', 'internal', 'restricted'),
            'corpus_invalid',
        )
        _live_require(
            type(member.support_mode) is str
            and member.support_mode in ('trusted_fact', 'source_observation'),
            'corpus_invalid',
        )
        identities.append(member.serving_identity_hmac)
        members.append(asdict(member))
    _live_require(identities == sorted(set(identities)), 'corpus_invalid')
    return {
        'corpus_generation': value.corpus_generation,
        'vector_index_generation': value.vector_index_generation,
        'embedding_model_bytes': exact_utf8_bytes(value.embedding_model_bytes.decode()),
        'index_policy_version_bytes': exact_utf8_bytes(
            value.index_policy_version_bytes.decode()
        ),
        'pgvector_cosine_policy_version': value.pgvector_cosine_policy_version,
        'members': members,
    }


def freeze_live_corpus(
    *,
    corpus_generation,
    vector_index_generation,
    embedding_model_bytes,
    index_policy_version_bytes,
    pgvector_cosine_policy_version,
    members,
    identity_secret,
) -> FrozenCorpusSnapshot:
    value = FrozenCorpusSnapshot(
        corpus_generation,
        vector_index_generation,
        embedding_model_bytes,
        index_policy_version_bytes,
        pgvector_cosine_policy_version,
        members,
        '',
    )
    from dataclasses import replace

    return replace(
        value,
        corpus_snapshot_hmac=_live_hmac(
            _corpus_payload(value), 'rag-live-corpus-snapshot:v1', identity_secret
        ),
    )


def _git_read(repository, *arguments):
    try:
        result = subprocess.run(
            ['git', '-C', str(repository), *arguments],
            capture_output=True,
            timeout=15,
            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
        )
    except (OSError, subprocess.SubprocessError):
        raise LiveGatePreviewError('git_unavailable') from None
    _live_require(result.returncode == 0, 'committed_source_unavailable')
    return result.stdout


def _clean_commit(repository, expected_commit):
    _live_require(
        type(expected_commit) is str
        and re.fullmatch('[0-9a-f]{40}', expected_commit) is not None,
        'commit_invalid',
    )
    _live_require(
        _git_read(repository, 'rev-parse', 'HEAD').decode().strip() == expected_commit,
        'commit_changed',
    )
    _live_require(
        not _git_read(repository, 'status', '--porcelain=v1', '--untracked-files=all'),
        'worktree_dirty',
    )


def _committed_bytes(repository, commit, path):
    target = repository / path
    _live_require(
        target.is_file()
        and not any(item.is_symlink() for item in (target, *target.parents)),
        'evaluator_unavailable'
        if path == LIVE_EVALUATOR_PATH
        else 'committed_source_unavailable',
    )
    committed = _git_read(repository, 'show', f'{commit}:{path}')
    try:
        current = target.read_bytes()
    except OSError:
        raise LiveGatePreviewError('committed_source_unavailable') from None
    _live_require(current == committed, 'committed_source_changed')
    return committed


def _source_snapshot(repository, expected_commit):
    repository = Path(repository).resolve()
    _clean_commit(repository, expected_commit)
    files = {
        path: _committed_bytes(repository, expected_commit, path)
        for path in (
            LIVE_EVALUATOR_PATH,
            LIVE_FIXTURE_PATH,
            *LIVE_RETRIEVER_SOURCE_PATHS,
        )
    }
    _clean_commit(repository, expected_commit)
    return files


def require_live_preview_sources(repository):
    """Read-only CLI readiness check. Missing Task25 evaluator stays a refusal."""
    repository = Path(repository).resolve()
    _live_require((repository / LIVE_EVALUATOR_PATH).is_file(), 'evaluator_unavailable')
    commit = _git_read(repository, 'rev-parse', 'HEAD').decode().strip()
    _source_snapshot(repository, commit)
    return commit


def _reference_map(entries, expected):
    _live_require(type(entries) is tuple, 'fixture_mapping_invalid')
    result = {}
    for entry in entries:
        _live_require(
            type(entry) is tuple and len(entry) == 2, 'fixture_mapping_invalid'
        )
        key, value = entry
        _fixture_id(key)
        _live_require(key not in result, 'fixture_mapping_invalid')
        result[key] = value
    _live_require(set(result) == set(expected), 'fixture_mapping_invalid')
    return result


def _provider_snapshot(inputs, secret):
    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier

    _live_require(
        type(inputs.provider_snapshot_bytes) is bytes, 'provider_snapshot_invalid'
    )
    try:
        body = json.loads(
            inputs.provider_snapshot_bytes, object_pairs_hook=_unique_json_object
        )
    except (ValueError, UnicodeError):
        raise LiveGatePreviewError('provider_snapshot_invalid') from None
    _live_require(
        type(body) is dict
        and set(body)
        == {
            'active_families',
            'authority_uuid',
            'designated_environment_id',
            'envelope_digest',
            'fingerprint_key_material_verifier',
            'fingerprint_key_version',
            'global_safety_generation',
        },
        'provider_snapshot_invalid',
    )
    _live_require(
        type(body['authority_uuid']) is str
        and str(UUID(body['authority_uuid'])) == body['authority_uuid'],
        'provider_snapshot_invalid',
    )
    _live_require(
        type(body['designated_environment_id']) is str
        and bool(body['designated_environment_id']),
        'provider_snapshot_invalid',
    )
    _live_require(
        _live_hmac(
            {
                'designated_environment_id_bytes': exact_utf8_bytes(
                    body['designated_environment_id']
                )
            },
            'rag-live-designated-environment-id:v1',
            secret,
        )
        == inputs.release.designated_environment_id_hmac,
        'provider_environment_changed',
    )
    _live_require(
        type(body['fingerprint_key_version']) is str
        and bool(body['fingerprint_key_version']),
        'provider_snapshot_invalid',
    )
    _live_number(body['global_safety_generation'])
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    _live_require(
        body['fingerprint_key_material_verifier']
        == verifier
        == inputs.release.fingerprint_key_material_verifier,
        'key_changed',
    )
    _live_require(
        body['fingerprint_key_version'] == inputs.release.fingerprint_key_version,
        'key_changed',
    )
    _live_digest(body['envelope_digest'])
    rows = body['active_families']
    _live_require(type(rows) is list and len(rows) == 2, 'provider_snapshot_invalid')
    policies = {}
    for component, row in zip(
        ('answer_generation', 'query_embedding'), rows, strict=True
    ):
        _live_require(
            type(row) is dict
            and set(row)
            == {
                'authorized_cost_policy_version',
                'authorized_fingerprint_key_version',
                'authorized_model_config_snapshot_hmac',
                'authorized_model_config_version',
                'authorized_policy_snapshot_hmac',
                'authorized_token_estimator_version',
                'component',
                'family_safety_generation',
                'model',
                'provider',
                'reasoning_or_config_identity',
                'state',
                'state_version',
            },
            'provider_snapshot_invalid',
        )
        query = component == 'query_embedding'
        expected = {
            'component': component,
            'provider': 'openai',
            'model': 'text-embedding-3-small' if query else 'gpt-5.4-mini-2026-03-17',
            'reasoning_or_config_identity': 'dimensions:1536' if query else 'none',
            'authorized_model_config_version': 'rag-query-embedding-config:v1'
            if query
            else 'rag-answer-model-config:v1',
            'authorized_cost_policy_version': 'rag-query-embedding-cost:v1'
            if query
            else 'rag-answer-cost:v1',
            'authorized_token_estimator_version': 'openai-cl100k-text-embedding-3-small:v1'
            if query
            else 'openai-o200k-rag-answer:v1',
            'authorized_fingerprint_key_version': body['fingerprint_key_version'],
            'state': 'ready',
        }
        _live_require(
            all(
                type(row[name]) is str and row[name] == item
                for name, item in expected.items()
            ),
            'provider_snapshot_invalid',
        )
        _live_number(row['family_safety_generation'])
        _live_number(row['state_version'], 1)
        for name in (
            'authorized_model_config_snapshot_hmac',
            'authorized_policy_snapshot_hmac',
        ):
            _live_digest(row[name])
        policies[component] = AuthorizedProviderPolicySnapshot(
            **{
                field.name: body['fingerprint_key_material_verifier']
                if field.name == 'fingerprint_key_material_verifier'
                else body['fingerprint_key_version']
                if field.name == 'fingerprint_key_version'
                else row[field.name]
                for field in fields(AuthorizedProviderPolicySnapshot)
            }
        )
    _live_require(
        _live_hmac(body, 'rag-provider-safety-approved-snapshot:v1', secret)
        == _live_digest(inputs.approved_provider_safety_snapshot_hmac),
        'provider_snapshot_changed',
    )
    return body, policies


def _pgvector_baseline_members(reader, corpus):
    """The locked reader owns the complete baseline participation roster."""
    read = getattr(reader, 'read_pgvector_baseline_members', None)
    _live_require(callable(read), 'pgvector_member_invalid')
    identities = read()
    _live_require(
        type(identities) is tuple
        and bool(identities)
        and all(type(item) is str for item in identities),
        'pgvector_member_invalid',
    )
    _live_require(
        identities == tuple(sorted(set(identities))), 'pgvector_member_invalid'
    )
    members = {member.serving_identity_hmac: member for member in corpus.members}
    for identity in identities:
        _live_digest(identity)
        _live_require(
            identity in members
            and members[identity].vector_index_state_hmac is not None,
            'pgvector_member_invalid',
        )
    return identities


def _build_manifest(raw, commit, inputs, secret, policies, pgvector_members):
    from types import SimpleNamespace

    from backend.app.agent_runtime.rag_v2_identity import (
        SecurityScope,
        security_scope_fingerprint,
    )
    from backend.app.agents.rag_orchestrator_agent.v2_input import (
        AssistantContextMessage,
        prepare_assistant_request_text,
        prepare_direct_request_text,
    )

    declaration = parse_live_fixture(raw)
    rows = declaration['cases']
    questions = _reference_map(
        inputs.questions, (row['question_fixture_id'] for row in rows)
    )
    scopes = _reference_map(
        inputs.security_scopes, (row['security_scope_fixture_id'] for row in rows)
    )
    contexts = _reference_map(
        inputs.prior_contexts,
        (
            row['prior_context_fixture_id']
            for row in rows
            if row['prior_context_fixture_id'] is not None
        ),
    )
    refs = _reference_map(
        inputs.serving_references,
        (ref for row in rows for ref in row['relevant_serving_fixture_ids']),
    )
    members = {member.serving_identity_hmac: member for member in inputs.corpus.members}
    _live_require(
        len(set(refs.values())) == len(refs) and set(refs.values()) <= members.keys(),
        'fixture_mapping_invalid',
    )
    fixture_sha = hashlib.sha256(raw).hexdigest()
    source_hmac = _live_hmac(
        {
            'fixture_manifest_path': LIVE_FIXTURE_PATH,
            'fixture_manifest_sha256': fixture_sha,
            'fixture_manifest_version': LIVE_FIXTURE_VERSION,
        },
        'rag-live-fixture-manifest:v1',
        secret,
    )
    claims, cases = [], []
    effective_context_distribution = Counter()
    for row in rows:
        scope = scopes[row['security_scope_fixture_id']]
        _live_require(type(scope) is SecurityScope, 'scope_invalid')
        scope.__post_init__()
        question = questions[row['question_fixture_id']]
        _live_require(type(question) is str and bool(question.strip()), 'query_invalid')
        if row['surface'] == 'assistant':
            context = (
                ()
                if row['prior_context_fixture_id'] is None
                else contexts[row['prior_context_fixture_id']]
            )
            _live_require(
                type(context) is tuple
                and all(type(item) is AssistantContextMessage for item in context),
                'context_invalid',
            )
            prepared = prepare_assistant_request_text(question, context, key=secret)
            without_context = prepare_assistant_request_text(question, (), key=secret)
            has_context = (
                prepared.retrieval_query_text != without_context.retrieval_query_text
            )
            _live_require(
                has_context == (row['prior_context_fixture_id'] is not None),
                'context_distribution_invalid',
            )
            effective_context_distribution[
                f'{row["configured_backend"]}_{"with" if has_context else "without"}_prior_context'
            ] += 1
        else:
            prepared = prepare_direct_request_text(question, key=secret)
        case_id_hmac = _live_hmac(
            {
                'case_id_bytes': exact_utf8_bytes(row['case_id']),
                'case_ordinal': row['ordinal'],
                'fixture_manifest_hmac': source_hmac,
            },
            'rag-live-case-id:v1',
            secret,
        )
        relevant = tuple(refs[ref] for ref in row['relevant_serving_fixture_ids'])
        required = tuple(refs[ref] for ref in row['required_serving_fixture_ids'])
        for identity in relevant:
            member = members[identity]
            _live_require(
                member.effective_permission in scope.allowed_permission_levels
                and member.support_mode in row['allowed_support_modes'],
                'fixture_mapping_invalid',
            )
            if row['configured_backend'] == 'pgvector':
                _live_require(identity in pgvector_members, 'pgvector_member_invalid')
        query_cost, answer_cost, case_cost = (
            _live_money(row[name])
            for name in (
                'query_embedding_reserved_cost_usd',
                'answer_generation_reserved_cost_usd',
                'case_total_reserved_cost_usd',
            )
        )
        cases.append(
            FrozenLiveManifestCase(
                row['ordinal'],
                case_id_hmac,
                row['case_kind'],
                row['surface'],
                row['configured_backend'],
                row['question_fixture_id'],
                row['security_scope_fixture_id'],
                row['prior_context_fixture_id'],
                prepared.retrieval_query_hmac,
                row['expected_no_answer'],
                tuple(row['allowed_support_modes']),
                tuple(row['allowed_slot_ids']),
                relevant,
                required,
                row['query_embedding_required'],
                True,
                query_cost,
                answer_cost,
                case_cost,
            )
        )
        claims.append(
            FrozenCaseClaimCase(
                row['ordinal'],
                case_id_hmac,
                row['surface'],
                row['configured_backend'],
                prepared.current_text_hmac,
                prepared.retrieval_query_hmac,
                security_scope_fingerprint(
                    scope,
                    settings=SimpleNamespace(
                        agent_runtime_fingerprint_secret=secret.decode(),
                        agent_runtime_fingerprint_key_version=inputs.release.fingerprint_key_version,
                        paraworks_env='production',
                    ),
                ),
                (
                    FrozenCaseClaimComponent(
                        policies['query_embedding'],
                        row['query_embedding_reserved_input_tokens'],
                        0,
                        query_cost,
                    ),
                    FrozenCaseClaimComponent(
                        policies['answer_generation'],
                        row['answer_generation_reserved_input_tokens'],
                        row['answer_generation_reserved_output_tokens'],
                        answer_cost,
                    ),
                ),
            )
        )
    _live_require(
        effective_context_distribution == _CONTEXT_DISTRIBUTION,
        'context_distribution_invalid',
    )
    executable = FrozenCaseClaimManifest(
        tuple(claims), source_manifest_hmac=source_hmac
    )
    _manifest_payload(executable)
    return FrozenLiveManifestSnapshot(
        _LIVE_POLICY,
        LIVE_FIXTURE_VERSION,
        LIVE_FIXTURE_PATH,
        fixture_sha,
        source_hmac,
        source_hmac,
        commit,
        LIVE_RUBRIC_VERSION,
        tuple(cases),
        executable,
    )


def _baseline(manifest, corpus, files, secret):
    policy = {
        'annotation_schema_version': 'rag-live-relevance-annotation:v1',
        'evaluator_version': 'rag-live-retrieval-evaluator:v1',
        'keyword_scorer_version': 'rag-keyword-lexical-compat:v1',
        'pgvector_distance_policy_version': 'rag-pgvector-cosine-distance:v1',
        'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
        'rubric_version': LIVE_RUBRIC_VERSION,
    }
    source = {
        'file_path_bytes': exact_utf8_bytes(LIVE_EVALUATOR_PATH),
        'file_sha256': hashlib.sha256(files[LIVE_EVALUATOR_PATH]).hexdigest(),
        'source_commit': manifest.clean_git_commit,
    }
    bundle = {
        'source_commit': manifest.clean_git_commit,
        'files': [
            {
                'ordinal': ordinal,
                'file_path_bytes': exact_utf8_bytes(path),
                'file_sha256': hashlib.sha256(files[path]).hexdigest(),
            }
            for ordinal, path in enumerate(LIVE_RETRIEVER_SOURCE_PATHS)
        ],
    }
    return {
        'annotation_schema_version': policy['annotation_schema_version'],
        'cases': [
            {
                'case_id_hmac': case.case_id_hmac,
                'configured_backend': case.configured_backend,
                'legacy_retriever_version': f'rag-v1-{case.configured_backend}-retriever:v1',
                'query_bytes_hmac': case.query_bytes_hmac,
                'relevant_serving_identity_hmacs': list(
                    case.relevant_serving_identity_hmacs
                ),
                'v2_retriever_version': f'rag-v2-{case.configured_backend}-retriever:v1',
            }
            for case in manifest.cases
        ],
        'corpus_snapshot_hmac': corpus.corpus_snapshot_hmac,
        'evaluator_code_hmac': _live_hmac(source, 'rag-live-source-file:v1', secret),
        'evaluator_path': LIVE_EVALUATOR_PATH,
        'evaluator_source_commit': manifest.clean_git_commit,
        'evaluator_version': policy['evaluator_version'],
        'fixture_manifest_hmac': manifest.fixture_manifest_hmac,
        'keyword_scorer_version': policy['keyword_scorer_version'],
        'pgvector_distance_policy_version': policy['pgvector_distance_policy_version'],
        'policy_snapshot_hmac': _live_hmac(
            policy, 'rag-live-baseline-policy-snapshot:v1', secret
        ),
        'retriever_source_bundle_hmac': _live_hmac(
            bundle, 'rag-live-retriever-source-bundle:v1', secret
        ),
    }


def _hard_negative_oracles(manifest, inputs, reader, pgvector_members):
    read = getattr(reader, 'read_hard_negative_oracle', None)
    _live_require(callable(read), 'hard_negative_oracle_unavailable')
    members = {member.serving_identity_hmac: member for member in inputs.corpus.members}
    scopes = dict(inputs.security_scopes)
    records = []
    for case, executable in zip(manifest.cases, manifest.executable.cases, strict=True):
        if case.case_kind != 'hard_negative':
            continue
        request = HardNegativeOracleRequest(
            manifest.fixture_manifest_hmac,
            case.case_id_hmac,
            case.query_bytes_hmac,
            executable.security_scope_fingerprint,
            inputs.corpus.corpus_snapshot_hmac,
            case.configured_backend,
        )
        result = read(request)
        _live_require(
            type(result) is FrozenHardNegativeOracleResult
            and type(result.request) is HardNegativeOracleRequest,
            'hard_negative_oracle_invalid',
        )
        _live_require(
            all(
                type(getattr(result.request, field.name)) is str
                and getattr(result.request, field.name) == getattr(request, field.name)
                for field in fields(request)
            ),
            'hard_negative_oracle_invalid',
        )
        _live_digest(result.oracle_definition_hmac)
        _live_number(result.hidden_match_count)
        _live_require(
            type(result.visible_candidates) is tuple, 'hard_negative_oracle_invalid'
        )
        _live_require(
            bool(result.visible_candidates),
            'hard_negative_hidden_only'
            if result.hidden_match_count
            else 'hard_negative_no_match',
        )
        _live_require(
            len(result.visible_candidates) <= 8, 'hard_negative_oracle_invalid'
        )
        seen_slots, seen_members, candidates = set(), set(), []
        scope = scopes[case.security_scope_fixture_id]
        for candidate in result.visible_candidates:
            _live_require(
                type(candidate) is FrozenOracleVisibleCandidate,
                'hard_negative_oracle_invalid',
            )
            _live_require(
                type(candidate.slot_id) is str
                and candidate.slot_id in case.allowed_slot_ids
                and candidate.slot_id not in seen_slots
                and type(candidate.entailment) is str
                and candidate.entailment == 'not_entailed',
                'hard_negative_oracle_invalid',
            )
            identity = _live_digest(candidate.serving_identity_hmac)
            _live_require(
                identity in members and identity not in seen_members,
                'hard_negative_oracle_invalid',
            )
            member = members[identity]
            _live_require(
                member.effective_permission in scope.allowed_permission_levels
                and member.support_mode in case.allowed_support_modes,
                'hard_negative_oracle_invalid',
            )
            if case.configured_backend == 'pgvector':
                _live_require(identity in pgvector_members, 'pgvector_member_invalid')
            seen_slots.add(candidate.slot_id)
            seen_members.add(identity)
            candidates.append(asdict(candidate))
        records.append(
            {
                'request': asdict(request),
                'oracle_definition_hmac': result.oracle_definition_hmac,
                'visible_candidates': candidates,
                'hidden_match_count': result.hidden_match_count,
            }
        )
    return records


def _preview_from_inputs(files, commit, inputs, secret, reader):
    _live_require(
        type(inputs) is LiveGatePreviewInputs
        and type(inputs.release) is RagReleaseSnapshot,
        'snapshot_invalid',
    )
    release = inputs.release
    _live_require(type(release.ledger_uuid) is UUID, 'snapshot_invalid')
    _live_number(release.ledger_epoch, 1)
    _live_number(release.generation)
    _live_require(type(inputs.release_row_counts) is tuple, 'release_rows_invalid')
    expected_counts = (
        ('authorization', 0),
        ('case', 0),
        ('dispatch', 0),
        ('quality_report', 0),
        ('release_ledger', 1),
        ('release_transition', 0),
    )
    _live_require(
        len(inputs.release_row_counts) == len(expected_counts), 'release_rows_invalid'
    )
    for item, expected in zip(inputs.release_row_counts, expected_counts, strict=True):
        _live_require(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is int
            and item == expected,
            'release_rows_invalid',
        )
    _live_require(
        release.generation == 0 and release.last_transition_digest is None,
        'release_rows_invalid',
    )
    _live_require(
        release.marker_schema_version == 'rag-release-ledger-marker-body:v1',
        'snapshot_invalid',
    )
    for name in (
        'marker_file_digest',
        'fingerprint_key_material_verifier',
        'designated_environment_id_hmac',
        'designated_host_id_hmac',
        'validation_database_identity_hmac',
    ):
        _live_digest(getattr(release, name))
    for name in ('reviewer_roster_hmac', 'implementation_plan_reference_hmac'):
        _live_digest(getattr(inputs, name))
    corpus_payload = _corpus_payload(inputs.corpus)
    _live_require(
        _live_hmac(corpus_payload, 'rag-live-corpus-snapshot:v1', secret)
        == inputs.corpus.corpus_snapshot_hmac,
        'corpus_changed',
    )
    provider, policies = _provider_snapshot(inputs, secret)
    pgvector_members = _pgvector_baseline_members(reader, inputs.corpus)
    manifest = _build_manifest(
        files[LIVE_FIXTURE_PATH], commit, inputs, secret, policies, pgvector_members
    )
    manifest.limits.__post_init__()
    negative_oracles = _hard_negative_oracles(
        manifest, inputs, reader, pgvector_members
    )
    baseline = _baseline(manifest, inputs.corpus, files, secret)
    baseline_hmac = _live_hmac(baseline, 'rag-live-baseline-definition:v1', secret)
    limits = {
        'case_claims': 30,
        'answer_generation_dispatches': 30,
        'query_embedding_dispatches': 10,
        'total_dispatches': 40,
        'case_max_cost_usd': '0.012000',
        'total_max_cost_usd': '0.360000',
    }
    payload = {
        'live_gate_contract_version': _LIVE_POLICY,
        'fixture_manifest_version': LIVE_FIXTURE_VERSION,
        'fixture_manifest_path': LIVE_FIXTURE_PATH,
        'fixture_manifest_sha256': manifest.fixture_manifest_sha256,
        'fixture_manifest_hmac': manifest.fixture_manifest_hmac,
        'manifest_hmac': manifest.manifest_hmac,
        'clean_git_commit': commit,
        'rubric_version': LIVE_RUBRIC_VERSION,
        'distribution': _DISTRIBUTION,
        'assistant_context_distribution': _CONTEXT_DISTRIBUTION,
        'limits': limits,
        'total_reserved_cost_usd': '0.360000',
        'ledger_uuid': str(release.ledger_uuid),
        'ledger_epoch': release.ledger_epoch,
        'approval_base_generation': release.generation,
        'approval_base_release_marker_file_digest': release.marker_file_digest,
        'validation_database_identity_hmac': release.validation_database_identity_hmac,
        'designated_environment_id_hmac': release.designated_environment_id_hmac,
        'designated_host_id_hmac': release.designated_host_id_hmac,
        'implementation_plan_reference_hmac': inputs.implementation_plan_reference_hmac,
        'fingerprint_key_version': release.fingerprint_key_version,
        'fingerprint_key_material_verifier': release.fingerprint_key_material_verifier,
        'approved_corpus_snapshot_hmac': inputs.corpus.corpus_snapshot_hmac,
        'pgvector_baseline_serving_identity_hmacs': list(pgvector_members),
        'hard_negative_oracles': negative_oracles,
        'hard_negative_oracles_hmac': _live_hmac(
            negative_oracles, 'rag-live-hard-negative-oracles:v1', secret
        ),
        'approved_provider_safety_snapshot_hmac': inputs.approved_provider_safety_snapshot_hmac,
        'provider_authority_uuid': provider['authority_uuid'],
        'provider_safety_envelope_digest': provider['envelope_digest'],
        'provider_global_safety_generation': provider['global_safety_generation'],
        'active_families': provider['active_families'],
        'baseline_hmac': baseline_hmac,
        'reviewer_roster_hmac': inputs.reviewer_roster_hmac,
        'executable_preimage': _manifest_payload(manifest.executable),
    }
    canonical = canonical_json_bytes(payload)
    return LiveGatePreview(
        manifest,
        inputs.corpus,
        manifest.limits,
        Decimal('0.360000'),
        canonical_json_bytes(baseline),
        baseline_hmac,
        canonical,
        _live_hmac(payload, 'rag-live-preview:v1', secret),
    )


def _build_live_gate_preview(
    *,
    repository,
    expected_commit,
    snapshot_reader,
    identity_secret,
    expected_preview_hmac=None,
) -> LiveGatePreview:
    """Read, derive, re-read under caller-owned locks. No writer/model ports.

    The reader is the explicit approved-snapshot integration boundary. Task24-A
    has no production reader/composition until Task25's evaluator and Task24-B's
    verified reviewer inputs exist. A preview cannot authorize case claims.
    """
    _live_require(
        type(identity_secret) is bytes and len(identity_secret) >= 32, 'key_unavailable'
    )
    files = _source_snapshot(repository, expected_commit)
    _live_require(snapshot_reader is not None, 'snapshot_reader_unavailable')
    try:
        with snapshot_reader.locked() as reader:
            first = _preview_from_inputs(
                files, expected_commit, reader.read(), identity_secret, reader
            )
            second = _preview_from_inputs(
                files, expected_commit, reader.read(), identity_secret, reader
            )
            _live_require(
                first.canonical_bytes == second.canonical_bytes, 'snapshot_changed'
            )
            if expected_preview_hmac is not None:
                _live_require(
                    first.preview_hmac == _live_digest(expected_preview_hmac),
                    'preview_changed',
                )
            _live_require(
                _source_snapshot(repository, expected_commit) == files,
                'committed_source_changed',
            )
            return first
    except LiveGatePreviewError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError):
        raise LiveGatePreviewError('snapshot_invalid') from None


def _preview_source_boundary():
    issued = WeakKeyDictionary()

    class VerifiedSource:
        __slots__ = ('__weakref__',)

        def __init__(self):
            raise TypeError('preview sources are issued only by verified construction')

        def __copy__(self):
            raise TypeError('preview source bindings cannot be copied')

        def __deepcopy__(self, memo):
            raise TypeError('preview source bindings cannot be copied')

        def __reduce_ex__(self, protocol):
            raise TypeError('preview source bindings cannot be serialized')

        def __repr__(self):
            return '<VerifiedLiveManifestSource no execution authority>'

    def build(**kwargs):
        from dataclasses import replace

        result = _build_live_gate_preview(**kwargs)
        # The lock owner's exit can perform work too. Recheck after it and all
        # derivation, immediately before issuing the in-memory provenance token.
        _source_snapshot(kwargs['repository'], kwargs['expected_commit'])
        source = object.__new__(VerifiedSource)
        issued[source] = (
            result.canonical_bytes,
            result.preview_hmac,
            Path(kwargs['repository']).resolve(),
            kwargs['expected_commit'],
        )
        return replace(result, source_binding=source)

    def require(source, *, manifest, authorization, identity_secret):
        _live_require(
            type(source) is VerifiedSource and source in issued,
            'source_binding_unavailable',
        )
        canonical, digest, repository, commit = issued[source]
        value = json.loads(canonical)
        _live_require(
            _live_hmac(value, 'rag-live-preview:v1', identity_secret) == digest,
            'source_binding_changed',
        )
        try:
            candidate = _manifest_payload(manifest)
        except Exception:
            raise LiveGatePreviewError('source_binding_changed') from None
        _live_require(
            candidate == value['executable_preimage'], 'source_binding_changed'
        )
        _live_require(
            manifest.source_manifest_hmac == value['manifest_hmac'],
            'source_binding_changed',
        )
        _live_require(type(authorization) is dict, 'source_binding_changed')
        for name in (
            'ledger_uuid',
            'ledger_epoch',
            'manifest_hmac',
            'baseline_hmac',
            'reviewer_roster_hmac',
            'approved_corpus_snapshot_hmac',
            'approved_provider_safety_snapshot_hmac',
            'provider_safety_envelope_digest',
            'validation_database_identity_hmac',
        ):
            _live_require(
                type(authorization.get(name)) is type(value[name])
                and authorization[name] == value[name],
                'source_binding_changed',
            )
        current = _source_snapshot(repository, commit)
        _live_require(
            hashlib.sha256(current[LIVE_FIXTURE_PATH]).hexdigest()
            == value['fixture_manifest_sha256'],
            'source_binding_changed',
        )

    return VerifiedSource, build, require


(
    VerifiedLiveManifestSource,
    build_live_gate_preview,
    require_verified_preview_source,
) = _preview_source_boundary()


def _live_money(value):
    _live_require(
        type(value) is str and re.fullmatch(r'0\.[0-9]{6}', value) is not None
    )
    return Decimal(value)


def parse_live_fixture(raw: bytes) -> dict:
    """Validate the committed declarative roster before resolving any references."""
    from backend.app.agent_runtime.rag_cost_policy import (
        MAX_ANSWER_INPUT_TOKENS,
        MAX_ANSWER_OUTPUT_TOKENS,
        MAX_QUERY_EMBEDDING_TOKENS,
    )

    _live_require(type(raw) is bytes and 0 < len(raw) <= 128_000)
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_json_object)
    except (UnicodeError, ValueError, RecursionError):
        raise LiveGatePreviewError() from None
    _live_require(
        type(value) is dict and set(value) == {'fixture_manifest_version', 'cases'}
    )
    _live_require(value['fixture_manifest_version'] == LIVE_FIXTURE_VERSION)
    rows = value['cases']
    _live_require(type(rows) is list and len(rows) == 30)
    seen, distribution, context = set(), Counter(), Counter()
    total, labels = _ZERO, set()
    for ordinal, row in enumerate(rows):
        _live_require(type(row) is dict and set(row) == _CASE_KEYS)
        _live_require(type(row['ordinal']) is int and row['ordinal'] == ordinal)
        case_id = _fixture_id(row['case_id'])
        _live_require(case_id not in seen)
        seen.add(case_id)
        _live_require(
            row['surface'] in ('ask', 'assistant')
            and row['configured_backend'] in ('keyword', 'pgvector')
        )
        _live_require(row['case_kind'] in ('positive', 'hard_negative'))
        labels.add(row['case_kind'])
        negative = row['case_kind'] == 'hard_negative'
        _live_require(
            type(row['expected_no_answer']) is bool
            and row['expected_no_answer'] == negative
        )
        for name in ('question_fixture_id', 'security_scope_fixture_id'):
            _fixture_id(row[name])
        prior = row['prior_context_fixture_id']
        if prior is not None:
            _fixture_id(prior)
            _live_require(row['surface'] == 'assistant')
        backend = row['configured_backend']
        distribution[f'{row["surface"]}_{backend}'] += 1
        if row['surface'] == 'assistant':
            context[
                f'{backend}_{"with" if prior is not None else "without"}_prior_context'
            ] += 1
        for name in (
            'allowed_support_modes',
            'allowed_slot_ids',
            'relevant_serving_fixture_ids',
            'required_serving_fixture_ids',
        ):
            items = row[name]
            _live_require(
                type(items) is list and all(type(item) is str for item in items)
            )
            _live_require(len(items) == len(set(items)))
        _live_require(
            set(row['allowed_support_modes']) <= {'trusted_fact', 'source_observation'}
        )
        _live_require(set(row['allowed_slot_ids']) <= {f'E{i}' for i in range(1, 9)})
        _live_require(
            bool(row['allowed_support_modes']) and bool(row['allowed_slot_ids'])
        )
        for name in ('relevant_serving_fixture_ids', 'required_serving_fixture_ids'):
            for item in row[name]:
                _fixture_id(item)
        _live_require(
            set(row['required_serving_fixture_ids'])
            <= set(row['relevant_serving_fixture_ids'])
        )
        if not negative:
            _live_require(
                all(
                    row[name]
                    for name in (
                        'allowed_support_modes',
                        'allowed_slot_ids',
                        'relevant_serving_fixture_ids',
                        'required_serving_fixture_ids',
                    )
                )
            )
        query = backend == 'pgvector'
        _live_require(
            type(row['query_embedding_required']) is bool
            and row['query_embedding_required'] == query
        )
        _live_require(row['answer_generation_required'] is True)
        for name, expected in (
            (
                'query_embedding_reserved_input_tokens',
                MAX_QUERY_EMBEDDING_TOKENS if query else 0,
            ),
            ('answer_generation_reserved_input_tokens', MAX_ANSWER_INPUT_TOKENS),
            ('answer_generation_reserved_output_tokens', MAX_ANSWER_OUTPUT_TOKENS),
        ):
            _live_require(type(row[name]) is int and row[name] == expected)
        embedding, generation, ceiling = (
            _live_money(row[name])
            for name in (
                'query_embedding_reserved_cost_usd',
                'answer_generation_reserved_cost_usd',
                'case_total_reserved_cost_usd',
            )
        )
        # Full caps use the existing frozen runtime token maxima. Decimal bounds
        # are checked before serialization; no rounding can erase a shortfall.
        _live_require(embedding == (Decimal('0.000160') if query else _ZERO))
        _live_require(generation >= Decimal('0.009804'))
        _live_require(embedding + generation == ceiling <= Decimal('0.012000'))
        total += ceiling
    _live_require(distribution == _DISTRIBUTION and context == _CONTEXT_DISTRIBUTION)
    _live_require(
        labels == {'positive', 'hard_negative'} and total == Decimal('0.360000')
    )
    return value


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
    """Resolved execution preimage, never independent authorization.

    The full quality manifest retains this complete declaratively derived image.
    Its source identity binds the committed fixture; its own digest serves only
    integrity checking. Task24-B must verify the separately approved runtime
    inputs before any use in a case claim.
    """

    cases: tuple[FrozenCaseClaimCase, ...]
    contract_version: str = _VERSION
    fixture_manifest_version: str = 'rag-live-quality-30:v1'
    runtime_contract_version: str = 'rag-run:v2'
    assembly_version: str = _ASSEMBLY
    source_manifest_hmac: str | None = None


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
    if manifest.source_manifest_hmac is not None:
        try:
            require_lower_hmac(manifest.source_manifest_hmac)
        except ValueError:
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
    """Integrity evidence only. Never the authorization's manifest authority."""
    return rag_identity_hmac(
        _manifest_payload(manifest),
        secret=identity_secret,
        schema_version=_VERSION,
        policy_version='rag-live-gate:v1',
    )


def require_approved_case_source(
    connection, *, manifest, source_binding, authorization, identity_secret
):
    """Task24-B's approved-runtime verifier is deliberately not composed yet.

    A verified preview proves provenance, not human execution approval. B must
    revalidate its source binding and the separately locked approved runtime
    snapshots here before returning. There is no self-signed-preimage fallback.
    """
    _refuse()


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

    def prepare(
        connection: Connection,
        *,
        manifest,
        payload,
        identity_secret,
        source_binding=None,
    ):
        from backend.app.rag.release_ledger import (
            RagReleaseMutationSet,
            ReleaseRowPrimaryKey,
        )

        binding = _claim_binding(payload)
        manifest_integrity_hmac = case_claim_manifest_hmac(
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
        if auth is None:
            _refuse()
        require_approved_case_source(
            connection,
            manifest=manifest,
            source_binding=source_binding,
            authorization=auth,
            identity_secret=identity_secret,
        )
        manifest_hmac = manifest.source_manifest_hmac
        if manifest_hmac is None or auth['manifest_hmac'] != manifest_hmac:
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
            'manifest_integrity_hmac': manifest_integrity_hmac,
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
            != state['manifest_integrity_hmac']
            or state['manifest'].source_manifest_hmac != state['manifest_hmac']
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
