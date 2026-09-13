from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from backend.app.rag import release_review as review

FIXTURE_PATH = 'backend/tests/fixtures/rag_v2_live_gate_30.json'
SECRET = b'task24-test-only-fingerprint-material'


def fingerprint(value, schema, policy='rag-live-gate:v1', secret=SECRET):
    # Independent stdlib oracle; no production identity/canonical helpers.
    raw = json.dumps(
        {
            'domain': 'paraworks:keyed-fingerprint:v1',
            'policy_version': policy,
            'schema_version': schema,
            'value': value,
        },
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    ).encode()
    return hmac.new(secret, raw, hashlib.sha256).hexdigest()


def utf8(text):
    raw = text.encode()
    return {'byte_length': len(raw), 'utf8_hex': raw.hex()}


def corpus():
    return review.freeze_live_corpus(
        corpus_generation=7,
        vector_index_generation=11,
        embedding_model_bytes=b'text-embedding-3-small',
        index_policy_version_bytes=b'rag-v2-serving-index:v1',
        pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
        members=tuple(
            review.FrozenCorpusMember(
                ordinal=i,
                serving_identity_hmac=str(i + 1) * 64,
                serving_version_fingerprint='3' * 64,
                model_content_hmac='4' * 64,
                canonical_citation_projection_hmac='5' * 64,
                effective_permission='internal',
                support_mode='trusted_fact' if i == 0 else 'source_observation',
                vector_index_state_hmac='6' * 64,
            )
            for i in range(2)
        ),
        identity_secret=SECRET,
    )


def test_frozen_corpus_has_exact_independent_payload_identity():
    value = corpus()
    expected = {
        'corpus_generation': 7,
        'vector_index_generation': 11,
        'embedding_model_bytes': utf8('text-embedding-3-small'),
        'index_policy_version_bytes': utf8('rag-v2-serving-index:v1'),
        'pgvector_cosine_policy_version': 'pgvector-cosine-indexable:v1',
        'members': [
            {
                'ordinal': i,
                'serving_identity_hmac': str(i + 1) * 64,
                'serving_version_fingerprint': '3' * 64,
                'model_content_hmac': '4' * 64,
                'canonical_citation_projection_hmac': '5' * 64,
                'effective_permission': 'internal',
                'support_mode': 'trusted_fact' if i == 0 else 'source_observation',
                'vector_index_state_hmac': '6' * 64,
            }
            for i in range(2)
        ],
    }
    assert value.corpus_snapshot_hmac == fingerprint(
        expected, 'rag-live-corpus-snapshot:v1'
    )


@pytest.mark.parametrize(
    'change',
    [
        'bool_generation',
        'order',
        'duplicate',
        'permission',
        'support',
        'hmac',
        'vector',
        'model',
        'policy',
    ],
)
def test_invalid_corpus_refuses(change):
    original = corpus()
    arguments = {
        name: getattr(original, name)
        for name in (
            'corpus_generation',
            'vector_index_generation',
            'embedding_model_bytes',
            'index_policy_version_bytes',
            'pgvector_cosine_policy_version',
            'members',
        )
    }
    if change == 'bool_generation':
        arguments['corpus_generation'] = True
    elif change == 'order':
        arguments['members'] = original.members[::-1]
    elif change == 'duplicate':
        arguments['members'] = (
            original.members[0],
            replace(original.members[0], ordinal=1),
        )
    elif change == 'model':
        arguments['embedding_model_bytes'] = b'unapproved-model'
    elif change == 'policy':
        arguments['pgvector_cosine_policy_version'] = 'unknown:v2'
    else:
        name, val = {
            'permission': ('effective_permission', 'unknown'),
            'support': ('support_mode', 'llm_fact'),
            'hmac': ('model_content_hmac', 'A' * 64),
            'vector': ('vector_index_state_hmac', None),
        }[change]
        arguments['members'] = (
            replace(original.members[0], **{name: val}),
            original.members[1],
        )
    with pytest.raises(review.LiveGatePreviewError):
        review.freeze_live_corpus(**arguments, identity_secret=SECRET)


class SnapshotReader:
    def __init__(self, value, after=None):
        self.value, self.after, self.reads, self.locked_now = value, after, 0, False

    @contextmanager
    def locked(self):
        assert not self.locked_now
        self.locked_now = True
        try:
            yield self
        finally:
            self.locked_now = False

    def read(self):
        assert self.locked_now
        self.reads += 1
        return self.after if self.reads > 1 and self.after is not None else self.value


def preview_inputs():
    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
    from backend.app.agent_runtime.rag_v2_identity import SecurityScope
    from backend.app.agents.rag_orchestrator_agent.v2_input import (
        AssistantContextMessage,
    )
    from backend.app.rag.release_authority import RagReleaseSnapshot

    key_verifier = fingerprint_key_material_verifier(SECRET.decode())
    provider = {
        'authority_uuid': '33333333-3333-3333-3333-333333333333',
        'designated_environment_id': 'isolated-validation',
        'envelope_digest': '8' * 64,
        'fingerprint_key_material_verifier': key_verifier,
        'fingerprint_key_version': 'v1',
        'global_safety_generation': 0,
        'active_families': [],
    }
    for component in ('answer_generation', 'query_embedding'):
        query = component == 'query_embedding'
        provider['active_families'].append(
            {
                'component': component,
                'provider': 'openai',
                'model': 'text-embedding-3-small'
                if query
                else 'gpt-5.4-mini-2026-03-17',
                'reasoning_or_config_identity': 'dimensions:1536' if query else 'none',
                'authorized_model_config_version': 'rag-query-embedding-config:v1'
                if query
                else 'rag-answer-model-config:v1',
                'authorized_model_config_snapshot_hmac': '1' * 64,
                'authorized_cost_policy_version': 'rag-query-embedding-cost:v1'
                if query
                else 'rag-answer-cost:v1',
                'authorized_token_estimator_version': 'openai-cl100k-text-embedding-3-small:v1'
                if query
                else 'openai-o200k-rag-answer:v1',
                'authorized_fingerprint_key_version': 'v1',
                'authorized_policy_snapshot_hmac': '2' * 64,
                'state': 'ready',
                'state_version': 1,
                'family_safety_generation': 0,
            }
        )
    scope = SecurityScope(
        'rag-security-scope:v1',
        'fixture-user',
        'fixture-workspace',
        'all_current_scope',
        (),
        (),
        ('public', 'internal'),
        'demo-auth:v1',
        'rag-permission-policy:v1',
    )
    return review.LiveGatePreviewInputs(
        corpus=corpus(),
        questions=tuple(
            (f'question-{i:02}', f'검증용 일정 {i}의 근거는 무엇인가요?')
            for i in range(1, 31)
        ),
        security_scopes=(('scope-internal', scope),),
        prior_contexts=tuple(
            (
                f'context-{i:02}',
                (
                    AssistantContextMessage(
                        i,
                        datetime(2026, 9, 1, tzinfo=UTC),
                        'user',
                        '검증용 일정의 이전 논의',
                    ),
                ),
            )
            for i in (16, 17, 18, 19, 20, 26, 27, 28)
        ),
        serving_references=(('serving-01', '1' * 64), ('serving-02', '2' * 64)),
        provider_snapshot_bytes=encoded(provider),
        release=RagReleaseSnapshot(
            'rag-release-ledger-marker-body:v1',
            UUID('11111111-1111-1111-1111-111111111111'),
            1,
            0,
            None,
            None,
            None,
            'v1',
            key_verifier,
            '7' * 64,
            '8' * 64,
            fingerprint(
                {'designated_environment_id_bytes': utf8('isolated-validation')},
                'rag-live-designated-environment-id:v1',
            ),
            'a' * 64,
            'b' * 64,
            'c' * 64,
            'd' * 64,
            'e' * 64,
            'release-ledger-init',
        ),
        approved_provider_safety_snapshot_hmac=fingerprint(
            provider, 'rag-provider-safety-approved-snapshot:v1'
        ),
        reviewer_roster_hmac='f' * 64,
        implementation_plan_reference_hmac='0' * 64,
        release_row_counts=(
            ('authorization', 0),
            ('case', 0),
            ('dispatch', 0),
            ('quality_report', 0),
            ('release_ledger', 1),
            ('release_transition', 0),
        ),
    )


def committed_repo(tmp_path, *, missing_evaluator=False):
    repo = tmp_path / 'isolated-git'
    repo.mkdir()
    source_paths = review.LIVE_RETRIEVER_SOURCE_PATHS + (review.LIVE_EVALUATOR_PATH,)
    for name in source_paths:
        if missing_evaluator and name == review.LIVE_EVALUATOR_PATH:
            continue
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f'# isolated provider-free source: {name}\n'.encode())
    fixture = repo / FIXTURE_PATH
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(encoded(declaration()))
    for args in (
        ['init', '-q'],
        ['add', '.'],
        [
            '-c',
            'user.name=Task24 Test',
            '-c',
            'user.email=task24@example.invalid',
            'commit',
            '-qm',
            'isolated fixture',
        ],
    ):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    commit = (
        subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'])
        .decode()
        .strip()
    )
    return repo, commit


def test_preview_binds_complete_manifest_and_baseline_without_calls_or_mutations(
    tmp_path, monkeypatch
):
    import socket

    import httpx
    from sqlalchemy import Engine, create_engine, event, text
    from sqlalchemy.pool import StaticPool

    repo, commit = committed_repo(tmp_path)
    reader = SnapshotReader(preview_inputs())
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('provider or network call')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    from langchain_core.language_models import BaseChatModel

    from backend.app.rag.release_authority import _RagReleaseBarrierGuard
    from backend.app.rag.release_ledger import RagReleaseLedger

    monkeypatch.setattr(BaseChatModel, 'invoke', forbidden)
    monkeypatch.setattr(RagReleaseLedger, 'append', forbidden)
    monkeypatch.setattr(_RagReleaseBarrierGuard, 'apply_provider_incident', forbidden)
    engine = create_engine(
        'sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False}
    )
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE mutation_probe (value INTEGER)'))
        connection.execute(text('INSERT INTO mutation_probe VALUES (41)'))
    sql = []

    def observe_sql(*args):
        sql.append(args[2])

    event.listen(Engine, 'before_cursor_execute', observe_sql)
    try:
        with engine.connect() as connection:
            before = connection.execute(text('SELECT value FROM mutation_probe')).all()
            result = review.build_live_gate_preview(
                repository=repo,
                expected_commit=commit,
                snapshot_reader=reader,
                identity_secret=SECRET,
            )
            after = connection.execute(text('SELECT value FROM mutation_probe')).all()
    finally:
        event.remove(Engine, 'before_cursor_execute', observe_sql)
        engine.dispose()
    assert before == after == [(41,)]
    assert not any(
        s.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')) for s in sql
    )
    assert calls == [] and reader.reads == 2
    assert result.limits.case_claims == 30
    assert result.limits.answer_generation_dispatches == 30
    assert result.limits.query_embedding_dispatches == 10
    assert result.limits.total_dispatches == 40
    assert result.limits.case_max_cost_usd == Decimal('0.012000')
    assert result.limits.total_max_cost_usd == Decimal('0.360000')
    assert result.total_reserved_cost_usd == Decimal('0.360000')
    fixture_sha = hashlib.sha256((repo / FIXTURE_PATH).read_bytes()).hexdigest()
    root = fingerprint(
        {
            'fixture_manifest_path': FIXTURE_PATH,
            'fixture_manifest_sha256': fixture_sha,
            'fixture_manifest_version': 'rag-live-quality-30:v1',
        },
        'rag-live-fixture-manifest:v1',
    )
    assert (
        result.manifest.manifest_hmac == result.manifest.fixture_manifest_hmac == root
    )
    assert result.manifest.executable.source_manifest_hmac == root
    assert len(result.manifest.executable.cases) == 30
    for case, executable in zip(
        result.manifest.cases, result.manifest.executable.cases, strict=True
    ):
        assert case.case_id_hmac == executable.case_id_hmac
        assert case.query_bytes_hmac == executable.retrieval_query_hmac
        assert case.ordinal == executable.ordinal
        question = f'검증용 일정 {case.ordinal + 1}의 근거는 무엇인가요?'
        query = (
            question
            if case.surface == 'ask'
            else (
                (
                    '최근 대화:\nuser: 검증용 일정의 이전 논의\n'
                    if case.prior_context_fixture_id
                    else ''
                )
                + f'현재 질문: {question}'
            )
        )
        assert case.query_bytes_hmac == fingerprint(
            utf8(query),
            'rag-retrieval-query-bytes:v1',
            'direct-query:v1' if case.surface == 'ask' else 'assistant-context:v1',
        )
        assert case.case_id_hmac == fingerprint(
            {
                'case_id_bytes': utf8(f'live-{case.ordinal + 1:02}'),
                'case_ordinal': case.ordinal,
                'fixture_manifest_hmac': root,
            },
            'rag-live-case-id:v1',
        )
    baseline = json.loads(result.baseline_definition_bytes)
    assert baseline['evaluator_source_commit'] == commit
    assert baseline['fixture_manifest_hmac'] == root
    assert len(baseline['cases']) == 30
    assert result.baseline_hmac == fingerprint(
        baseline, 'rag-live-baseline-definition:v1'
    )
    assert result.preview_hmac == fingerprint(
        json.loads(result.canonical_bytes), 'rag-live-preview:v1'
    )
    shown = result.sanitized_payload()
    assert '검증용' not in json.dumps(shown, ensure_ascii=False)
    assert SECRET.decode() not in repr(result)
    assert shown['provider_dispatch_count'] == 0
    assert shown['authorization_issued'] is False


@pytest.mark.parametrize(
    'change',
    [
        'fixture',
        'commit',
        'dirty',
        'source',
        'evaluator',
        'corpus',
        'corpus_permission',
        'provider',
        'key',
        'release_epoch',
        'release_rows',
        'scope',
        'question',
        'context',
        'mapping_missing',
        'mapping_extra',
        'mapping_duplicate',
        'reference_alias',
        'preview_hmac',
        'reviewer',
        'plan',
    ],
)
def test_preview_drift_and_invalid_inputs_are_zero_call_refusals(
    tmp_path, monkeypatch, change
):
    import socket

    import httpx

    repo, commit = committed_repo(tmp_path)
    initial = preview_inputs()
    first = review.build_live_gate_preview(
        repository=repo,
        expected_commit=commit,
        snapshot_reader=SnapshotReader(initial),
        identity_secret=SECRET,
    )
    updated, secret, expected = initial, SECRET, first.preview_hmac
    if change == 'fixture':
        raw = declaration()
        raw['cases'][0]['question_fixture_id'] = 'question-other'
        (repo / FIXTURE_PATH).write_bytes(encoded(raw))
    elif change == 'commit':
        subprocess.run(
            [
                'git',
                '-C',
                str(repo),
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@example.invalid',
                'commit',
                '--allow-empty',
                '-qm',
                'changed',
            ],
            check=True,
            capture_output=True,
        )
    elif change == 'dirty':
        (repo / 'untracked').write_text('changed')
    elif change == 'source':
        (repo / review.LIVE_RETRIEVER_SOURCE_PATHS[0]).write_text('# changed')
    elif change == 'evaluator':
        (repo / review.LIVE_EVALUATOR_PATH).unlink()
    elif change == 'corpus':
        updated = replace(initial, corpus=replace(initial.corpus, corpus_generation=8))
    elif change == 'corpus_permission':
        updated = replace(
            initial,
            corpus=replace(
                initial.corpus,
                members=(
                    replace(initial.corpus.members[0], effective_permission='public'),
                    initial.corpus.members[1],
                ),
            ),
        )
    elif change == 'provider':
        provider = json.loads(initial.provider_snapshot_bytes)
        provider['active_families'][0]['state'] = 'blocked_overrun'
        updated = replace(initial, provider_snapshot_bytes=encoded(provider))
    elif change == 'key':
        secret = b'a-different-test-fingerprint-material'
    elif change == 'release_epoch':
        updated = replace(initial, release=replace(initial.release, ledger_epoch=2))
    elif change == 'release_rows':
        updated = replace(initial, release_row_counts=(('authorization', 1),))
    elif change == 'scope':
        updated = replace(
            initial,
            security_scopes=(
                (
                    'scope-internal',
                    replace(
                        initial.security_scopes[0][1], principal_subject='other-user'
                    ),
                ),
            ),
        )
    elif change == 'question':
        updated = replace(
            initial, questions=(('question-01', '변경된 질문'), *initial.questions[1:])
        )
    elif change == 'context':
        updated = replace(
            initial,
            prior_contexts=tuple((name, ()) for name, _ in initial.prior_contexts),
        )
    elif change == 'mapping_missing':
        updated = replace(initial, questions=initial.questions[1:])
    elif change == 'mapping_extra':
        updated = replace(initial, questions=(*initial.questions, ('unused', 'extra')))
    elif change == 'mapping_duplicate':
        updated = replace(initial, questions=(*initial.questions, initial.questions[0]))
    elif change == 'reference_alias':
        updated = replace(
            initial,
            serving_references=(('serving-01', '1' * 64), ('serving-02', '1' * 64)),
        )
    elif change == 'preview_hmac':
        expected = '0' * 64
    elif change == 'reviewer':
        updated = replace(initial, reviewer_roster_hmac='1' * 64)
    else:
        updated = replace(initial, implementation_plan_reference_hmac='1' * 64)
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError('provider call forbidden')

    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    with pytest.raises(review.LiveGatePreviewError):
        review.build_live_gate_preview(
            repository=repo,
            expected_commit=commit,
            snapshot_reader=SnapshotReader(updated),
            identity_secret=secret,
            expected_preview_hmac=expected,
        )
    assert calls == []


def test_preview_rechecks_locked_snapshot_and_refuses_midread_drift(tmp_path):
    repo, commit = committed_repo(tmp_path)
    first = preview_inputs()
    reader = SnapshotReader(first, replace(first, reviewer_roster_hmac='1' * 64))
    with pytest.raises(review.LiveGatePreviewError, match='snapshot_changed'):
        review.build_live_gate_preview(
            repository=repo,
            expected_commit=commit,
            snapshot_reader=reader,
            identity_secret=SECRET,
        )
    assert reader.reads == 2 and not reader.locked_now


def authorization_for(preview):
    payload = json.loads(preview.canonical_bytes)
    return {
        key: payload[key]
        for key in (
            'ledger_uuid',
            'ledger_epoch',
            'manifest_hmac',
            'baseline_hmac',
            'reviewer_roster_hmac',
            'approved_corpus_snapshot_hmac',
            'approved_provider_safety_snapshot_hmac',
            'provider_safety_envelope_digest',
            'validation_database_identity_hmac',
        )
    }


def test_verified_source_retains_full_preimage_and_refuses_independent_self_attestation(
    tmp_path,
):
    repo, commit = committed_repo(tmp_path)
    value = review.build_live_gate_preview(
        repository=repo,
        expected_commit=commit,
        snapshot_reader=SnapshotReader(preview_inputs()),
        identity_secret=SECRET,
    )
    manifest = value.manifest.executable
    authority = authorization_for(value)
    review.require_verified_preview_source(
        value.source_binding,
        manifest=manifest,
        authorization=authority,
        identity_secret=SECRET,
    )
    assert (
        review.case_claim_manifest_hmac(manifest, identity_secret=SECRET)
        != authority['manifest_hmac']
    )
    changes = [
        replace(manifest, source_manifest_hmac='a' * 64),
        replace(manifest, cases=manifest.cases[::-1]),
        replace(
            manifest,
            cases=(
                replace(manifest.cases[0], security_scope_fingerprint='a' * 64),
                *manifest.cases[1:],
            ),
        ),
        replace(
            manifest,
            cases=(
                replace(
                    manifest.cases[0],
                    components=(
                        manifest.cases[0].components[0],
                        replace(
                            manifest.cases[0].components[1], reserved_input_tokens=9999
                        ),
                    ),
                ),
                *manifest.cases[1:],
            ),
        ),
    ]
    for changed in changes:
        with pytest.raises(review.LiveGatePreviewError):
            review.require_verified_preview_source(
                value.source_binding,
                manifest=changed,
                authorization=authority,
                identity_secret=SECRET,
            )
    for field in authority:
        changed = {**authority, field: 2 if field == 'ledger_epoch' else 'a' * 64}
        with pytest.raises(review.LiveGatePreviewError):
            review.require_verified_preview_source(
                value.source_binding,
                manifest=manifest,
                authorization=changed,
                identity_secret=SECRET,
            )


@pytest.mark.parametrize(
    'kind', ['missing', 'forged', 'unissued', 'copy', 'key', 'different_preview']
)
def test_unissued_copied_or_cross_preview_source_never_becomes_authority(
    tmp_path, kind
):
    from copy import copy

    repo, commit = committed_repo(tmp_path)
    value = review.build_live_gate_preview(
        repository=repo,
        expected_commit=commit,
        snapshot_reader=SnapshotReader(preview_inputs()),
        identity_secret=SECRET,
    )
    source, key = value.source_binding, SECRET
    if kind == 'missing':
        source = None
    elif kind == 'forged':
        source = object()
    elif kind == 'unissued':
        source = object.__new__(type(source))
    elif kind == 'copy':
        with pytest.raises(TypeError):
            copy(source)
        return
    elif kind == 'key':
        key = b'other-test-only-fingerprint-material'
    else:
        inputs = preview_inputs()
        other = review.build_live_gate_preview(
            repository=repo,
            expected_commit=commit,
            snapshot_reader=SnapshotReader(
                replace(
                    inputs,
                    questions=(('question-01', '다른 질문'), *inputs.questions[1:]),
                )
            ),
            identity_secret=SECRET,
        )
        source = other.source_binding
    with pytest.raises(review.LiveGatePreviewError):
        review.require_verified_preview_source(
            source,
            manifest=value.manifest.executable,
            authorization=authorization_for(value),
            identity_secret=key,
        )


def test_missing_evaluator_is_a_clear_provider_free_readiness_refusal(tmp_path):
    repo, commit = committed_repo(tmp_path, missing_evaluator=True)
    with pytest.raises(review.LiveGatePreviewError, match='evaluator_unavailable'):
        review.build_live_gate_preview(
            repository=repo,
            expected_commit=commit,
            snapshot_reader=None,
            identity_secret=SECRET,
        )


@pytest.mark.parametrize('change', ['provider_environment', 'duplicate_provider_key'])
def test_authenticated_provider_snapshot_cannot_alias_a_different_environment_or_json(
    tmp_path, change
):
    repo, commit = committed_repo(tmp_path)
    inputs = preview_inputs()
    provider = json.loads(inputs.provider_snapshot_bytes)
    if change == 'provider_environment':
        provider['designated_environment_id'] = 'another-validation-environment'
        raw = encoded(provider)
    else:
        raw = encoded(provider).replace(b'{', b'{"global_safety_generation":999,', 1)
    changed = replace(
        inputs,
        provider_snapshot_bytes=raw,
        approved_provider_safety_snapshot_hmac=fingerprint(
            provider, 'rag-provider-safety-approved-snapshot:v1'
        ),
    )
    with pytest.raises(review.LiveGatePreviewError):
        review.build_live_gate_preview(
            repository=repo,
            expected_commit=commit,
            snapshot_reader=SnapshotReader(changed),
            identity_secret=SECRET,
        )


def test_baseline_binds_the_actual_legacy_retriever_and_scorer_source(tmp_path):
    repo, commit = committed_repo(tmp_path)
    preview = review.build_live_gate_preview(
        repository=repo,
        expected_commit=commit,
        snapshot_reader=SnapshotReader(preview_inputs()),
        identity_secret=SECRET,
    )
    paths = (
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
    bundle = {
        'source_commit': commit,
        'files': [
            {
                'ordinal': ordinal,
                'file_path_bytes': utf8(path),
                'file_sha256': hashlib.sha256(
                    f'# isolated provider-free source: {path}\n'.encode()
                ).hexdigest(),
            }
            for ordinal, path in enumerate(paths)
        ],
    }
    baseline = json.loads(preview.baseline_definition_bytes)
    assert baseline['retriever_source_bundle_hmac'] == fingerprint(
        bundle, 'rag-live-retriever-source-bundle:v1'
    )


def test_full_thirty_case_source_selects_exact_sql_images_and_never_inner_digest(
    tmp_path, monkeypatch
):
    from sqlalchemy import create_engine, event, insert

    from backend.app.models.agent_runs import AgentRun
    from backend.app.models.rag_runtime import AgentRunCostComponent
    from backend.app.rag.release_schema import (
        build_rag_release_metadata,
        release_tables,
    )

    repo, commit = committed_repo(tmp_path)
    preview = review.build_live_gate_preview(
        repository=repo,
        expected_commit=commit,
        snapshot_reader=SnapshotReader(preview_inputs()),
        identity_secret=SECRET,
    )
    engine = create_engine('sqlite://')
    metadata = build_rag_release_metadata()
    table = release_tables(metadata).authorizations
    with engine.begin() as connection:
        table.create(connection)
        AgentRun.__table__.create(connection)
        AgentRunCostComponent.__table__.create(connection)
        for ordinal in range(30):
            connection.execute(
                insert(table).values(
                    **authorization_for(preview),
                    approval_id_hmac=f'{ordinal + 1:064x}',
                    approval_hmac=f'{ordinal + 100:064x}',
                    base_generation=0,
                    state='unused' if ordinal == 0 else 'started',
                    case_claim_count=ordinal,
                    reserved_cost_usd=Decimal('0.012000') * ordinal,
                    execution_process_instance_hmac=None if ordinal == 0 else 'd' * 64,
                    execution_runner_fence_hmac=None if ordinal == 0 else 'e' * 64,
                )
            )

    def fake_approved_source(
        connection, *, manifest, source_binding, authorization, identity_secret
    ):
        # The only fake is B's approval proof; real source/projection validation
        # and SQL SELECTs run. The fixture predetermines every allowed auth row.
        assert connection.engine is engine
        assert (
            authorization['approval_hmac']
            == f'{authorization["case_claim_count"] + 100:064x}'
        )
        review.require_verified_preview_source(
            source_binding,
            manifest=manifest,
            authorization=authorization,
            identity_secret=identity_secret,
        )

    monkeypatch.setattr(review, 'require_approved_case_source', fake_approved_source)
    writes = []

    def observe_sql(_conn, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            writes.append(statement)

    event.listen(engine, 'before_cursor_execute', observe_sql)
    try:
        with engine.connect() as connection:
            for ordinal, case in enumerate(preview.manifest.cases):
                payload = {
                    key: authorization_for(preview)[key]
                    for key in (
                        'ledger_uuid',
                        'ledger_epoch',
                        'approved_corpus_snapshot_hmac',
                        'approved_provider_safety_snapshot_hmac',
                        'provider_safety_envelope_digest',
                        'validation_database_identity_hmac',
                    )
                }
                payload.update(
                    approval_id_hmac=f'{ordinal + 1:064x}',
                    approval_hmac=f'{ordinal + 100:064x}',
                    from_generation=ordinal + 1,
                    execution_process_instance_hmac='d' * 64,
                    execution_runner_fence_hmac='e' * 64,
                    case_id_hmac=case.case_id_hmac,
                )
                source = preview.manifest.executable
                projection = review.prepare_case_claim_projection(
                    connection,
                    manifest=source,
                    source_binding=preview.source_binding,
                    payload=payload,
                    identity_secret=SECRET,
                )
                parent, query, answer = projection.runtime_images
                assert parent['permission_level'] == 'restricted'
                assert (
                    parent['metadata']['configured_backend'] == case.configured_backend
                )
                assert query['component'] == 'query_embedding'
                assert answer['component'] == 'answer_generation'
                assert query['reserved_cost_usd'] == (
                    Decimal('0.000160')
                    if case.query_embedding_required
                    else Decimal('0.000000')
                )
                assert answer['reserved_cost_usd'] + query[
                    'reserved_cost_usd'
                ] == Decimal('0.012000')
                assert (
                    answer['reserved_input_tokens'] == 10000
                    and answer['reserved_output_tokens'] == 512
                )
                assert (
                    parent['metadata']['retrieval_query_hmac'] == case.query_bytes_hmac
                )
                changed_case = replace(
                    source.cases[ordinal], security_scope_fingerprint='f' * 64
                )
                altered = replace(
                    source,
                    cases=source.cases[:ordinal]
                    + (changed_case,)
                    + source.cases[ordinal + 1 :],
                )
                with pytest.raises(review.LiveGatePreviewError):
                    review.prepare_case_claim_projection(
                        connection,
                        manifest=altered,
                        source_binding=preview.source_binding,
                        payload=payload,
                        identity_secret=SECRET,
                    )
        assert writes == []
    finally:
        event.remove(engine, 'before_cursor_execute', observe_sql)
        engine.dispose()


def declaration():
    """Test-owned declarative cases; no runtime authority or private records."""
    rows = []
    for ordinal in range(30):
        surface, backend, prior = (
            ('ask', 'keyword', False)
            if ordinal < 10
            else ('ask', 'pgvector', False)
            if ordinal < 15
            else ('assistant', 'keyword', ordinal < 20)
            if ordinal < 25
            else ('assistant', 'pgvector', ordinal < 28)
        )
        negative = ordinal % 5 == 4
        rows.append(
            {
                'ordinal': ordinal,
                'case_id': f'live-{ordinal + 1:02}',
                'case_kind': 'hard_negative' if negative else 'positive',
                'surface': surface,
                'configured_backend': backend,
                'question_fixture_id': f'question-{ordinal + 1:02}',
                'security_scope_fixture_id': 'scope-internal',
                'prior_context_fixture_id': f'context-{ordinal + 1:02}'
                if prior
                else None,
                'expected_no_answer': negative,
                'allowed_support_modes': []
                if negative
                else ['trusted_fact', 'source_observation'],
                'allowed_slot_ids': [] if negative else ['E1', 'E2'],
                'relevant_serving_fixture_ids': []
                if negative
                else ['serving-01', 'serving-02'],
                'required_serving_fixture_ids': [] if negative else ['serving-01'],
                'query_embedding_required': backend == 'pgvector',
                'answer_generation_required': True,
                'query_embedding_reserved_input_tokens': 8000
                if backend == 'pgvector'
                else 0,
                'answer_generation_reserved_input_tokens': 10000,
                'answer_generation_reserved_output_tokens': 512,
                'query_embedding_reserved_cost_usd': '0.000160'
                if backend == 'pgvector'
                else '0.000000',
                'answer_generation_reserved_cost_usd': '0.011840'
                if backend == 'pgvector'
                else '0.012000',
                'case_total_reserved_cost_usd': '0.012000',
            }
        )
    return {'fixture_manifest_version': 'rag-live-quality-30:v1', 'cases': rows}


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode()


def test_live_fixture_enforces_exact_roster_distribution_and_reserves():
    result = review.parse_live_fixture(encoded(declaration()))
    assert len(result['cases']) == 30
    assert result['cases'][25]['prior_context_fixture_id'] == 'context-26'


@pytest.mark.parametrize(
    'field,value',
    [
        ('case_claims', 31),
        ('case_claims', True),
        ('answer_generation_dispatches', 29),
        ('query_embedding_dispatches', 11),
        ('total_dispatches', 41),
        ('case_max_cost_usd', Decimal('0.012001')),
        ('case_max_cost_usd', Decimal('0.0120001')),
        ('total_max_cost_usd', Decimal('0.360001')),
        ('total_max_cost_usd', 0.36),
    ],
)
def test_live_limits_refuse_expansion_shortfall_or_type_alias(field, value):
    with pytest.raises(review.LiveGatePreviewError):
        review.LiveGateLimits(**{field: value})


@pytest.mark.parametrize(
    'change',
    [
        'version',
        'missing',
        'extra',
        'duplicate',
        'order',
        'bool_ordinal',
        'distribution',
        'prior_distribution',
        'ask_prior',
        'unsafe_reference',
        'unknown_label',
        'expected_no_answer',
        'required_not_relevant',
        'duplicate_relevant',
        'unknown_support',
        'duplicate_slot',
        'invalid_slot',
        'missing_generation',
        'missing_embedding',
        'nonzero_keyword_embedding',
        'split',
        'overage',
        'subprecision',
        'shortfall',
        'negative',
        'nan',
        'numeric_money',
        'extra_field',
        'reserve_token_shortfall',
    ],
)
def test_invalid_live_fixture_refuses_before_resolution(change):
    value = declaration()
    row = value['cases'][0]
    if change == 'version':
        value['fixture_manifest_version'] = 'rag-live-quality-30:v2'
    elif change == 'missing':
        value['cases'].pop()
    elif change == 'extra':
        value['cases'].append(deepcopy(row))
    elif change == 'duplicate':
        value['cases'][1]['case_id'] = row['case_id']
    elif change == 'order':
        value['cases'][:2] = value['cases'][:2][::-1]
    elif change == 'bool_ordinal':
        row['ordinal'] = False
    elif change == 'distribution':
        row['surface'] = 'assistant'
    elif change == 'prior_distribution':
        value['cases'][15]['prior_context_fixture_id'] = None
    elif change == 'ask_prior':
        row['prior_context_fixture_id'] = 'context-01'
    elif change == 'unsafe_reference':
        row['question_fixture_id'] = '../secret.env'
    elif change == 'unknown_label':
        row['case_kind'] = 'maybe'
    elif change == 'expected_no_answer':
        row['expected_no_answer'] = True
    elif change == 'required_not_relevant':
        row['required_serving_fixture_ids'] = ['serving-03']
    elif change == 'duplicate_relevant':
        row['relevant_serving_fixture_ids'] *= 2
    elif change == 'unknown_support':
        row['allowed_support_modes'] = ['unreviewed']
    elif change == 'duplicate_slot':
        row['allowed_slot_ids'] = ['E1', 'E1']
    elif change == 'invalid_slot':
        row['allowed_slot_ids'] = ['E9']
    elif change == 'missing_generation':
        row['answer_generation_required'] = False
    elif change == 'missing_embedding':
        value['cases'][10]['query_embedding_required'] = False
    elif change == 'nonzero_keyword_embedding':
        row['query_embedding_reserved_cost_usd'] = '0.000001'
    elif change == 'split':
        row['answer_generation_reserved_cost_usd'] = '0.011999'
    elif change == 'overage':
        row['case_total_reserved_cost_usd'] = '0.012001'
    elif change == 'subprecision':
        row['answer_generation_reserved_cost_usd'] = '0.0120001'
    elif change == 'shortfall':
        row['answer_generation_reserved_cost_usd'] = '0.000001'
        row['case_total_reserved_cost_usd'] = '0.000001'
    elif change == 'negative':
        row['answer_generation_reserved_cost_usd'] = '-0.012000'
    elif change == 'nan':
        row['answer_generation_reserved_cost_usd'] = 'NaN'
    elif change == 'numeric_money':
        row['answer_generation_reserved_cost_usd'] = 0.012
    elif change == 'extra_field':
        row['raw_query'] = 'untrusted secret'
    else:
        row['answer_generation_reserved_input_tokens'] = 9999
    with pytest.raises(review.LiveGatePreviewError):
        review.parse_live_fixture(encoded(value))


def test_committed_fixture_is_the_sanitized_exact_thirty_case_roster():
    fixture = Path(__file__).parents[2] / FIXTURE_PATH
    assert fixture.is_file()
    actual = review.parse_live_fixture(fixture.read_bytes())
    assert [row['case_id'] for row in actual['cases']] == [
        f'live-{i:02}' for i in range(1, 31)
    ]
    assert {row['case_kind'] for row in actual['cases']} == {
        'positive',
        'hard_negative',
    }
