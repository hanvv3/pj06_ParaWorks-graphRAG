"""Oracle adapters cannot rewrite the manifest binding through shared aliases."""

import hashlib
import json
from dataclasses import replace

import pytest

from backend.app.rag import release_review as review
from backend.tests.test_rag_live_gate_preview import (
    FIXTURE_PATH,
    SnapshotReader,
    committed_repo,
    fingerprint,
    preview_inputs,
    utf8,
)
from backend.tests.test_rag_live_gate_preview_r1 import build
from backend.tests.test_rag_live_gate_preview_r1 import side_effects as side_effects

FIELDS = (
    'fixture_manifest_hmac',
    'case_id_hmac',
    'query_bytes_hmac',
    'security_scope_fingerprint',
    'corpus_snapshot_hmac',
    'configured_backend',
)


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize(
    'mode', ['alias', 'copy', 'restored_bad_copy', 'valid_copy_changed_input', 'raise']
)
def test_adapter_cannot_rewrite_request_binding(tmp_path, side_effects, field, mode):
    repo, commit = committed_repo(tmp_path)

    class Reader(SnapshotReader):
        def read_hard_negative_oracle(self, request):
            result = super().read_hard_negative_oracle(request)
            valid_copy = replace(request)
            original = getattr(request, field)
            changed = (
                ('pgvector' if original == 'keyword' else 'keyword')
                if field == 'configured_backend'
                else 'a' * 64
            )
            object.__setattr__(request, field, changed)
            if mode == 'raise':
                raise ValueError('adapter rejected its mutated input')
            if mode == 'alias':
                return result
            if mode == 'valid_copy_changed_input':
                return replace(result, request=valid_copy)
            copied = replace(request)
            if mode == 'restored_bad_copy':
                object.__setattr__(request, field, original)
            return replace(result, request=copied)

    with pytest.raises(review.LiveGatePreviewError):
        build(repo, commit, Reader(preview_inputs()))
    assert side_effects['issued'] == []


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('container', ['list', 'dict'])
def test_adapter_cannot_replace_scalar_binding_with_mutable_container(
    tmp_path, side_effects, field, container
):
    repo, commit = committed_repo(tmp_path)

    class Reader(SnapshotReader):
        def read_hard_negative_oracle(self, request):
            result = super().read_hard_negative_oracle(request)
            nested = (
                [getattr(request, field)]
                if container == 'list'
                else {'value': getattr(request, field)}
            )
            object.__setattr__(request, field, nested)
            if container == 'list':
                nested[0] = 'tampered'
            else:
                nested['value'] = 'tampered'
            return result

    with pytest.raises(review.LiveGatePreviewError):
        build(repo, commit, Reader(preview_inputs()))
    assert side_effects['issued'] == []


@pytest.mark.parametrize('copied_result', [False, True])
def test_restored_exact_bindings_cannot_change_preview_identity(
    tmp_path, side_effects, copied_result
):
    repo, commit = committed_repo(tmp_path)
    initial = preview_inputs()
    original = build(repo, commit, SnapshotReader(initial))

    class Reader(SnapshotReader):
        def read_hard_negative_oracle(self, request):
            result = super().read_hard_negative_oracle(request)
            for field in FIELDS:
                original_value = getattr(request, field)
                object.__setattr__(request, field, 'temporary mutation')
                object.__setattr__(request, field, original_value)
            return (
                replace(result, request=replace(request)) if copied_result else result
            )

    restored = build(repo, commit, Reader(initial))
    assert restored.canonical_bytes == original.canonical_bytes
    assert restored.preview_hmac == original.preview_hmac
    payload = json.loads(restored.canonical_bytes)
    first = payload['hard_negative_oracles'][0]['request']
    # Exact independent bindings: derive from committed bytes and literal case/query,
    # never the adapter's request or the produced manifest's query value.
    root = fingerprint(
        {
            'fixture_manifest_path': FIXTURE_PATH,
            'fixture_manifest_sha256': hashlib.sha256(
                (repo / FIXTURE_PATH).read_bytes()
            ).hexdigest(),
            'fixture_manifest_version': 'rag-live-quality-30:v1',
        },
        'rag-live-fixture-manifest:v1',
    )
    assert first['fixture_manifest_hmac'] == root
    assert first['case_id_hmac'] == fingerprint(
        {
            'case_id_bytes': utf8('live-05'),
            'case_ordinal': 4,
            'fixture_manifest_hmac': root,
        },
        'rag-live-case-id:v1',
    )
    assert first['query_bytes_hmac'] == fingerprint(
        utf8('검증용 일정 5의 근거는 무엇인가요?'),
        'rag-retrieval-query-bytes:v1',
        'direct-query:v1',
    )
    assert first['security_scope_fingerprint'] == fingerprint(
        {
            'contract_version': 'rag-security-scope:v1',
            'principal_subject_bytes': utf8('fixture-user'),
            'workspace_scope_id_bytes': utf8('fixture-workspace'),
            'resource_scope_mode': 'all_current_scope',
            'project_constraint_bytes': [],
            'source_constraint_bytes': [],
            'allowed_permission_levels': ['public', 'internal'],
            'auth_policy_version_bytes': utf8('demo-auth:v1'),
            'permission_policy_version_bytes': utf8('rag-permission-policy:v1'),
        },
        'rag-security-scope-fingerprint:v1',
        'rag-security-scope:v1',
    )
    assert first['corpus_snapshot_hmac'] == initial.corpus.corpus_snapshot_hmac
    assert first['configured_backend'] == 'keyword'
    assert payload['hard_negative_oracles_hmac'] == fingerprint(
        payload['hard_negative_oracles'],
        'rag-live-hard-negative-oracles:v1',
    )
    assert restored.preview_hmac == fingerprint(payload, 'rag-live-preview:v1')
