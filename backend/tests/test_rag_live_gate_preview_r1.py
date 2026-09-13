"""Independent regression probes for the four Task24-A review findings."""

import json
import socket
import subprocess
from dataclasses import replace
from weakref import WeakKeyDictionary

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, event

from backend.app.rag import release_review as review
from backend.app.rag.release_authority import _RagReleaseBarrierGuard
from backend.app.rag.release_ledger import RagReleaseLedger
from backend.tests.test_rag_live_gate_preview import (
    FIXTURE_PATH,
    SECRET,
    SnapshotReader,
    committed_repo,
    declaration,
    encoded,
    fingerprint,
    preview_inputs,
)


@pytest.fixture
def side_effects(monkeypatch):
    observed = {'provider': [], 'mutation': [], 'issued': []}

    def forbidden(*args, **kwargs):
        observed['provider'].append(True)
        raise AssertionError('unexpected provider or authority mutation')

    for owner, name in (
        (httpx.Client, 'send'),
        (httpx.AsyncClient, 'send'),
        (socket.socket, 'connect'),
        (BaseChatModel, 'invoke'),
        (RagReleaseLedger, 'append'),
        (_RagReleaseBarrierGuard, 'apply_provider_incident'),
    ):
        monkeypatch.setattr(owner, name, forbidden)

    def sql(*args):
        if args[2].lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            observed['mutation'].append(args[2])

    original = WeakKeyDictionary.__setitem__

    def issue(mapping, key, value):
        if type(key) is review.VerifiedLiveManifestSource:
            observed['issued'].append(True)
        return original(mapping, key, value)

    monkeypatch.setattr(WeakKeyDictionary, '__setitem__', issue)
    event.listen(Engine, 'before_cursor_execute', sql)
    try:
        yield observed
    finally:
        event.remove(Engine, 'before_cursor_execute', sql)
        assert observed['provider'] == []
        assert observed['mutation'] == []


def build(repo, commit, reader, **kwargs):
    return review.build_live_gate_preview(
        repository=repo,
        expected_commit=commit,
        snapshot_reader=reader,
        identity_secret=SECRET,
        **kwargs,
    )


def commit_declaration(repo, value):
    (repo / FIXTURE_PATH).write_bytes(encoded(value))
    for args in (
        ['add', FIXTURE_PATH],
        [
            '-c',
            'user.name=Test',
            '-c',
            'user.email=test@example.invalid',
            'commit',
            '-qm',
            'updated synthetic declaration',
        ],
    ):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    return (
        subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'])
        .decode()
        .strip()
    )


@pytest.mark.parametrize('kind', ['missing', 'no_match', 'hidden_only'])
def test_negative_oracle_is_required_before_provenance(tmp_path, side_effects, kind):
    repo, commit = committed_repo(tmp_path)

    class Reader(SnapshotReader):
        def read_hard_negative_oracle(self, request):
            raise review.LiveGatePreviewError(f'hard_negative_{kind}')

    reader = Reader(preview_inputs())
    if kind == 'missing':
        reader.read_hard_negative_oracle = None
    with pytest.raises(review.LiveGatePreviewError):
        build(repo, commit, reader)
    assert side_effects['issued'] == []


@pytest.mark.parametrize('kind', ['blank', 'unsafe', 'current_duplicate', 'truncated'])
def test_declared_prior_context_must_survive_preparation(tmp_path, side_effects, kind):
    repo, commit = committed_repo(tmp_path)
    initial = preview_inputs()
    question = dict(initial.questions)['question-16']
    content = {
        'blank': ' \t\n ',
        'unsafe': 'OPENAI_API_KEY=abcdEFGH1234ijklMNOP',
        'current_duplicate': question,
        'truncated': '이전 논의',
    }[kind]
    context = replace(initial.prior_contexts[0][1][0], content=content)
    initial = replace(
        initial,
        prior_contexts=(
            (initial.prior_contexts[0][0], (context,)),
            *initial.prior_contexts[1:],
        ),
    )
    if kind == 'truncated':
        initial = replace(
            initial,
            questions=tuple(
                (name, '가' * 7993 if name == 'question-16' else value)
                for name, value in initial.questions
            ),
        )
    with pytest.raises(
        review.LiveGatePreviewError, match='context_distribution_invalid'
    ):
        build(repo, commit, SnapshotReader(initial))
    assert side_effects['issued'] == []


@pytest.mark.parametrize('phase', ['final_read', 'unlock'])
@pytest.mark.parametrize('kind', ['source', 'fixture', 'commit'])
def test_final_reader_cannot_issue_stale_git_provenance(
    tmp_path, side_effects, kind, phase
):
    from contextlib import contextmanager

    repo, commit = committed_repo(tmp_path)

    def mutate():
        if kind == 'commit':
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
        else:
            path = FIXTURE_PATH if kind == 'fixture' else review.LIVE_EVALUATOR_PATH
            (repo / path).write_bytes((repo / path).read_bytes() + b'\n')

    class Reader(SnapshotReader):
        def read(self):
            value = super().read()
            if self.reads == 2 and phase == 'final_read':
                mutate()
            return value

        @contextmanager
        def locked(self):
            with super().locked() as reader:
                yield reader
            if phase == 'unlock':
                mutate()

    with pytest.raises(review.LiveGatePreviewError):
        build(repo, commit, Reader(preview_inputs()))
    assert side_effects['issued'] == []


def refreeze(value, members):
    return review.freeze_live_corpus(
        corpus_generation=value.corpus_generation,
        vector_index_generation=value.vector_index_generation,
        embedding_model_bytes=value.embedding_model_bytes,
        index_policy_version_bytes=value.index_policy_version_bytes,
        pgvector_cosine_policy_version=value.pgvector_cosine_policy_version,
        members=members,
        identity_secret=SECRET,
    )


def test_keyword_only_corpus_member_allows_null_vector(tmp_path, side_effects):
    repo, commit = committed_repo(tmp_path)
    initial = preview_inputs()
    extra = replace(
        initial.corpus.members[0],
        ordinal=2,
        serving_identity_hmac='3' * 64,
        vector_index_state_hmac=None,
    )
    value = refreeze(initial.corpus, (*initial.corpus.members, extra))
    fixture = declaration()
    fixture['cases'][0]['relevant_serving_fixture_ids'].append('serving-03')
    commit = commit_declaration(repo, fixture)
    initial = replace(
        initial,
        corpus=value,
        serving_references=(*initial.serving_references, ('serving-03', '3' * 64)),
    )
    result = build(repo, commit, SnapshotReader(initial))
    assert result.corpus.members[2].vector_index_state_hmac is None
    assert '3' * 64 in result.manifest.cases[0].relevant_serving_identity_hmacs
    assert json.loads(result.canonical_bytes)[
        'pgvector_baseline_serving_identity_hmacs'
    ] == ['1' * 64, '2' * 64]
    assert result.source_binding is not None


@pytest.mark.parametrize('member_index', [0, 1])
def test_pgvector_relevant_or_required_member_requires_vector(
    tmp_path, side_effects, member_index
):
    repo, commit = committed_repo(tmp_path)
    initial = preview_inputs()
    members = tuple(
        replace(member, vector_index_state_hmac=None) if i == member_index else member
        for i, member in enumerate(initial.corpus.members)
    )
    value = refreeze(initial.corpus, members)
    with pytest.raises(review.LiveGatePreviewError, match='pgvector_member_invalid'):
        build(repo, commit, SnapshotReader(replace(initial, corpus=value)))
    assert side_effects['issued'] == []


@pytest.mark.parametrize(
    'kind', ['missing', 'empty', 'duplicate', 'unmapped', 'omits_relevant']
)
def test_pgvector_baseline_roster_must_cover_actual_members(
    tmp_path, side_effects, kind
):
    repo, commit = committed_repo(tmp_path)
    reader = SnapshotReader(preview_inputs())
    values = {
        'empty': (),
        'duplicate': ('1' * 64, '1' * 64),
        'unmapped': ('a' * 64,),
        'omits_relevant': ('1' * 64,),
    }
    reader.read_pgvector_baseline_members = (
        None if kind == 'missing' else lambda: values[kind]
    )
    with pytest.raises(review.LiveGatePreviewError, match='pgvector_member_invalid'):
        build(repo, commit, reader)
    assert side_effects['issued'] == []


@pytest.mark.parametrize(
    'kind',
    [
        'no_match',
        'hidden_only',
        'entailing',
        'ambiguous',
        'unknown_member',
        'disallowed_slot',
        'duplicate_slot',
        'duplicate_member',
        'scope',
        'case',
        'source',
        'query',
        'corpus',
        'definition',
        'bool_result',
        'unindexed_pgvector',
    ],
)
def test_reader_negative_candidates_are_validated_before_issuance(
    tmp_path, side_effects, kind
):
    repo, commit = committed_repo(tmp_path)

    class Reader(SnapshotReader):
        def read_hard_negative_oracle(self, request):
            result = super().read_hard_negative_oracle(request)
            candidate = result.visible_candidates[0]
            if kind in ('no_match', 'hidden_only'):
                return replace(
                    result,
                    visible_candidates=(),
                    hidden_match_count=int(kind == 'hidden_only'),
                )
            if kind in ('entailing', 'ambiguous'):
                return replace(
                    result,
                    visible_candidates=(
                        replace(
                            candidate,
                            entailment='entailed' if kind == 'entailing' else kind,
                        ),
                    ),
                )
            if kind == 'unknown_member':
                return replace(
                    result,
                    visible_candidates=(
                        replace(candidate, serving_identity_hmac='a' * 64),
                    ),
                )
            if kind == 'disallowed_slot':
                return replace(
                    result, visible_candidates=(replace(candidate, slot_id='E8'),)
                )
            if kind == 'duplicate_slot':
                return replace(result, visible_candidates=(candidate, candidate))
            if kind == 'duplicate_member':
                return replace(
                    result,
                    visible_candidates=(candidate, replace(candidate, slot_id='E2')),
                )
            field = {
                'scope': 'security_scope_fingerprint',
                'case': 'case_id_hmac',
                'source': 'fixture_manifest_hmac',
                'query': 'query_bytes_hmac',
                'corpus': 'corpus_snapshot_hmac',
            }.get(kind)
            if field:
                return replace(result, request=replace(request, **{field: 'a' * 64}))
            if kind == 'definition':
                return replace(result, oracle_definition_hmac='not-a-digest')
            if kind == 'bool_result':
                return True
            if request.configured_backend == 'pgvector':
                return replace(
                    result,
                    visible_candidates=(
                        replace(candidate, serving_identity_hmac='3' * 64),
                    ),
                )
            return result

    initial = preview_inputs()
    if kind == 'unindexed_pgvector':
        extra = replace(
            initial.corpus.members[0],
            ordinal=2,
            serving_identity_hmac='3' * 64,
            vector_index_state_hmac=None,
        )
        initial = replace(
            initial, corpus=refreeze(initial.corpus, (*initial.corpus.members, extra))
        )
    with pytest.raises(review.LiveGatePreviewError):
        build(repo, commit, Reader(initial))
    assert side_effects['issued'] == []


def test_negative_oracle_request_and_candidate_result_are_bound_in_preview(
    tmp_path, side_effects
):
    repo, commit = committed_repo(tmp_path)
    result = build(repo, commit, SnapshotReader(preview_inputs()))
    payload = json.loads(result.canonical_bytes)
    records = payload['hard_negative_oracles']
    assert len(records) == 6
    first = records[0]
    assert first['request'] == {
        'fixture_manifest_hmac': result.manifest.fixture_manifest_hmac,
        'case_id_hmac': result.manifest.cases[4].case_id_hmac,
        'query_bytes_hmac': result.manifest.cases[4].query_bytes_hmac,
        'security_scope_fingerprint': result.manifest.executable.cases[
            4
        ].security_scope_fingerprint,
        'corpus_snapshot_hmac': result.corpus.corpus_snapshot_hmac,
        'configured_backend': 'keyword',
    }
    assert first['visible_candidates'] == [
        {
            'slot_id': 'E1',
            'serving_identity_hmac': '1' * 64,
            'entailment': 'not_entailed',
        },
        {
            'slot_id': 'E2',
            'serving_identity_hmac': '2' * 64,
            'entailment': 'not_entailed',
        },
    ]
    assert payload['hard_negative_oracles_hmac'] == fingerprint(
        records, 'rag-live-hard-negative-oracles:v1'
    )
    assert result.preview_hmac == fingerprint(payload, 'rag-live-preview:v1')


@pytest.mark.parametrize('field', ['allowed_support_modes', 'allowed_slot_ids'])
def test_negative_declaration_must_allow_visible_evidence(field, side_effects):
    value = declaration()
    value['cases'][4][field] = []
    with pytest.raises(review.LiveGatePreviewError):
        review.parse_live_fixture(encoded(value))
    assert side_effects['issued'] == []


def test_hard_negative_scope_cannot_be_hidden_only(tmp_path, side_effects):
    repo, _ = committed_repo(tmp_path)
    fixture = declaration()
    fixture['cases'][4]['security_scope_fixture_id'] = 'scope-hidden'
    commit = commit_declaration(repo, fixture)
    initial = preview_inputs()
    hidden_scope = replace(
        initial.security_scopes[0][1], allowed_permission_levels=('public',)
    )
    initial = replace(
        initial,
        security_scopes=(*initial.security_scopes, ('scope-hidden', hidden_scope)),
    )
    with pytest.raises(review.LiveGatePreviewError, match='hard_negative_hidden_only'):
        build(repo, commit, SnapshotReader(initial))
    assert side_effects['issued'] == []


def test_nonrelevant_pgvector_participant_also_requires_vector(tmp_path, side_effects):
    repo, commit = committed_repo(tmp_path)
    initial = preview_inputs()
    extra = replace(
        initial.corpus.members[0],
        ordinal=2,
        serving_identity_hmac='3' * 64,
        vector_index_state_hmac=None,
    )
    initial = replace(
        initial, corpus=refreeze(initial.corpus, (*initial.corpus.members, extra))
    )
    reader = SnapshotReader(initial)
    reader.read_pgvector_baseline_members = lambda: ('1' * 64, '2' * 64, '3' * 64)
    with pytest.raises(review.LiveGatePreviewError, match='pgvector_member_invalid'):
        build(repo, commit, reader)
    assert side_effects['issued'] == []


@pytest.mark.parametrize('change', ['definition', 'candidates', 'participation'])
def test_oracle_or_participation_drift_refuses_prior_preview(
    tmp_path, side_effects, change
):
    repo, commit = committed_repo(tmp_path)
    initial = preview_inputs()
    extra = replace(
        initial.corpus.members[0], ordinal=2, serving_identity_hmac='3' * 64
    )
    initial = replace(
        initial, corpus=refreeze(initial.corpus, (*initial.corpus.members, extra))
    )
    original = build(repo, commit, SnapshotReader(initial))
    side_effects['issued'].clear()

    class Reader(SnapshotReader):
        def read_hard_negative_oracle(self, request):
            result = super().read_hard_negative_oracle(request)
            if change == 'definition':
                return replace(result, oracle_definition_hmac='a' * 64)
            if change == 'candidates':
                return replace(result, visible_candidates=result.visible_candidates[:1])
            return result

        def read_pgvector_baseline_members(self):
            if change == 'participation':
                return ('1' * 64, '2' * 64, '3' * 64)
            return super().read_pgvector_baseline_members()

    with pytest.raises(review.LiveGatePreviewError, match='preview_changed'):
        build(
            repo, commit, Reader(initial), expected_preview_hmac=original.preview_hmac
        )
    assert side_effects['issued'] == []


def test_oracle_second_read_drift_is_refused_before_issuance(tmp_path, side_effects):
    repo, commit = committed_repo(tmp_path)

    class Reader(SnapshotReader):
        def read_hard_negative_oracle(self, request):
            result = super().read_hard_negative_oracle(request)
            return (
                replace(result, oracle_definition_hmac='a' * 64)
                if self.reads == 2
                else result
            )

    with pytest.raises(review.LiveGatePreviewError, match='snapshot_changed'):
        build(repo, commit, Reader(preview_inputs()))
    assert side_effects['issued'] == []
