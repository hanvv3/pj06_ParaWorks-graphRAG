from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
from copy import copy, deepcopy
from dataclasses import replace
from uuid import UUID

import pytest
from sqlalchemy import select, update

from backend.app.rag import release_review as review
from backend.app.rag.release_quality import (
    FrozenLegacyBaselineMetrics,
    RagReleaseQualityError,
    RagReleaseQualityEvaluator,
    SignedReviewLabel,
    build_review_signature_hmac,
)
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables
from backend.tests.test_rag_live_gate_authorization import runtime_source_harness
from backend.tests.test_rag_live_gate_preview import committed_repo
from backend.tests.test_rag_release_ledger import (
    _deterministic_non_product_database_seam,  # noqa: F401
)
from backend.tests.test_rag_release_quality import _terminal_cases

try:
    from backend.app.rag.live_gate import (
        ApprovedRagExecutionCapability,
        RagLiveGateCapabilityError,
        issue_approved_execution_capability,
    )
except ModuleNotFoundError:
    ApprovedRagExecutionCapability = None
    RagLiveGateCapabilityError = ValueError
    issue_approved_execution_capability = None


def _git(repo, *arguments: str) -> bytes:
    return subprocess.check_output(
        ['git', '-C', str(repo), *arguments],
        stderr=subprocess.STDOUT,
    )


def test_clean_crlf_checkout_uses_committed_blob_identity(
    tmp_path, monkeypatch
) -> None:
    """Dropping committed-blob reads would make a clean Windows checkout fail."""

    repo = tmp_path / 'clean-crlf-checkout'
    repo.mkdir()
    commit = 'a' * 40
    source_paths = (
        review.LIVE_EVALUATOR_PATH,
        review.LIVE_FIXTURE_PATH,
        *review.LIVE_RETRIEVER_SOURCE_PATHS,
    )
    committed = {}
    object_ids = {}
    for ordinal, path in enumerate(source_paths, 1):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        committed[path] = f'# committed LF blob: {path}\n'.encode()
        object_ids[path] = f'{ordinal:040x}'
        target.write_bytes(committed[path].replace(b'\n', b'\r\n'))

    calls = []

    def fake_git_read(repository, *arguments, input_bytes=None):
        calls.append((repository, arguments))
        if arguments == ('rev-parse', '--show-toplevel'):
            return f'{repo.resolve()}\n'.encode()
        if arguments == ('rev-parse', 'HEAD'):
            return (commit + '\n').encode()
        if arguments == ('status', '--porcelain=v1', '--untracked-files=all'):
            return b''
        if arguments[0] == 'rev-parse':
            selected = []
            for reference in arguments[1:]:
                selected_commit, path = reference.split(':', 1)
                assert selected_commit == commit
                selected.append(object_ids[path])
            return ('\n'.join(selected) + '\n').encode()
        if arguments[:2] == ('cat-file', '--batch-check'):
            selected = input_bytes.decode().splitlines()
            return ''.join(f'{item} blob 1\n' for item in selected).encode()
        if arguments[:3] == ('ls-files', '-v', '--'):
            return ''.join(f'H {path}\n' for path in sorted(arguments[3:])).encode()
        if arguments[:3] == ('ls-files', '--stage', '--'):
            return ''.join(
                f'100644 {object_ids[path]} 0\t{path}\n'
                for path in sorted(arguments[3:])
            ).encode()
        if arguments[0] == 'check-attr':
            selected_paths = arguments[5:]
            return ''.join(
                f'{path}: {name}: unspecified\n'
                for path in selected_paths
                for name in ('filter', 'working-tree-encoding', 'ident')
            ).encode()
        if arguments[:2] == ('hash-object', '--stdin-paths'):
            return ''.join(
                f'{object_ids[path]}\n' for path in input_bytes.decode().splitlines()
            ).encode()
        if arguments[0] == 'show':
            selected_commit, path = arguments[1].split(':', 1)
            assert selected_commit == commit
            return committed[path]
        raise AssertionError(arguments)

    monkeypatch.setattr(review, '_git_read', fake_git_read)

    assert review.require_live_preview_sources(repo) == commit
    assert any(arguments[0] == 'show' for _, arguments in calls)


def test_true_dirty_source_still_refuses_before_blob_approval(tmp_path) -> None:
    """Removing the Git-clean fence would authorize a real source edit."""

    repo, _commit = committed_repo(tmp_path)
    target = repo / review.LIVE_RETRIEVER_SOURCE_PATHS[0]
    target.write_bytes(target.read_bytes() + b'# true dirty change\n')

    assert _git(repo, 'status', '--porcelain=v1', '--untracked-files=all')
    with pytest.raises(review.LiveGatePreviewError, match='worktree_dirty'):
        review.require_live_preview_sources(repo)


@pytest.mark.parametrize('index_flag', ['--assume-unchanged', '--skip-worktree'])
def test_hidden_index_flag_cannot_mask_dirty_bound_source(tmp_path, index_flag) -> None:
    """Index optimization flags cannot hide bytes executed by the live gate."""

    repo, _commit = committed_repo(tmp_path)
    path = review.LIVE_RETRIEVER_SOURCE_PATHS[0]
    clear_flag = (
        '--no-assume-unchanged'
        if index_flag == '--assume-unchanged'
        else '--no-skip-worktree'
    )
    try:
        _git(repo, 'update-index', index_flag, path)
        target = repo / path
        target.write_bytes(target.read_bytes() + b'# hidden dirty execution change\n')
        assert _git(repo, 'status', '--porcelain=v1', '--untracked-files=all') == b''

        with pytest.raises(review.LiveGatePreviewError, match='worktree_dirty'):
            review.require_live_preview_sources(repo)
    finally:
        _git(repo, 'update-index', clear_flag, path)


@pytest.mark.parametrize('injection', ['work_tree', 'index', 'config'])
def test_git_environment_cannot_redirect_bound_source_checks(
    tmp_path, monkeypatch, injection
) -> None:
    """Inherited Git routing/configuration cannot select a benign alternate tree."""

    repo, _commit = committed_repo(tmp_path)
    alternate = tmp_path / 'alternate-clean-tree'
    shutil.copytree(repo, alternate, ignore=shutil.ignore_patterns('.git'))
    target = repo / review.LIVE_EVALUATOR_PATH
    target.write_bytes(target.read_bytes() + b'# executable dirty bytes\n')
    assert _git(repo, 'status', '--porcelain=v1', '--untracked-files=all')

    if injection == 'work_tree':
        monkeypatch.setenv('GIT_WORK_TREE', str(alternate))
    elif injection == 'index':
        index_path = _git(repo, 'rev-parse', '--git-path', 'index').decode().strip()
        index_path = str((repo / index_path).resolve())
        alternate_index = tmp_path / 'alternate.index'
        shutil.copy2(index_path, alternate_index)
        monkeypatch.setenv('GIT_INDEX_FILE', str(alternate_index))
        monkeypatch.setenv('GIT_WORK_TREE', str(alternate))
    else:
        monkeypatch.setenv('GIT_CONFIG_COUNT', '1')
        monkeypatch.setenv('GIT_CONFIG_KEY_0', 'core.worktree')
        monkeypatch.setenv('GIT_CONFIG_VALUE_0', str(alternate))

    with pytest.raises(review.LiveGatePreviewError, match='worktree_dirty'):
        review.require_live_preview_sources(repo)


def test_bound_source_path_alias_is_rejected(tmp_path) -> None:
    """A lexical alias must not select an approved source through another name."""

    repo, commit = committed_repo(tmp_path)
    alias = 'backend/app/rag/../rag/release_quality.py'

    with pytest.raises(
        review.LiveGatePreviewError, match='committed_source_unavailable'
    ):
        review._committed_bytes(repo, commit, alias)


@pytest.mark.parametrize(
    'name',
    [
        'GIT_DIR',
        'GIT_INDEX_FILE',
        'GIT_OBJECT_DIRECTORY',
        'GIT_ALTERNATE_OBJECT_DIRECTORIES',
        'GIT_REPLACE_REF_BASE',
        'GIT_CONFIG_GLOBAL',
        'GIT_CONFIG_KEY_0',
    ],
)
def test_git_subprocess_environment_drops_repository_overrides(monkeypatch, name):
    """Only module-owned neutral Git controls survive into subprocesses."""

    monkeypatch.setenv(name, 'attacker-controlled')

    environment = review._git_environment()

    assert name not in environment
    assert {key for key in environment if key.startswith('GIT_')} == {
        'GIT_CONFIG_COUNT',
        'GIT_OPTIONAL_LOCKS',
        'GIT_TERMINAL_PROMPT',
    }
    assert environment['GIT_CONFIG_COUNT'] == '0'


def test_symlinked_bound_source_is_rejected_when_supported(tmp_path) -> None:
    """The executable source path must resolve to its own tracked regular file."""

    repo, commit = committed_repo(tmp_path)
    path = review.LIVE_EVALUATOR_PATH
    target = repo / path
    replacement = tmp_path / 'replacement.py'
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    try:
        os.symlink(replacement, target)
    except OSError:
        pytest.skip('symlink creation is unavailable on this Windows host')

    with pytest.raises(review.LiveGatePreviewError, match='evaluator_unavailable'):
        review._committed_bytes(repo, commit, path)


def _require_capability_api() -> None:
    assert ApprovedRagExecutionCapability is not None
    assert issue_approved_execution_capability is not None


def _issue(h, *, authorization=None, authorization_row=None, source=None):
    _require_capability_api()
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority

    marker = DurableFileAuthority.open_runtime(h.reader.authority.marker_path)
    with (
        h.engine.connect() as connection,
        h.reader.authority._authority_barrier(connection, marker=marker) as guard,
    ):
        return issue_approved_execution_capability(
            connection,
            authorization=h.authorization if authorization is None else authorization,
            authorization_row=dict(h.auth_row)
            if authorization_row is None
            else authorization_row,
            source_binding=h.source if source is None else source,
            identity_secret=h.options['identity_secret'],
            barrier_guard=guard,
        )


def _evaluate_capability(h, evaluator, capability, quality):
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority

    marker = DurableFileAuthority.open_runtime(h.reader.authority.marker_path)
    with (
        h.engine.connect() as connection,
        h.reader.authority._authority_barrier(connection, marker=marker) as guard,
    ):
        return evaluator.evaluate(
            capability=capability,
            connection=connection,
            barrier_guard=guard,
            **quality,
        )


def _quality_inputs(h):
    manifest = h.authorization.manifest
    corpus = h.authorization.corpus
    positive_count = sum(case.case_kind == 'positive' for case in manifest.cases)
    definition = json.loads(h.preview.baseline_definition_bytes)
    baseline = FrozenLegacyBaselineMetrics(
        baseline_hmac=h.authorization.baseline_hmac,
        baseline_policy_snapshot_hmac=definition['policy_snapshot_hmac'],
        evaluator_code_hmac=definition['evaluator_code_hmac'],
        retriever_source_bundle_hmac=definition['retriever_source_bundle_hmac'],
        evaluator_source_commit=definition['evaluator_source_commit'],
        fixture_manifest_hmac=definition['fixture_manifest_hmac'],
        corpus_snapshot_hmac=definition['corpus_snapshot_hmac'],
        precision_numerator=positive_count,
        precision_denominator=30,
        recall_numerator=positive_count,
        recall_denominator=positive_count,
    )
    cases = _terminal_cases(manifest, corpus, h.authorization)
    subjects = {proof.role: proof.reviewer_subject_hmac for proof in h.proofs}
    labels = []
    for case in cases:
        for block in case.blocks:
            pair = (
                ('entailed', 'not_entailed')
                if case.ordinal == 0
                else ('entailed', 'entailed')
            )
            for role, label in zip(('reviewer_a', 'reviewer_b'), pair, strict=True):
                labels.append(
                    SignedReviewLabel(
                        case_id_hmac=case.case_id_hmac,
                        block_ordinal=block.block_ordinal,
                        reviewer_role=role,
                        reviewer_subject_hmac=subjects[role],
                        label=label,
                        signature_hmac=build_review_signature_hmac(
                            approval_hmac=h.authorization.approval_hmac,
                            fixture_manifest_hmac=manifest.fixture_manifest_hmac,
                            case_id_hmac=case.case_id_hmac,
                            block=block,
                            reviewer_role=role,
                            reviewer_subject_hmac=subjects[role],
                            label=label,
                            identity_secret=h.options['identity_secret'],
                        ),
                    )
                )
            if pair[0] != pair[1]:
                role, label = 'adjudicator_c', 'entailed'
                labels.append(
                    SignedReviewLabel(
                        case_id_hmac=case.case_id_hmac,
                        block_ordinal=block.block_ordinal,
                        reviewer_role=role,
                        reviewer_subject_hmac=subjects[role],
                        label=label,
                        signature_hmac=build_review_signature_hmac(
                            approval_hmac=h.authorization.approval_hmac,
                            fixture_manifest_hmac=manifest.fixture_manifest_hmac,
                            case_id_hmac=case.case_id_hmac,
                            block=block,
                            reviewer_role=role,
                            reviewer_subject_hmac=subjects[role],
                            label=label,
                            identity_secret=h.options['identity_secret'],
                        ),
                    )
                )
    return {
        'terminal_cases': cases,
        'signed_labels': tuple(labels),
        'baseline_metrics': baseline,
    }, subjects


def test_evaluator_consumes_exact_approved_capability_once(tmp_path) -> None:
    """Removing capability consumption would permit replay of one approval."""

    h = runtime_source_harness(tmp_path)
    capability = _issue(h)
    quality, subjects = _quality_inputs(h)
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=h.options['identity_secret'],
        reviewer_subject_hmacs=subjects,
    )

    assert (
        _evaluate_capability(h, evaluator, capability, quality).gate_outcome == 'green'
    )
    with pytest.raises(RagReleaseQualityError, match='execution_capability_invalid'):
        _evaluate_capability(h, evaluator, capability, quality)


def test_evaluator_rejects_caller_supplied_authority_without_capability() -> None:
    """Restoring the Task25-A DTO-only API would bypass the approved source."""

    from backend.tests.test_rag_release_quality import quality_inputs as _fixture

    values = _fixture.__wrapped__()
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=b'rag-release-quality-test-secret-v1',
        reviewer_subject_hmacs={
            role: f'{7000 + ordinal:064x}'
            for ordinal, role in enumerate(
                ('reviewer_a', 'reviewer_b', 'adjudicator_c'), 1
            )
        },
    )

    with pytest.raises(RagReleaseQualityError, match='execution_capability_required'):
        evaluator.evaluate(**values)


def test_internal_dto_entry_is_only_a_bounded_refusal() -> None:
    """An underscore method is not an execution authority boundary."""

    from backend.tests.test_rag_release_quality import quality_inputs as _fixture

    values = _fixture.__wrapped__()
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=b'rag-release-quality-test-secret-v1',
        reviewer_subject_hmacs={
            role: f'{7000 + ordinal:064x}'
            for ordinal, role in enumerate(
                ('reviewer_a', 'reviewer_b', 'adjudicator_c'), 1
            )
        },
    )

    with pytest.raises(RagReleaseQualityError, match='execution_capability'):
        evaluator._evaluate_approved(**values)


def test_public_issuer_does_not_expose_registry_closure(tmp_path) -> None:
    """A public mint function must not hand callers its registry via closure cells."""

    h = runtime_source_harness(tmp_path)
    capability = _issue(h)

    assert issue_approved_execution_capability.__closure__ is None
    forged = object.__new__(ApprovedRagExecutionCapability)
    quality, subjects = _quality_inputs(h)
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=h.options['identity_secret'],
        reviewer_subject_hmacs=subjects,
    )
    with pytest.raises(RagReleaseQualityError, match='execution_capability'):
        _evaluate_capability(h, evaluator, forged, quality)

    # Keep the legitimate capability live for the current-context tests below.
    assert repr(capability) == '<ApprovedRagExecutionCapability opaque>'


@pytest.mark.parametrize('drift', ['generation', 'transition'])
def test_release_identity_drift_refuses_and_burns_capability(tmp_path, drift) -> None:
    """An issued capability cannot outlive its exact release generation/digest."""

    h = runtime_source_harness(tmp_path)
    capability = _issue(h)
    quality, subjects = _quality_inputs(h)
    ledger = release_tables(build_rag_release_metadata()).ledgers
    with h.engine.begin() as connection:
        before = (
            connection.execute(
                select(ledger).where(
                    ledger.c.ledger_uuid == str(h.authorization.ledger_uuid),
                    ledger.c.ledger_epoch == h.authorization.ledger_epoch,
                )
            )
            .mappings()
            .one()
        )
        values = (
            {'generation': before['generation'] + 1}
            if drift == 'generation'
            else {'last_transition_digest': 'f' * 64}
        )
        connection.execute(
            update(ledger)
            .where(
                ledger.c.ledger_uuid == str(h.authorization.ledger_uuid),
                ledger.c.ledger_epoch == h.authorization.ledger_epoch,
            )
            .values(**values)
        )

    evaluator = RagReleaseQualityEvaluator(
        identity_secret=h.options['identity_secret'],
        reviewer_subject_hmacs=subjects,
    )
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority

    marker = DurableFileAuthority.open_runtime(h.reader.authority.marker_path)
    with (
        h.engine.connect() as connection,
        h.reader.authority._authority_barrier(connection, marker=marker) as guard,
    ):
        for _ in range(2):
            with pytest.raises(RagReleaseQualityError, match='execution_capability'):
                evaluator.evaluate(
                    capability=capability,
                    connection=connection,
                    barrier_guard=guard,
                    **quality,
                )


@pytest.mark.parametrize('drift', ['source', 'corpus', 'provider'])
def test_live_source_snapshot_drift_refuses_and_burns_capability(
    tmp_path, drift
) -> None:
    """Consumption repeats source/corpus/provider validation under the barrier."""

    h = runtime_source_harness(tmp_path)
    capability = _issue(h)
    quality, subjects = _quality_inputs(h)
    if drift == 'source':
        (h.repo / 'post-issue-source-drift').write_text('drift')
    elif drift == 'corpus':
        h.reader.value = replace(
            h.reader.value,
            corpus=replace(h.reader.value.corpus, corpus_generation=99),
        )
    else:
        h.reader.value = replace(
            h.reader.value,
            approved_provider_safety_snapshot_hmac='e' * 64,
        )
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=h.options['identity_secret'],
        reviewer_subject_hmacs=subjects,
    )

    for _ in range(2):
        with pytest.raises(RagReleaseQualityError, match='execution_capability'):
            _evaluate_capability(h, evaluator, capability, quality)


def test_exact_approved_source_can_issue_only_one_capability(tmp_path) -> None:
    """Allowing re-issuance would bypass the process-local one-use boundary."""

    h = runtime_source_harness(tmp_path)
    _issue(h)

    with pytest.raises(
        RagLiveGateCapabilityError, match='execution_capability_already_issued'
    ):
        _issue(h)


@pytest.mark.parametrize('mutation', ['token', 'provider', 'current_text_hmac'])
def test_capability_refuses_mutated_executable_preimage(tmp_path, mutation) -> None:
    """Trusting a re-signed DTO would admit a changed paid execution preimage."""

    h = runtime_source_harness(tmp_path)
    executable = h.authorization.manifest.executable
    selected = executable.cases[0]
    if mutation == 'token':
        component = replace(
            selected.components[1],
            reserved_input_tokens=selected.components[1].reserved_input_tokens + 1,
        )
        selected = replace(selected, components=(selected.components[0], component))
    elif mutation == 'provider':
        policy = replace(selected.components[1].policy, provider='forged-provider')
        selected = replace(
            selected,
            components=(
                selected.components[0],
                replace(selected.components[1], policy=policy),
            ),
        )
    else:
        selected = replace(selected, current_text_hmac='e' * 64)
    executable = replace(executable, cases=(selected,) + executable.cases[1:])
    manifest = replace(h.authorization.manifest, executable=executable)
    authorization = replace(h.authorization, manifest=manifest)

    with pytest.raises(RagLiveGateCapabilityError):
        _issue(h, authorization=authorization)


def test_capability_refuses_manifest_and_approval_comutation(tmp_path) -> None:
    """Treating caller-consistent DTOs as authority would admit co-mutation."""

    h = runtime_source_harness(tmp_path)
    case = replace(h.authorization.manifest.cases[0], allowed_slot_ids=('E8',))
    manifest = replace(
        h.authorization.manifest,
        cases=(case,) + h.authorization.manifest.cases[1:],
    )
    authorization = replace(
        h.authorization,
        manifest=manifest,
        approval_hmac='e' * 64,
    )
    row = dict(h.auth_row) | {
        'approval_hmac': authorization.approval_hmac,
        'manifest_hmac': manifest.manifest_hmac,
    }

    with pytest.raises(RagLiveGateCapabilityError):
        _issue(h, authorization=authorization, authorization_row=row)


def test_capability_refuses_copied_or_cross_ledger_authorization(tmp_path) -> None:
    """Dropping object/UUID binding would let an equal or cross-ledger DTO mint."""

    h = runtime_source_harness(tmp_path)
    with pytest.raises(RagLiveGateCapabilityError):
        _issue(h, authorization=copy(h.authorization))

    other_uuid = UUID('22222222-2222-4222-8222-222222222222')
    authorization = replace(h.authorization, ledger_uuid=other_uuid)
    row = dict(h.auth_row) | {'ledger_uuid': str(other_uuid)}
    with pytest.raises(RagLiveGateCapabilityError):
        _issue(h, authorization=authorization, authorization_row=row)


def test_capability_refuses_extra_or_noncanonical_authorization_row(tmp_path) -> None:
    """Ignoring extra callback-bearing DB-row input would retain caller state."""

    h = runtime_source_harness(tmp_path)

    class Alias:
        def __deepcopy__(self, memo):
            return self

    row = dict(h.auth_row) | {'unapproved_extra': Alias()}

    with pytest.raises(RagLiveGateCapabilityError):
        _issue(h, authorization_row=row)


def test_capability_is_opaque_noncopyable_nonserializable_and_unforgeable(
    tmp_path,
) -> None:
    """Removing the private issuance registry would make capability DTOs forgeable."""

    h = runtime_source_harness(tmp_path)
    capability = _issue(h)
    for operation in (
        lambda: copy(capability),
        lambda: deepcopy(capability),
        lambda: pickle.dumps(capability),
        lambda: json.dumps(capability),
    ):
        with pytest.raises((TypeError, pickle.PickleError)):
            operation()

    forged = object.__new__(ApprovedRagExecutionCapability)
    quality, subjects = _quality_inputs(h)
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=h.options['identity_secret'],
        reviewer_subject_hmacs=subjects,
    )
    with pytest.raises(RagReleaseQualityError, match='execution_capability_invalid'):
        _evaluate_capability(h, evaluator, forged, quality)


def test_capability_refuses_post_issue_authorization_identity_mutation(
    tmp_path,
) -> None:
    """Caching only HMAC strings would miss mutation of the issued authority object."""

    h = runtime_source_harness(tmp_path)
    capability = _issue(h)
    quality, subjects = _quality_inputs(h)
    object.__setattr__(
        h.authorization,
        'ledger_uuid',
        UUID('33333333-3333-4333-8333-333333333333'),
    )
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=h.options['identity_secret'],
        reviewer_subject_hmacs=subjects,
    )

    with pytest.raises(RagReleaseQualityError, match='execution_capability_invalid'):
        _evaluate_capability(h, evaluator, capability, quality)
