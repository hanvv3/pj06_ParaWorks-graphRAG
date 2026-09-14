from __future__ import annotations

import json
import pickle
import subprocess
from copy import copy, deepcopy
from dataclasses import replace
from uuid import UUID

import pytest

from backend.app.rag import release_review as review
from backend.app.rag.release_quality import (
    FrozenLegacyBaselineMetrics,
    RagReleaseQualityError,
    RagReleaseQualityEvaluator,
    SignedReviewLabel,
    build_review_signature_hmac,
)
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
    for path in source_paths:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        committed[path] = f'# committed LF blob: {path}\n'.encode()
        target.write_bytes(committed[path].replace(b'\n', b'\r\n'))

    calls = []

    def fake_git_read(repository, *arguments):
        calls.append((repository, arguments))
        if arguments == ('rev-parse', 'HEAD'):
            return (commit + '\n').encode()
        if arguments == ('status', '--porcelain=v1', '--untracked-files=all'):
            return b''
        assert arguments[0] == 'show'
        selected_commit, path = arguments[1].split(':', 1)
        assert selected_commit == commit
        return committed[path]

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

    assert evaluator.evaluate(capability=capability, **quality).gate_outcome == 'green'
    with pytest.raises(RagReleaseQualityError, match='execution_capability_invalid'):
        evaluator.evaluate(capability=capability, **quality)


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
        evaluator.evaluate(capability=forged, **quality)


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
        evaluator.evaluate(capability=capability, **quality)
