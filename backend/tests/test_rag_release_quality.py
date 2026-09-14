from __future__ import annotations

import json
from dataclasses import fields, replace
from decimal import Decimal
from uuid import UUID

import pytest

from backend.app.agent_runtime.fingerprints import (
    canonical_json_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
)
from backend.app.rag.release_quality import (
    FrozenLegacyBaselineMetrics,
    RagReleaseQualityError,
    RagReleaseQualityEvaluator,
    SanitizedLiveBlockResult,
    SanitizedLiveCaseResult,
    SignedReviewLabel,
    build_review_signature_hmac,
)
from backend.app.rag.release_review import (
    AuthorizedRagLiveGate,
    FrozenCaseClaimCase,
    FrozenCaseClaimComponent,
    FrozenCaseClaimManifest,
    FrozenCorpusMember,
    FrozenCorpusSnapshot,
    FrozenLiveManifestCase,
    FrozenLiveManifestSnapshot,
    LiveGateLimits,
    freeze_live_corpus,
)

SECRET = b'rag-release-quality-test-secret-v1'
RUBRIC = 'rag-live-quality-rubric:v1'
POLICY = 'rag-live-gate:v1'
ROLES = ('reviewer_a', 'reviewer_b', 'adjudicator_c')
SUBJECTS = {role: f'{7000 + ordinal:064x}' for ordinal, role in enumerate(ROLES, 1)}


class _TextAlias(str):
    pass


class _IntegerAlias(int):
    pass


def _hmac(value: object, schema: str, *, policy: str = POLICY) -> str:
    return keyed_fingerprint(
        value,
        secret=SECRET,
        schema_version=schema,
        policy_version=policy,
    )


def _reviewer_roster_hmac(subjects: dict[str, str] = SUBJECTS) -> str:
    return _hmac(
        {
            'adjudicator_c_subject_hmac': subjects['adjudicator_c'],
            'reviewer_a_subject_hmac': subjects['reviewer_a'],
            'reviewer_b_subject_hmac': subjects['reviewer_b'],
            'roster_version': 'rag-live-reviewer-roster:v1',
        },
        'rag-live-reviewer-roster:v1',
        policy=RUBRIC,
    )


def _case_shape(ordinal: int) -> tuple[str, str, str | None]:
    if ordinal < 10:
        return 'ask', 'keyword', None
    if ordinal < 15:
        return 'ask', 'pgvector', None
    if ordinal < 25:
        return (
            'assistant',
            'keyword',
            'context-' + str(ordinal) if ordinal < 20 else None,
        )
    return (
        'assistant',
        'pgvector',
        'context-' + str(ordinal) if ordinal < 28 else None,
    )


def _provider_policy(component: str) -> AuthorizedProviderPolicySnapshot:
    query = component == 'query_embedding'
    return AuthorizedProviderPolicySnapshot(
        component=component,
        provider='openai',
        model='text-embedding-3-small' if query else 'gpt-5.4-mini-2026-03-17',
        reasoning_or_config_identity=('dimensions:1536' if query else 'none'),
        authorized_model_config_version=(
            'rag-query-embedding-config:v1' if query else 'rag-answer-model-config:v1'
        ),
        authorized_model_config_snapshot_hmac=(f'{8101 if query else 8102:064x}'),
        authorized_cost_policy_version=(
            'rag-query-embedding-cost:v1' if query else 'rag-answer-cost:v1'
        ),
        authorized_token_estimator_version=(
            'openai-cl100k-text-embedding-3-small:v1'
            if query
            else 'openai-o200k-rag-answer:v1'
        ),
        fingerprint_key_version='test-v1',
        fingerprint_key_material_verifier=f'{8103:064x}',
        authorized_policy_snapshot_hmac=f'{8104 if query else 8105:064x}',
    )


def _manifest() -> FrozenLiveManifestSnapshot:
    fixture_hmac = f'{101:064x}'
    cases = []
    for ordinal in range(30):
        surface, backend, prior_context = _case_shape(ordinal)
        positive = ordinal % 2 == 0
        cases.append(
            FrozenLiveManifestCase(
                ordinal=ordinal,
                case_id_hmac=f'{1000 + ordinal:064x}',
                case_kind='positive' if positive else 'hard_negative',
                surface=surface,
                configured_backend=backend,
                question_fixture_id=f'question-{ordinal}',
                security_scope_fixture_id=f'scope-{ordinal}',
                prior_context_fixture_id=prior_context,
                query_bytes_hmac=f'{2000 + ordinal:064x}',
                expected_no_answer=not positive,
                allowed_support_modes=('trusted_fact',),
                allowed_slot_ids=('E1', 'E2'),
                relevant_serving_identity_hmacs=(f'{3000 + ordinal:064x}',),
                required_serving_identity_hmacs=(f'{3000 + ordinal:064x}',)
                if positive
                else (),
                query_embedding_required=backend == 'pgvector',
                answer_generation_required=True,
                query_embedding_reserved_cost_usd=Decimal('0.000160')
                if backend == 'pgvector'
                else Decimal('0.000000'),
                answer_generation_reserved_cost_usd=Decimal('0.012000')
                if backend == 'keyword'
                else Decimal('0.011840'),
                case_total_reserved_cost_usd=Decimal('0.012000'),
            )
        )
    executable = FrozenCaseClaimManifest(
        cases=tuple(
            FrozenCaseClaimCase(
                ordinal=case.ordinal,
                case_id_hmac=case.case_id_hmac,
                surface=case.surface,
                configured_backend=case.configured_backend,
                current_text_hmac=f'{2200 + case.ordinal:064x}',
                retrieval_query_hmac=case.query_bytes_hmac,
                security_scope_fingerprint=f'{2300 + case.ordinal:064x}',
                components=(
                    FrozenCaseClaimComponent(
                        policy=_provider_policy('query_embedding'),
                        reserved_input_tokens=(
                            128 if case.query_embedding_required else 0
                        ),
                        reserved_output_tokens=0,
                        reserved_cost_usd=case.query_embedding_reserved_cost_usd,
                    ),
                    FrozenCaseClaimComponent(
                        policy=_provider_policy('answer_generation'),
                        reserved_input_tokens=2048,
                        reserved_output_tokens=512,
                        reserved_cost_usd=case.answer_generation_reserved_cost_usd,
                    ),
                ),
            )
            for case in cases
        ),
        source_manifest_hmac=fixture_hmac,
    )
    return FrozenLiveManifestSnapshot(
        live_gate_contract_version=POLICY,
        fixture_manifest_version='rag-live-quality-30:v1',
        fixture_manifest_path='backend/tests/fixtures/rag_v2_live_gate_30.json',
        fixture_manifest_sha256=f'{102:064x}',
        fixture_manifest_hmac=fixture_hmac,
        manifest_hmac=fixture_hmac,
        clean_git_commit='a' * 40,
        rubric_version=RUBRIC,
        cases=tuple(cases),
        executable=executable,
    )


def _corpus() -> FrozenCorpusSnapshot:
    return freeze_live_corpus(
        corpus_generation=9,
        vector_index_generation=4,
        embedding_model_bytes=b'text-embedding-3-small',
        index_policy_version_bytes=b'rag-v2-serving-index:v1',
        pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
        members=tuple(
            FrozenCorpusMember(
                ordinal=ordinal,
                serving_identity_hmac=f'{3000 + ordinal:064x}',
                serving_version_fingerprint=f'{3100 + ordinal:064x}',
                model_content_hmac=f'{3200 + ordinal:064x}',
                canonical_citation_projection_hmac=f'{3300 + ordinal:064x}',
                effective_permission='internal',
                support_mode='trusted_fact',
                vector_index_state_hmac=f'{3400 + ordinal:064x}',
            )
            for ordinal in range(30)
        ),
        identity_secret=SECRET,
    )


def _baseline_policy_hmac() -> str:
    return _hmac(
        {
            'annotation_schema_version': 'rag-live-relevance-annotation:v1',
            'evaluator_version': 'rag-live-retrieval-evaluator:v1',
            'keyword_scorer_version': 'rag-keyword-lexical-compat:v1',
            'pgvector_distance_policy_version': 'rag-pgvector-cosine-distance:v1',
            'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
            'rubric_version': RUBRIC,
        },
        'rag-live-baseline-policy-snapshot:v1',
    )


def _baseline(manifest, corpus) -> FrozenLegacyBaselineMetrics:
    evaluator_code_hmac = f'{104:064x}'
    retriever_source_bundle_hmac = f'{105:064x}'
    policy_hmac = _baseline_policy_hmac()
    definition = {
        'annotation_schema_version': 'rag-live-relevance-annotation:v1',
        'cases': [
            {
                'case_id_hmac': case.case_id_hmac,
                'configured_backend': case.configured_backend,
                'legacy_retriever_version': (
                    f'rag-v1-{case.configured_backend}-retriever:v1'
                ),
                'query_bytes_hmac': case.query_bytes_hmac,
                'relevant_serving_identity_hmacs': list(
                    case.relevant_serving_identity_hmacs
                ),
                'v2_retriever_version': (
                    f'rag-v2-{case.configured_backend}-retriever:v1'
                ),
            }
            for case in manifest.cases
        ],
        'corpus_snapshot_hmac': corpus.corpus_snapshot_hmac,
        'evaluator_code_hmac': evaluator_code_hmac,
        'evaluator_path': 'backend/app/rag/release_quality.py',
        'evaluator_source_commit': manifest.clean_git_commit,
        'evaluator_version': 'rag-live-retrieval-evaluator:v1',
        'fixture_manifest_hmac': manifest.fixture_manifest_hmac,
        'keyword_scorer_version': 'rag-keyword-lexical-compat:v1',
        'pgvector_distance_policy_version': 'rag-pgvector-cosine-distance:v1',
        'policy_snapshot_hmac': policy_hmac,
        'retriever_source_bundle_hmac': retriever_source_bundle_hmac,
    }
    return FrozenLegacyBaselineMetrics(
        baseline_hmac=_hmac(definition, 'rag-live-baseline-definition:v1'),
        baseline_policy_snapshot_hmac=policy_hmac,
        evaluator_code_hmac=evaluator_code_hmac,
        retriever_source_bundle_hmac=retriever_source_bundle_hmac,
        evaluator_source_commit=manifest.clean_git_commit,
        fixture_manifest_hmac=manifest.fixture_manifest_hmac,
        corpus_snapshot_hmac=corpus.corpus_snapshot_hmac,
        precision_numerator=15,
        precision_denominator=30,
        recall_numerator=15,
        recall_denominator=15,
    )


def _approval(manifest, corpus, baseline) -> AuthorizedRagLiveGate:
    return AuthorizedRagLiveGate(
        live_gate_contract_version=POLICY,
        authorization_state='unused',
        ledger_uuid=UUID('00000000-0000-0000-0000-000000000001'),
        ledger_epoch=1,
        approval_base_generation=0,
        approval_base_release_marker_file_digest=f'{106:064x}',
        approval_id_hmac=f'{107:064x}',
        approval_hmac=f'{108:064x}',
        manifest=manifest,
        corpus=corpus,
        baseline_hmac=baseline.baseline_hmac,
        approved_provider_safety_snapshot_hmac=f'{109:064x}',
        reviewer_roster_hmac=_reviewer_roster_hmac(),
        validation_database_identity_hmac=f'{110:064x}',
        designated_environment_id_hmac=f'{111:064x}',
        designated_host_id_hmac=f'{112:064x}',
        implementation_plan_reference_hmac=f'{113:064x}',
        limits=LiveGateLimits(),
    )


def _terminal_cases(manifest, corpus, approval):
    results = []
    for case in manifest.cases:
        positive = case.case_kind == 'positive'
        blocks = (
            (
                SanitizedLiveBlockResult(
                    block_ordinal=0,
                    block_result_hmac=f'{4000 + case.ordinal:064x}',
                    evidence_projection_hmac=f'{5000 + case.ordinal:064x}',
                ),
            )
            if positive
            else ()
        )
        results.append(
            SanitizedLiveCaseResult(
                ordinal=case.ordinal,
                case_id_hmac=case.case_id_hmac,
                case_kind=case.case_kind,
                surface=case.surface,
                configured_backend=case.configured_backend,
                state='complete',
                outcome='supported' if positive else 'insufficient_evidence',
                runtime_agent_run_id_hmac=f'{6000 + case.ordinal:064x}',
                case_projection_hmac=f'{6100 + case.ordinal:064x}',
                current_corpus_snapshot_hmac=corpus.corpus_snapshot_hmac,
                provider_safety_snapshot_hmac=(
                    approval.approved_provider_safety_snapshot_hmac
                ),
                query_embedding_dispatch_count=1
                if case.configured_backend == 'pgvector'
                else 0,
                answer_generation_dispatch_count=1,
                reserved_cost_usd=case.case_total_reserved_cost_usd,
                charged_cost_usd=case.case_total_reserved_cost_usd,
                legacy_precision_numerator=1 if positive else 0,
                legacy_precision_denominator=1,
                legacy_recall_numerator=1 if positive else 0,
                legacy_recall_denominator=1 if positive else 0,
                v2_precision_numerator=1 if positive else 0,
                v2_precision_denominator=1,
                v2_recall_numerator=1 if positive else 0,
                v2_recall_denominator=1 if positive else 0,
                expected_no_answer=not positive,
                hard_negative_correct=not positive,
                positive_answered=positive,
                required_slot_covered=positive,
                blocks=blocks,
            )
        )
    return tuple(results)


def _signature(approval, manifest, case_id_hmac, block, role, subject, label):
    return _hmac(
        {
            'approval_hmac': approval.approval_hmac,
            'block_ordinal': block.block_ordinal,
            'block_result_hmac': block.block_result_hmac,
            'case_id_hmac': case_id_hmac,
            'evidence_projection_hmac': block.evidence_projection_hmac,
            'fixture_manifest_hmac': manifest.fixture_manifest_hmac,
            'label': label,
            'reviewer_role': role,
            'reviewer_subject_hmac': subject,
            'rubric_version': RUBRIC,
        },
        'rag-live-review-signature:v1',
        policy=RUBRIC,
    )


def _labels(cases, manifest, approval):
    labels = []
    for case in cases:
        for block in case.blocks:
            pair = (
                ('entailed', 'not_entailed')
                if case.ordinal == 0
                else (
                    'entailed',
                    'entailed',
                )
            )
            for role, label in zip(ROLES[:2], pair, strict=True):
                labels.append(
                    SignedReviewLabel(
                        case_id_hmac=case.case_id_hmac,
                        block_ordinal=block.block_ordinal,
                        reviewer_role=role,
                        reviewer_subject_hmac=SUBJECTS[role],
                        label=label,
                        signature_hmac=_signature(
                            approval,
                            manifest,
                            case.case_id_hmac,
                            block,
                            role,
                            SUBJECTS[role],
                            label,
                        ),
                    )
                )
            if pair[0] != pair[1]:
                role = 'adjudicator_c'
                label = 'entailed'
                labels.append(
                    SignedReviewLabel(
                        case_id_hmac=case.case_id_hmac,
                        block_ordinal=block.block_ordinal,
                        reviewer_role=role,
                        reviewer_subject_hmac=SUBJECTS[role],
                        label=label,
                        signature_hmac=_signature(
                            approval,
                            manifest,
                            case.case_id_hmac,
                            block,
                            role,
                            SUBJECTS[role],
                            label,
                        ),
                    )
                )
    return tuple(labels)


@pytest.fixture
def quality_inputs():
    manifest = _manifest()
    corpus = _corpus()
    baseline = _baseline(manifest, corpus)
    approval = _approval(manifest, corpus, baseline)
    terminal_cases = _terminal_cases(manifest, corpus, approval)
    return {
        'terminal_cases': terminal_cases,
        'signed_labels': _labels(terminal_cases, manifest, approval),
        'baseline_metrics': baseline,
        'manifest': manifest,
        'corpus': corpus,
        'approval': approval,
    }


def _evaluate(values):
    return RagReleaseQualityEvaluator(
        identity_secret=SECRET,
        reviewer_subject_hmacs=SUBJECTS,
    )._evaluate_approved(**values)


@pytest.mark.parametrize('ordinal', [True, -1, object()])
def test_public_review_signer_rejects_noncanonical_block_ordinal(
    quality_inputs, ordinal
) -> None:
    """Skipping native ordinal validation would sign an impossible block."""

    block = SanitizedLiveBlockResult(
        block_ordinal=ordinal,
        block_result_hmac=f'{9801:064x}',
        evidence_projection_hmac=f'{9802:064x}',
    )

    with pytest.raises(RagReleaseQualityError, match='review_labels_invalid'):
        build_review_signature_hmac(
            approval_hmac=quality_inputs['approval'].approval_hmac,
            fixture_manifest_hmac=quality_inputs['manifest'].fixture_manifest_hmac,
            case_id_hmac=quality_inputs['terminal_cases'][0].case_id_hmac,
            block=block,
            reviewer_role='reviewer_a',
            reviewer_subject_hmac=SUBJECTS['reviewer_a'],
            label='entailed',
            identity_secret=SECRET,
        )


def _replace_case(values, ordinal, **changes):
    cases = list(values['terminal_cases'])
    cases[ordinal] = replace(cases[ordinal], **changes)
    return values | {'terminal_cases': tuple(cases)}


def _resign(values, case_ordinal, *, a, b, c=None):
    cases = values['terminal_cases']
    manifest = values['manifest']
    approval = values['approval']
    labels = []
    for case in cases:
        for block in case.blocks:
            selected = (
                (a, b, c)
                if case.ordinal == case_ordinal
                else (
                    ('entailed', 'not_entailed', 'entailed')
                    if case.ordinal == 0
                    else ('entailed', 'entailed', None)
                )
            )
            for role, label in zip(ROLES, selected, strict=True):
                if label is None:
                    continue
                labels.append(
                    SignedReviewLabel(
                        case_id_hmac=case.case_id_hmac,
                        block_ordinal=block.block_ordinal,
                        reviewer_role=role,
                        reviewer_subject_hmac=SUBJECTS[role],
                        label=label,
                        signature_hmac=_signature(
                            approval,
                            manifest,
                            case.case_id_hmac,
                            block,
                            role,
                            SUBJECTS[role],
                            label,
                        ),
                    )
                )
    return values | {'signed_labels': tuple(labels)}


def test_evaluator_builds_exact_canonical_green_report(quality_inputs) -> None:
    report = _evaluate(quality_inputs)

    assert report.case_count == 30
    assert report.gate_outcome == 'green'
    assert report.failure_reasons == ()
    assert report.faithfulness_entailed_blocks == 15
    assert report.faithfulness_total_blocks == 15
    assert report.hard_negative_case_count == 15
    assert report.hard_negative_correct_count == 15
    assert report.positive_case_count == 15
    assert report.positive_answered_count == 15
    assert report.legacy_precision_numerator == 15
    assert report.legacy_precision_denominator == 30
    assert report.v2_recall_numerator == 15
    assert report.v2_recall_denominator == 15
    assert report.payload_canonical_bytes == canonical_json_bytes(
        json.loads(report.payload_canonical_bytes)
    )
    assert report.quality_report_hmac == _hmac(
        json.loads(report.payload_canonical_bytes), 'rag-live-quality-report:v1'
    )
    assert b'answer_text' not in report.payload_canonical_bytes
    assert b'source_snippet' not in report.payload_canonical_bytes


@pytest.mark.parametrize(
    ('mutation', 'reason'),
    [
        ({'required_slot_covered': False}, ('positive_answer_coverage',)),
    ],
)
def test_evaluator_reports_boolean_quality_gate_failures(
    quality_inputs, mutation, reason
) -> None:
    values = _replace_case(quality_inputs, 2, **mutation)

    report = _evaluate(values)

    assert report.gate_outcome == 'quality_gate_failed'
    assert report.failure_reasons == reason
    assert report.positive_answered_count == 15


def test_evaluator_reports_a_structured_answer_on_a_hard_negative(
    quality_inputs,
) -> None:
    block = SanitizedLiveBlockResult(0, f'{9001:064x}', f'{9002:064x}')
    values = _replace_case(
        quality_inputs,
        1,
        hard_negative_correct=False,
        outcome='supported',
        blocks=(block,),
    )
    values = values | {
        'signed_labels': _labels(
            values['terminal_cases'], values['manifest'], values['approval']
        )
    }

    report = _evaluate(values)

    assert report.failure_reasons == ('hard_negative_accuracy',)


def test_evaluator_reports_a_missing_positive_answer(quality_inputs) -> None:
    values = _replace_case(
        quality_inputs,
        2,
        positive_answered=False,
        required_slot_covered=False,
        outcome='insufficient_evidence',
        blocks=(),
    )
    values = values | {
        'signed_labels': _labels(
            values['terminal_cases'], values['manifest'], values['approval']
        )
    }

    report = _evaluate(values)

    assert report.failure_reasons == ('positive_answer_coverage',)


def test_evaluator_counts_ambiguous_as_unfaithful(quality_inputs) -> None:
    values = _resign(quality_inputs, 2, a='ambiguous', b='ambiguous')

    report = _evaluate(values)

    assert report.faithfulness_entailed_blocks == 14
    assert report.faithfulness_total_blocks == 15
    assert report.failure_reasons == ('faithfulness',)


def test_evaluator_accepts_the_exact_95_percent_faithfulness_boundary(
    quality_inputs,
) -> None:
    original = quality_inputs['terminal_cases'][2].blocks[0]
    blocks = (original,) + tuple(
        SanitizedLiveBlockResult(
            ordinal, f'{9100 + ordinal:064x}', f'{9200 + ordinal:064x}'
        )
        for ordinal in range(1, 6)
    )
    values = _replace_case(quality_inputs, 2, blocks=blocks)
    labels = list(
        _labels(values['terminal_cases'], values['manifest'], values['approval'])
    )
    case = values['terminal_cases'][2]
    target_index = next(
        index
        for index, label in enumerate(labels)
        if label.case_id_hmac == case.case_id_hmac
        and label.block_ordinal == 0
        and label.reviewer_role == 'reviewer_a'
    )
    for offset, role in enumerate(('reviewer_a', 'reviewer_b')):
        prior = labels[target_index + offset]
        labels[target_index + offset] = replace(
            prior,
            label='ambiguous',
            signature_hmac=_signature(
                values['approval'],
                values['manifest'],
                case.case_id_hmac,
                case.blocks[0],
                role,
                SUBJECTS[role],
                'ambiguous',
            ),
        )

    report = _evaluate(values | {'signed_labels': tuple(labels)})

    assert report.faithfulness_entailed_blocks == 19
    assert report.faithfulness_total_blocks == 20
    assert report.gate_outcome == 'green'


@pytest.mark.parametrize(
    ('field', 'numerator', 'denominator', 'reason'),
    [
        ('v2_precision', 0, 1, 'retrieval_precision'),
        ('v2_recall', 0, 1, 'retrieval_recall'),
    ],
)
def test_evaluator_compares_exact_fractions_without_rounding(
    quality_inputs, field, numerator, denominator, reason
) -> None:
    values = _replace_case(
        quality_inputs,
        2,
        **{
            f'{field}_numerator': numerator,
            f'{field}_denominator': denominator,
        },
    )

    report = _evaluate(values)

    assert reason in report.failure_reasons


@pytest.mark.parametrize(
    'cases',
    [
        lambda rows: rows[:-1],
        lambda rows: rows[:1] + (rows[0],) + rows[2:],
        lambda rows: (rows[1], rows[0]) + rows[2:],
    ],
)
def test_evaluator_rejects_non_exact_30_case_roster(quality_inputs, cases) -> None:
    values = quality_inputs | {
        'terminal_cases': cases(quality_inputs['terminal_cases'])
    }

    with pytest.raises(RagReleaseQualityError, match='case_roster_invalid'):
        _evaluate(values)


@pytest.mark.parametrize(
    'changes',
    [
        {'case_kind': 'hard_negative'},
        {'surface': 'assistant'},
        {'configured_backend': 'pgvector'},
        {'expected_no_answer': True},
        {'current_corpus_snapshot_hmac': f'{999:064x}'},
        {'provider_safety_snapshot_hmac': f'{998:064x}'},
    ],
)
def test_evaluator_rejects_case_or_authority_identity_drift(
    quality_inputs, changes
) -> None:
    values = _replace_case(quality_inputs, 0, **changes)

    with pytest.raises(RagReleaseQualityError):
        _evaluate(values)


@pytest.mark.parametrize(
    'mutator',
    [
        lambda v: v | {'signed_labels': v['signed_labels'][:-1]},
        lambda v: v | {'signed_labels': v['signed_labels'] + (v['signed_labels'][0],)},
        lambda v: (
            v
            | {
                'signed_labels': (
                    replace(v['signed_labels'][0], signature_hmac=f'{997:064x}'),
                )
                + v['signed_labels'][1:]
            }
        ),
        lambda v: (
            v
            | {
                'signed_labels': (
                    replace(v['signed_labels'][0], reviewer_role='reviewer_b'),
                )
                + v['signed_labels'][1:]
            }
        ),
        lambda v: (
            v
            | {
                'signed_labels': (
                    replace(v['signed_labels'][0], case_id_hmac=f'{996:064x}'),
                )
                + v['signed_labels'][1:]
            }
        ),
    ],
)
def test_evaluator_rejects_missing_duplicate_mismatched_or_reordered_labels(
    quality_inputs, mutator
) -> None:
    with pytest.raises(RagReleaseQualityError, match='review_labels_invalid'):
        _evaluate(mutator(quality_inputs))


def test_evaluator_rejects_reviewer_takeover_even_with_a_valid_signature(
    quality_inputs,
) -> None:
    labels = list(quality_inputs['signed_labels'])
    target = labels[3]
    case = quality_inputs['terminal_cases'][2]
    block = case.blocks[0]
    attacker = f'{995:064x}'
    labels[3] = replace(
        target,
        reviewer_subject_hmac=attacker,
        signature_hmac=_signature(
            quality_inputs['approval'],
            quality_inputs['manifest'],
            case.case_id_hmac,
            block,
            target.reviewer_role,
            attacker,
            target.label,
        ),
    )

    with pytest.raises(RagReleaseQualityError, match='review_labels_invalid'):
        _evaluate(quality_inputs | {'signed_labels': tuple(labels)})


def test_evaluator_requires_adjudicator_only_after_disagreement(quality_inputs) -> None:
    missing = _resign(quality_inputs, 0, a='entailed', b='not_entailed')
    extra = _resign(quality_inputs, 2, a='entailed', b='entailed', c='entailed')
    no_majority = _resign(
        quality_inputs, 2, a='entailed', b='not_entailed', c='ambiguous'
    )

    for values in (missing, extra, no_majority):
        with pytest.raises(RagReleaseQualityError, match='review_labels_invalid'):
            _evaluate(values)


def test_evaluator_cannot_infer_faithfulness_from_sanitized_rows_alone(
    quality_inputs,
) -> None:
    with pytest.raises(RagReleaseQualityError, match='review_labels_invalid'):
        _evaluate(quality_inputs | {'signed_labels': ()})


@pytest.mark.parametrize(
    'mutator',
    [
        lambda v: (
            v
            | {
                'baseline_metrics': replace(
                    v['baseline_metrics'], evaluator_code_hmac=f'{994:064x}'
                )
            }
        ),
        lambda v: (
            v
            | {
                'baseline_metrics': replace(
                    v['baseline_metrics'], precision_numerator=14
                )
            }
        ),
        lambda v: (
            v | {'corpus': replace(v['corpus'], corpus_snapshot_hmac=f'{993:064x}')}
        ),
        lambda v: v | {'manifest': replace(v['manifest'], manifest_hmac=f'{992:064x}')},
        lambda v: v | {'approval': replace(v['approval'], approval_hmac=f'{991:064x}')},
    ],
)
def test_evaluator_rejects_baseline_manifest_corpus_or_approval_drift(
    quality_inputs, mutator
) -> None:
    with pytest.raises(RagReleaseQualityError):
        _evaluate(mutator(quality_inputs))


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('v2_precision_numerator', -1),
        ('v2_precision_denominator', -1),
        ('v2_recall_numerator', 2),
        ('legacy_recall_denominator', True),
        ('blocks', []),
    ],
)
def test_evaluator_rejects_invalid_metric_and_block_shapes(
    quality_inputs, field, value
) -> None:
    values = _replace_case(quality_inputs, 0, **{field: value})

    with pytest.raises(RagReleaseQualityError, match='case_result_invalid'):
        _evaluate(values)


def test_evaluator_rejects_result_outcome_that_disagrees_with_block_presence(
    quality_inputs,
) -> None:
    missing_answer = _replace_case(
        quality_inputs,
        2,
        positive_answered=False,
        required_slot_covered=False,
        blocks=(),
    )

    with pytest.raises(RagReleaseQualityError, match='case_result_invalid'):
        _evaluate(
            missing_answer
            | {
                'signed_labels': _labels(
                    missing_answer['terminal_cases'],
                    missing_answer['manifest'],
                    missing_answer['approval'],
                )
            }
        )


@pytest.mark.parametrize(
    'subjects',
    [
        {'reviewer_a': SUBJECTS['reviewer_a'], 'reviewer_b': SUBJECTS['reviewer_b']},
        {
            'reviewer_a': SUBJECTS['reviewer_a'],
            'reviewer_b': SUBJECTS['reviewer_a'],
            'adjudicator_c': SUBJECTS['adjudicator_c'],
        },
    ],
)
def test_evaluator_requires_the_complete_pairwise_distinct_frozen_roster(
    subjects,
) -> None:
    with pytest.raises(RagReleaseQualityError, match='reviewer_roster_invalid'):
        RagReleaseQualityEvaluator(
            identity_secret=SECRET,
            reviewer_subject_hmacs=subjects,
        )


def test_evaluator_turns_malformed_manifest_collections_into_a_bounded_refusal(
    quality_inputs,
) -> None:
    cases = list(quality_inputs['manifest'].cases)
    cases[0] = replace(cases[0], required_serving_identity_hmacs=[])
    manifest = replace(quality_inputs['manifest'], cases=tuple(cases))
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


def test_evaluator_rejects_an_unapproved_manifest_support_mode(quality_inputs) -> None:
    cases = list(quality_inputs['manifest'].cases)
    cases[0] = replace(cases[0], allowed_support_modes=('direct_fact',))
    manifest = replace(quality_inputs['manifest'], cases=tuple(cases))
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


def test_sanitized_contracts_have_no_raw_answer_or_evidence_fields() -> None:
    names = {
        field.name
        for contract in (SanitizedLiveBlockResult, SanitizedLiveCaseResult)
        for field in fields(contract)
    }

    assert 'answer_text' not in names
    assert 'evidence_projection' not in names
    assert 'source_snippets' not in names


def test_evaluator_rejects_a_mutated_internal_reviewer_map_and_resigned_takeover(
    quality_inputs,
) -> None:
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=SECRET,
        reviewer_subject_hmacs=SUBJECTS,
    )
    attacker = f'{999_001:064x}'
    evaluator._reviewer_subjects['reviewer_a'] = attacker
    cases_by_id = {case.case_id_hmac: case for case in quality_inputs['terminal_cases']}
    labels = []
    for signed in quality_inputs['signed_labels']:
        if signed.reviewer_role != 'reviewer_a':
            labels.append(signed)
            continue
        case = cases_by_id[signed.case_id_hmac]
        block = case.blocks[signed.block_ordinal]
        labels.append(
            replace(
                signed,
                reviewer_subject_hmac=attacker,
                signature_hmac=_signature(
                    quality_inputs['approval'],
                    quality_inputs['manifest'],
                    case.case_id_hmac,
                    block,
                    'reviewer_a',
                    attacker,
                    signed.label,
                ),
            )
        )

    with pytest.raises(RagReleaseQualityError, match='reviewer_roster_invalid'):
        evaluator._evaluate_approved(
            **(quality_inputs | {'signed_labels': tuple(labels)})
        )


def test_evaluator_defensively_copies_the_constructor_reviewer_map(
    quality_inputs,
) -> None:
    subjects = dict(SUBJECTS)
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=SECRET,
        reviewer_subject_hmacs=subjects,
    )
    subjects['reviewer_a'] = f'{999_002:064x}'

    report = evaluator._evaluate_approved(**quality_inputs)

    assert report.gate_outcome == 'green'


def test_evaluator_prevents_rebinding_the_frozen_reviewer_authority() -> None:
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=SECRET,
        reviewer_subject_hmacs=SUBJECTS,
    )

    with pytest.raises(AttributeError):
        evaluator._frozen_reviewer_subjects = (
            ('reviewer_a', f'{999_010:064x}'),
            ('reviewer_b', SUBJECTS['reviewer_b']),
            ('adjudicator_c', SUBJECTS['adjudicator_c']),
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('limits', None),
        ('clean_git_commit', 7),
    ],
)
def test_evaluator_bounds_malformed_manifest_top_level_values(
    quality_inputs, field, value
) -> None:
    manifest = replace(quality_inputs['manifest'], **{field: value})
    approval = replace(
        quality_inputs['approval'], manifest=manifest, limits=manifest.limits
    )

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('allowed_support_modes', ([],)),
        ('allowed_slot_ids', ([],)),
        ('relevant_serving_identity_hmacs', ([],)),
        ('required_serving_identity_hmacs', ([],)),
        ('query_embedding_required', 1),
        ('case_total_reserved_cost_usd', True),
    ],
)
def test_evaluator_bounds_malformed_manifest_case_leaf_values(
    quality_inputs, field, value
) -> None:
    cases = list(quality_inputs['manifest'].cases)
    cases[0] = replace(cases[0], **{field: value})
    manifest = replace(quality_inputs['manifest'], cases=tuple(cases))
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


def test_evaluator_requires_a_valid_nonempty_hmac_bound_corpus(
    quality_inputs,
) -> None:
    empty = replace(quality_inputs['corpus'], members=())
    forged = replace(quality_inputs['corpus'], corpus_snapshot_hmac=f'{999_003:064x}')

    for corpus in (empty, forged):
        approval = replace(quality_inputs['approval'], corpus=corpus)
        with pytest.raises(RagReleaseQualityError, match='corpus_drift'):
            _evaluate(quality_inputs | {'corpus': corpus, 'approval': approval})


def test_evaluator_rejects_the_superseded_corpus_policy_literal(
    quality_inputs,
) -> None:
    corpus = replace(
        quality_inputs['corpus'],
        pgvector_cosine_policy_version='rag-pgvector-cosine-distance:v1',
        corpus_snapshot_hmac=f'{999_005:064x}',
    )
    baseline = _baseline(quality_inputs['manifest'], corpus)
    approval = _approval(quality_inputs['manifest'], corpus, baseline)
    cases = _terminal_cases(quality_inputs['manifest'], corpus, approval)

    with pytest.raises(RagReleaseQualityError, match='corpus_drift'):
        _evaluate(
            {
                'terminal_cases': cases,
                'signed_labels': _labels(cases, quality_inputs['manifest'], approval),
                'baseline_metrics': baseline,
                'manifest': quality_inputs['manifest'],
                'corpus': corpus,
                'approval': approval,
            }
        )


@pytest.mark.parametrize(
    'field',
    ['relevant_serving_identity_hmacs', 'required_serving_identity_hmacs'],
)
def test_evaluator_requires_manifest_identities_to_exist_in_frozen_corpus(
    quality_inputs, field
) -> None:
    cases = list(quality_inputs['manifest'].cases)
    cases[0] = replace(cases[0], **{field: (f'{999_004:064x}',)})
    manifest = replace(quality_inputs['manifest'], cases=tuple(cases))
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


def test_evaluator_binds_manifest_support_modes_to_frozen_corpus_members(
    quality_inputs,
) -> None:
    cases = list(quality_inputs['manifest'].cases)
    cases[0] = replace(cases[0], allowed_support_modes=('source_observation',))
    manifest = replace(quality_inputs['manifest'], cases=tuple(cases))
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


@pytest.mark.parametrize(
    'outcome',
    [
        'budget_exceeded',
        'retriever_unavailable',
        'model_unavailable',
        'provider_usage_overrun',
        'model_provider_failed',
        'structured_output_invalid',
        'citation_validation_failed',
        'persistence_failed',
        'unexpected_internal_error',
        'provider_safety_unavailable',
        'provider_response_identity_invalid',
        'provider_embedding_payload_invalid',
        'live_corpus_snapshot_changed',
        'abandoned_unknown',
    ],
)
def test_evaluator_rejects_non_product_terminal_outcomes(
    quality_inputs, outcome
) -> None:
    values = _replace_case(quality_inputs, 1, outcome=outcome)

    with pytest.raises(RagReleaseQualityError, match='case_result_invalid'):
        _evaluate(values)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('reserved_cost_usd', 0),
        ('charged_cost_usd', True),
        ('charged_cost_usd', Decimal('0.0000001')),
        ('charged_cost_usd', Decimal('0.012001')),
        ('charged_cost_usd', Decimal('99.000000')),
        ('charged_cost_usd', Decimal('NaN')),
        ('query_embedding_dispatch_count', True),
        ('answer_generation_dispatch_count', True),
        ('outcome', []),
    ],
)
def test_evaluator_bounds_malformed_or_over_budget_case_accounting(
    quality_inputs, field, value
) -> None:
    values = _replace_case(quality_inputs, 2, **{field: value})

    with pytest.raises(RagReleaseQualityError, match='case_result_invalid'):
        _evaluate(values)


@pytest.mark.parametrize(
    ('field', 'value', 'code'),
    [
        ('terminal_cases', None, 'case_roster_invalid'),
        ('signed_labels', None, 'review_labels_invalid'),
        ('baseline_metrics', None, 'baseline_drift'),
        ('manifest', None, 'manifest_invalid'),
        ('corpus', None, 'corpus_drift'),
        ('approval', None, 'approval_drift'),
    ],
)
def test_evaluator_bounds_malformed_top_level_inputs(
    quality_inputs, field, value, code
) -> None:
    with pytest.raises(RagReleaseQualityError, match=code):
        _evaluate(quality_inputs | {field: value})


def test_evaluator_bounds_an_unhashable_signed_label(quality_inputs) -> None:
    labels = list(quality_inputs['signed_labels'])
    labels[0] = replace(labels[0], label=[])

    with pytest.raises(RagReleaseQualityError, match='review_labels_invalid'):
        _evaluate(quality_inputs | {'signed_labels': tuple(labels)})


@pytest.mark.parametrize(
    'mutator',
    [
        lambda executable: None,
        lambda executable: replace(executable, cases=()),
        lambda executable: replace(executable, cases=executable.cases[:-1]),
        lambda executable: replace(
            executable,
            cases=(executable.cases[1], executable.cases[0]) + executable.cases[2:],
        ),
        lambda executable: replace(
            executable,
            cases=(
                executable.cases[0],
                replace(
                    executable.cases[1],
                    case_id_hmac=executable.cases[0].case_id_hmac,
                ),
            )
            + executable.cases[2:],
        ),
        lambda executable: replace(
            executable,
            cases=(
                replace(executable.cases[0], retrieval_query_hmac=f'{998_001:064x}'),
            )
            + executable.cases[1:],
        ),
        lambda executable: replace(executable, source_manifest_hmac=f'{998_002:064x}'),
    ],
)
def test_evaluator_requires_the_complete_resolved_executable_manifest(
    quality_inputs, mutator
) -> None:
    manifest = replace(
        quality_inputs['manifest'],
        executable=mutator(quality_inputs['manifest'].executable),
    )
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


def test_evaluator_binds_executable_component_reserves_to_annotations(
    quality_inputs,
) -> None:
    executable = quality_inputs['manifest'].executable
    first = executable.cases[0]
    components = list(first.components)
    components[1] = replace(components[1], reserved_cost_usd=Decimal('0.011000'))
    changed = replace(first, components=tuple(components))
    manifest = replace(
        quality_inputs['manifest'],
        executable=replace(executable, cases=(changed,) + executable.cases[1:]),
    )
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


@pytest.mark.parametrize(
    'mutator',
    [
        lambda executable: replace(
            executable,
            runtime_contract_version=_TextAlias(executable.runtime_contract_version),
        ),
        lambda executable: replace(
            executable,
            cases=(
                replace(
                    executable.cases[0],
                    current_text_hmac=_TextAlias(executable.cases[0].current_text_hmac),
                ),
            )
            + executable.cases[1:],
        ),
        lambda executable: replace(
            executable,
            cases=(
                replace(
                    executable.cases[0],
                    components=(
                        replace(
                            executable.cases[0].components[0],
                            policy=replace(
                                executable.cases[0].components[0].policy,
                                provider=_TextAlias('openai'),
                            ),
                        ),
                        executable.cases[0].components[1],
                    ),
                ),
            )
            + executable.cases[1:],
        ),
        lambda executable: replace(
            executable,
            cases=(
                replace(
                    executable.cases[0],
                    components=(
                        replace(
                            executable.cases[0].components[0],
                            reserved_input_tokens=True,
                        ),
                        executable.cases[0].components[1],
                    ),
                ),
            )
            + executable.cases[1:],
        ),
        lambda executable: replace(
            executable,
            cases=(
                replace(
                    executable.cases[0],
                    components=tuple(reversed(executable.cases[0].components)),
                ),
            )
            + executable.cases[1:],
        ),
    ],
)
def test_evaluator_requires_canonical_executable_runtime_and_input_identities(
    quality_inputs, mutator
) -> None:
    manifest = replace(
        quality_inputs['manifest'],
        executable=mutator(quality_inputs['manifest'].executable),
    )
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('reserved_cost_usd', Decimal('0.012')),
        ('charged_cost_usd', Decimal('0.012')),
        ('charged_cost_usd', Decimal('-0.000000')),
        ('charged_cost_usd', Decimal('NaN')),
        ('charged_cost_usd', Decimal('Infinity')),
        ('charged_cost_usd', True),
    ],
)
def test_evaluator_rejects_noncanonical_case_cost_decimals(
    quality_inputs, field, value
) -> None:
    values = _replace_case(quality_inputs, 2, **{field: value})

    with pytest.raises(RagReleaseQualityError, match='case_result_invalid'):
        _evaluate(values)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('query_embedding_reserved_cost_usd', Decimal('0')),
        ('query_embedding_reserved_cost_usd', Decimal('-0.000000')),
        ('answer_generation_reserved_cost_usd', Decimal('0.012')),
        ('case_total_reserved_cost_usd', Decimal('0.012')),
        ('case_total_reserved_cost_usd', Decimal('-0.000000')),
    ],
)
def test_evaluator_rejects_noncanonical_manifest_cost_decimals(
    quality_inputs, field, value
) -> None:
    cases = list(quality_inputs['manifest'].cases)
    cases[0] = replace(cases[0], **{field: value})
    manifest = replace(quality_inputs['manifest'], cases=tuple(cases))
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


@pytest.mark.parametrize(
    'value',
    [
        Decimal('0.000'),
        Decimal('-0.000000'),
        Decimal('NaN'),
        Decimal('Infinity'),
        True,
    ],
)
def test_evaluator_rejects_noncanonical_executable_cost_scale(
    quality_inputs, value
) -> None:
    executable = quality_inputs['manifest'].executable
    first = executable.cases[0]
    components = list(first.components)
    components[0] = replace(components[0], reserved_cost_usd=value)
    changed = replace(first, components=tuple(components))
    manifest = replace(
        quality_inputs['manifest'],
        executable=replace(executable, cases=(changed,) + executable.cases[1:]),
    )
    approval = replace(quality_inputs['approval'], manifest=manifest)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


def test_evaluator_rejects_noncanonical_limit_decimal_scale(
    quality_inputs,
) -> None:
    limits = replace(
        quality_inputs['manifest'].limits,
        case_max_cost_usd=Decimal('0.012'),
    )
    manifest = replace(quality_inputs['manifest'], limits=limits)
    approval = replace(quality_inputs['approval'], manifest=manifest, limits=limits)

    with pytest.raises(RagReleaseQualityError, match='manifest_invalid'):
        _evaluate(quality_inputs | {'manifest': manifest, 'approval': approval})


@pytest.mark.parametrize(
    'field',
    ['case_kind', 'surface', 'configured_backend'],
)
def test_evaluator_rejects_result_primitive_subclasses(quality_inputs, field) -> None:
    original = getattr(quality_inputs['terminal_cases'][0], field)
    values = _replace_case(quality_inputs, 0, **{field: _TextAlias(original)})

    with pytest.raises(RagReleaseQualityError, match='case_result_invalid'):
        _evaluate(values)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('state', _TextAlias('complete')),
        ('outcome', _TextAlias('supported')),
        ('runtime_agent_run_id_hmac', _TextAlias(f'{3100:064x}')),
        ('case_projection_hmac', _TextAlias(f'{4100:064x}')),
        ('current_corpus_snapshot_hmac', _TextAlias(f'{301:064x}')),
        ('provider_safety_snapshot_hmac', _TextAlias(f'{1008:064x}')),
        ('query_embedding_dispatch_count', _IntegerAlias(1)),
        ('answer_generation_dispatch_count', _IntegerAlias(1)),
        ('legacy_precision_numerator', _IntegerAlias(1)),
        ('expected_no_answer', _IntegerAlias(0)),
        ('hard_negative_correct', _IntegerAlias(0)),
        ('positive_answered', _IntegerAlias(1)),
        ('required_slot_covered', _IntegerAlias(1)),
    ],
)
def test_evaluator_validates_all_case_primitives_before_comparison(
    quality_inputs, field, value
) -> None:
    values = _replace_case(quality_inputs, 0, **{field: value})

    with pytest.raises(RagReleaseQualityError, match='case_result_invalid'):
        _evaluate(values)


@pytest.mark.parametrize(
    'malformed',
    [
        (('reviewer_a',),),
        (['reviewer_a'],),
        {'reviewer_a': SUBJECTS['reviewer_a']},
        None,
    ],
)
def test_evaluator_bounds_forced_malformed_frozen_reviewer_rosters(
    quality_inputs, malformed
) -> None:
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=SECRET,
        reviewer_subject_hmacs=SUBJECTS,
    )
    object.__setattr__(evaluator, '_frozen_reviewer_subjects', malformed)

    with pytest.raises(RagReleaseQualityError, match='reviewer_roster_invalid'):
        evaluator._evaluate_approved(**quality_inputs)


@pytest.mark.parametrize(
    'attribute',
    [
        '_identity_secret',
        '_reviewer_subjects',
        '_frozen_reviewer_subjects',
        '_reviewer_roster_hmac',
    ],
)
def test_evaluator_forbids_deleting_reviewer_authority_slots(attribute) -> None:
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=SECRET,
        reviewer_subject_hmacs=SUBJECTS,
    )

    with pytest.raises(AttributeError, match='immutable'):
        delattr(evaluator, attribute)


def test_evaluator_bounds_forced_missing_frozen_reviewer_roster(
    quality_inputs,
) -> None:
    evaluator = RagReleaseQualityEvaluator(
        identity_secret=SECRET,
        reviewer_subject_hmacs=SUBJECTS,
    )
    object.__delattr__(evaluator, '_frozen_reviewer_subjects')

    with pytest.raises(RagReleaseQualityError, match='reviewer_roster_invalid'):
        evaluator._evaluate_approved(**quality_inputs)


@pytest.mark.parametrize(
    'field',
    [
        'precision_numerator',
        'precision_denominator',
        'recall_numerator',
        'recall_denominator',
    ],
)
def test_evaluator_reports_malformed_baseline_ratios_as_baseline_drift(
    quality_inputs, field
) -> None:
    baseline = replace(quality_inputs['baseline_metrics'], **{field: True})

    with pytest.raises(RagReleaseQualityError, match='baseline_drift'):
        _evaluate(quality_inputs | {'baseline_metrics': baseline})


def test_evaluator_rejects_reviewer_role_key_subclasses() -> None:
    subjects = dict(SUBJECTS)
    reviewer_a = subjects.pop('reviewer_a')
    subjects[_TextAlias('reviewer_a')] = reviewer_a

    with pytest.raises(RagReleaseQualityError, match='reviewer_roster_invalid'):
        RagReleaseQualityEvaluator(
            identity_secret=SECRET,
            reviewer_subject_hmacs=subjects,
        )
