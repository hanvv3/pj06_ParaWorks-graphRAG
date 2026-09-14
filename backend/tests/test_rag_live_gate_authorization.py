from __future__ import annotations

import hashlib
import hmac
import io
import json
from copy import copy, deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, event, insert, select

from backend.app.admin import rag_live_gate as cli
from backend.app.rag import release_review as review
from backend.tests.test_rag_live_gate_preview import (
    SECRET,
    SnapshotReader,
    committed_repo,
    fingerprint,
    preview_inputs,
    utf8,
)
from backend.tests.test_rag_release_ledger import (
    _deterministic_non_product_database_seam,  # noqa: F401
)
from backend.tests.test_rag_release_reviewer import (
    authenticated_roster,
    encoded,
    make_reviewer_harness,
)

REVIEW_SECRET = b'task24-b-external-test-review-material'
APPROVAL_ID = UUID('77777777-7777-4777-8777-777777777777')


def expected_approval(preview, approval_id=APPROVAL_ID):
    p = json.loads(preview.canonical_bytes)
    keys = (
        'approval_base_generation',
        'approved_corpus_snapshot_hmac',
        'approved_provider_safety_snapshot_hmac',
        'assistant_context_distribution',
        'baseline_hmac',
        'clean_git_commit',
        'distribution',
        'fixture_manifest_hmac',
        'fixture_manifest_path',
        'fixture_manifest_sha256',
        'fixture_manifest_version',
        'ledger_epoch',
        'ledger_uuid',
        'limits',
        'live_gate_contract_version',
        'approval_base_release_marker_file_digest',
        'reviewer_roster_hmac',
        'rubric_version',
        'validation_database_identity_hmac',
    )
    return {key: p[key] for key in keys} | {
        'approval_id_hmac': fingerprint(
            {'approval_id': str(approval_id)}, 'rag-live-approval-id:v1'
        ),
        'designated_environment_id_bytes': utf8('isolated-validation'),
        'designated_host_id_bytes': utf8('isolated-host'),
    }


def signed_record(h, *, change=None):
    # External review is simulated only here; production exposes no signer.
    approval = expected_approval(h.preview, h.approval_id)
    context = {
        'preview_hmac': h.preview.preview_hmac,
        'approval_preimage': approval,
        'fingerprint_key_version': 'v1',
        'fingerprint_key_material_verifier': h.reader.value.release.fingerprint_key_material_verifier,
    }
    signed = {
        'schema_version': 'rag-release-admin-review:v1',
        'operation': 'authorization-bootstrap',
        'actor_subject_hmac': h.proofs[0].reviewer_subject_hmac,
        'expected_context': context,
        'implementation_plan_reference_hmac': '0' * 64,
        'nonce': str(h.approval_id),
        'rebootstrap_reason_hmac': None,
        'review_authority_key_id': 'test-external-review',
        'target': h.target.review_identity,
    }
    if change:
        change(signed)
    signature = hmac.new(
        REVIEW_SECRET,
        b'paraworks:rag-release-admin-review:v1\x00' + encoded(signed),
        hashlib.sha256,
    ).hexdigest()
    return encoded({'signed_payload': signed, 'hmac_sha256': signature})


def authorization_harness(tmp_path):
    cls = getattr(review, 'RagLiveGateAuthorizer', None)
    assert cls is not None, 'Task24-B authorization constructor is missing'
    repo, commit = committed_repo(tmp_path)
    rh = make_reviewer_harness()
    proofs = authenticated_roster(rh)
    engine = create_engine('sqlite://')
    target = cli.RagReleaseAdminTarget.build(
        database_url='postgresql://localhost/task24_b_validation',
        marker_path=tmp_path / 'release-marker',
        provider_safety_latch_path=tmp_path / 'provider-marker',
        designated_environment_id='isolated-validation',
        designated_host_id='isolated-host',
        review_secret=REVIEW_SECRET,
    )
    # Only SQL/locking transport is SQLite; keep the selected connection locator
    # explicit so production must compare the pinned engine and signed target.
    from sqlalchemy.engine import make_url

    engine.url = make_url(target.database_url)
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
    from backend.app.rag.release_authority import (
        RagReleaseAuthority,
        ValidationDatabaseIdentity,
    )
    from backend.tests.test_rag_release_ledger import _TestProviderPeer

    DurableFileAuthority(target.provider_safety_latch_path).write(
        {'provider': 'test-only'}
    )
    authority = RagReleaseAuthority(
        marker_path=target.marker_path,
        provider_safety_latch_path=target.provider_safety_latch_path,
        identity_secret=SECRET,
        fingerprint_key_version='v1',
        designated_environment_id='isolated-validation',
        designated_host_id='isolated-host',
        provider_safety_release_peer=_TestProviderPeer(),
    )
    database_identity = ValidationDatabaseIdentity('task24_b_validation', 53)
    with engine.connect() as connection:
        initial = authority.initialize(
            connection,
            database_identity=database_identity,
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    inputs = replace(
        preview_inputs(),
        release=initial,
        reviewer_roster_hmac=rh.verifier.reviewer_roster_hmac(proofs),
    )
    reader = SnapshotReader(inputs)
    reader.engine, reader.authority, reader.database_identity = (
        engine,
        authority,
        database_identity,
    )
    from contextlib import contextmanager

    from backend.app.models.rag_runtime import (
        RagProviderReadiness,
        RagProviderSafetyAuthority,
    )
    from backend.app.models.rag_serving import RagServingCorpusGeneration
    from backend.tests.release_ledger_fixtures import provider_rows

    body = json.loads(inputs.provider_snapshot_bytes)
    with engine.begin() as connection:
        RagServingCorpusGeneration.__table__.create(connection)
        connection.execute(
            insert(RagServingCorpusGeneration).values(
                id=1,
                corpus_generation=inputs.corpus.corpus_generation,
                vector_index_generation=inputs.corpus.vector_index_generation,
                embedding_model='text-embedding-3-small',
                embedding_dimensions=1536,
                index_policy_version='rag-v2-serving-index:v1',
                pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier=initial.fingerprint_key_material_verifier,
            )
        )
        for model in (RagProviderSafetyAuthority, RagProviderReadiness):
            model.__table__.create(connection, checkfirst=True)
        for key, row in provider_rows():
            if key.row_kind == 'provider_safety_authority':
                row.update(
                    {
                        field: body[field]
                        for field in (
                            'authority_uuid',
                            'designated_environment_id',
                            'global_safety_generation',
                            'envelope_digest',
                            'fingerprint_key_version',
                            'fingerprint_key_material_verifier',
                        )
                    }
                )
                table = RagProviderSafetyAuthority.__table__
            else:
                row.update(
                    next(
                        f
                        for f in body['active_families']
                        if f['component'] == row['component']
                    )
                )
                row['authorized_fingerprint_key_material_verifier'] = body[
                    'fingerprint_key_material_verifier'
                ]
                table = RagProviderReadiness.__table__
            connection.execute(insert(table).values(**row))
    reader.runtime_lock_count = 0

    @contextmanager
    def locked_approved(connection, *, barrier_guard):
        assert connection.engine is engine
        reader.runtime_lock_count += 1
        authority._assert_barrier_guard(barrier_guard, connection)
        with reader.locked():
            reader.barrier_guard = barrier_guard
            try:
                yield reader
            finally:
                reader.barrier_guard = None

    reader.locked_approved = locked_approved
    preview = review.build_live_gate_preview(
        repository=repo,
        expected_commit=commit,
        snapshot_reader=reader,
        identity_secret=SECRET,
    )
    options = {
        'repository': repo,
        'expected_commit': commit,
        'snapshot_reader': reader,
        'reviewer_verifier': rh.verifier,
        'identity_secret': SECRET,
        'target': target,
        'review_secret': REVIEW_SECRET,
        'review_key_id': 'test-external-review',
        'review_key_registry': {
            'test-external-review': cli.release_review_key_material_verifier(
                REVIEW_SECRET
            )
        },
        'implementation_plan_reference_hmac': '0' * 64,
    }
    authorizer = cls(**options)
    return SimpleNamespace(
        repo=repo,
        commit=commit,
        reader=reader,
        engine=engine,
        rh=rh,
        proofs=proofs,
        target=target,
        preview=preview,
        authorizer=authorizer,
        options=options,
        approval_id=uuid4(),
    )


def authorize(h, record=None, preview=None, proofs=None):
    return h.authorizer.authorize(
        preview=h.preview if preview is None else preview,
        reviewers=h.proofs if proofs is None else proofs,
        approval_record=signed_record(h) if record is None else record,
    )


def test_authorization_binds_exact_preimage_and_is_single_use(tmp_path):
    h = authorization_harness(tmp_path)
    authorization = authorize(h)
    p = expected_approval(h.preview, h.approval_id)
    assert authorization.approval_id_hmac == p['approval_id_hmac']
    assert authorization.approval_hmac == fingerprint(
        p, 'rag-live-execution-approval:v1'
    )
    assert (
        authorization.manifest.manifest_hmac == h.preview.manifest.fixture_manifest_hmac
    )
    assert authorization.ledger_uuid == h.reader.value.release.ledger_uuid
    assert authorization.authorization_state == 'unused'
    assert authorization.limits.total_dispatches == 40
    assert h.authorizer.approved_source(authorization) is not None
    with pytest.raises(review.LiveGatePreviewError):
        authorize(h)
    for invalid in (
        copy(authorization),
        replace(authorization, baseline_hmac='e' * 64),
    ):
        with pytest.raises(review.LiveGatePreviewError):
            h.authorizer.approved_source(invalid)


@pytest.mark.parametrize(
    'attack',
    [
        'preview_hmac',
        'canonical_bytes',
        'manifest',
        'corpus',
        'baseline',
        'limits',
        'source',
        'reviewers',
        'dirty',
        'key',
        'provider',
        'snapshot_corpus',
        'epoch',
        'generation',
        'marker',
        'host',
        'environment',
        'database',
        'fixture',
        'commit',
        'plan',
        'roster',
    ],
)
def test_authorization_drift_is_zero_call_zero_mutation(tmp_path, monkeypatch, attack):
    h = authorization_harness(tmp_path)
    before_calls = len(h.rh.calls)
    record = signed_record(h)
    p, proofs = h.preview, h.proofs
    if attack in ('preview_hmac', 'canonical_bytes', 'baseline', 'source'):
        key, value = {
            'preview_hmac': ('preview_hmac', 'e' * 64),
            'canonical_bytes': ('canonical_bytes', b'{}'),
            'baseline': ('baseline_hmac', 'e' * 64),
            'source': ('source_binding', None),
        }[attack]
        p = replace(p, **{key: value})
    if attack == 'manifest':
        p = replace(p, manifest=replace(p.manifest, manifest_hmac='e' * 64))
    if attack == 'corpus':
        p = replace(p, corpus=replace(p.corpus, corpus_generation=99))
    if attack == 'limits':
        limit = deepcopy(p.limits)
        object.__setattr__(limit, 'total_dispatches', True)
        p = replace(p, limits=limit)
    if attack == 'reviewers':
        proofs = (proofs[1], proofs[0], proofs[2])
    if attack == 'dirty':
        (h.repo / 'untracked.py').write_text('dirty')
    if attack == 'fixture':
        (h.repo / review.LIVE_FIXTURE_PATH).write_text('{}')
    if attack == 'commit':
        import subprocess

        subprocess.run(
            [
                'git',
                '-C',
                str(h.repo),
                '-c',
                'user.name=test',
                '-c',
                'user.email=test@example.invalid',
                'commit',
                '--allow-empty',
                '-qm',
                'drift',
            ],
            check=True,
            capture_output=True,
        )
    if attack in (
        'key',
        'epoch',
        'generation',
        'marker',
        'host',
        'environment',
        'database',
    ):
        key, value = {
            'key': ('fingerprint_key_version', 'v2'),
            'epoch': ('ledger_epoch', 2),
            'generation': ('generation', 1),
            'marker': ('marker_file_digest', 'e' * 64),
            'host': ('designated_host_id_hmac', 'e' * 64),
            'environment': ('designated_environment_id_hmac', 'e' * 64),
            'database': ('validation_database_identity_hmac', 'e' * 64),
        }[attack]
        h.reader.value = replace(
            h.reader.value, release=replace(h.reader.value.release, **{key: value})
        )
    if attack == 'provider':
        h.reader.value = replace(
            h.reader.value, approved_provider_safety_snapshot_hmac='e' * 64
        )
    if attack == 'snapshot_corpus':
        h.reader.value = replace(
            h.reader.value, corpus=replace(h.reader.value.corpus, corpus_generation=99)
        )
    if attack == 'plan':
        h.reader.value = replace(
            h.reader.value, implementation_plan_reference_hmac='e' * 64
        )
    if attack == 'roster':
        h.reader.value = replace(h.reader.value, reviewer_roster_hmac='e' * 64)
    writes = []
    event.listen(
        h.engine, 'before_cursor_execute', lambda _c, _cu, sql, *_: writes.append(sql)
    )
    with pytest.raises(review.LiveGatePreviewError):
        authorize(h, record, p, proofs)
    assert not any(
        sql.lstrip()
        .lower()
        .startswith(('insert', 'update', 'delete', 'create', 'drop', 'alter'))
        for sql in writes
    )
    assert len(h.rh.calls) == before_calls


@pytest.mark.parametrize(
    'attack',
    [
        'signature',
        'whitespace',
        'wrong_preview',
        'wrong_approval',
        'wrong_actor',
        'wrong_target',
        'wrong_key',
        'reason',
        'plan_only',
        'balance_only',
    ],
)
def test_only_external_exact_canonical_approval_can_authorize(tmp_path, attack):
    h = authorization_harness(tmp_path)
    changes = {
        'wrong_preview': lambda p: p['expected_context'].update(preview_hmac='e' * 64),
        'wrong_approval': lambda p: p['expected_context']['approval_preimage'].update(
            ledger_epoch=2
        ),
        'wrong_actor': lambda p: p.update(actor_subject_hmac=True),
        'wrong_target': lambda p: p['target'].update(designated_host_id_hmac='e' * 64),
        'wrong_key': lambda p: p.update(review_authority_key_id='unregistered'),
        'reason': lambda p: p.update(rebootstrap_reason_hmac='e' * 64),
    }
    raw = signed_record(h, change=changes.get(attack))
    if attack == 'signature':
        raw = encoded(json.loads(raw) | {'hmac_sha256': 'e' * 64})
    if attack == 'whitespace':
        raw += b'\n'
    if attack == 'plan_only':
        raw = b'plan approved'
    if attack == 'balance_only':
        raw = b'USD 100 available'
    with pytest.raises(review.LiveGatePreviewError):
        authorize(h, raw)


def test_authorization_cli_refuses_before_authority_or_approval_read():
    class NeverRead:
        def read(self, *args):
            raise AssertionError('approval was read before snapshot-reader readiness')

    def never():
        raise AssertionError('authority opened before snapshot-reader readiness')

    outcome = cli._run_cli(
        ['authorization-bootstrap'], stdin=NeverRead(), service_factory=never
    )
    assert outcome.payload == {
        'code': 'committed_source_changed',
        'ok': False,
        'provider_dispatch_count': 0,
        'authorization_issued': False,
    }


def test_cli_does_not_use_secret_environment_or_create_self_approval(
    monkeypatch, capsys
):
    for name in (
        'GOOGLE_CODE',
        'GOOGLE_TOKEN',
        'GOOGLE_STATE',
        'RAG_APPROVAL',
        'RAG_LIVE_APPROVAL',
    ):
        monkeypatch.setenv(name, 'never-use-or-print-this-secret')
    monkeypatch.setattr(
        cli,
        '_build_default_resources',
        lambda *_: pytest.fail('unready command opened authority'),
    )
    assert cli.main(['authorization-bootstrap']) == 2
    output = capsys.readouterr().out
    assert json.loads(output)['code'] == 'committed_source_changed'
    assert 'never-use-or-print-this-secret' not in output


def test_external_execution_approver_need_not_be_a_frozen_quality_reviewer(tmp_path):
    h = authorization_harness(tmp_path)
    raw = signed_record(h, change=lambda p: p.update(actor_subject_hmac='e' * 64))
    assert authorize(h, raw).authorization_state == 'unused'


@pytest.mark.parametrize(
    'field',
    ['marker_path', 'provider_safety_latch_path', 'database_url', 'review_identity'],
)
def test_signed_target_must_equal_pinned_real_resources(tmp_path, field):
    h = authorization_harness(tmp_path)
    if field == 'review_identity':
        h.target = replace(
            h.target,
            review_identity=h.target.review_identity | {'marker_target_hmac': 'e' * 64},
        )
    else:
        options = {
            k: getattr(h.target, k)
            for k in (
                'marker_path',
                'provider_safety_latch_path',
                'database_url',
                'designated_environment_id',
                'designated_host_id',
            )
        }
        options[field] = (
            'postgresql://localhost/another_validation'
            if field == 'database_url'
            else tmp_path / 'other-marker'
        )
        h.target = cli.RagReleaseAdminTarget.build(
            **options, review_secret=REVIEW_SECRET
        )
    h.options['target'] = h.target
    h.authorizer = review.RagLiveGateAuthorizer(**h.options)
    with pytest.raises(review.LiveGatePreviewError):
        authorize(h)


def test_review_key_rotation_in_final_snapshot_read_refuses_issue(tmp_path):
    h = authorization_harness(tmp_path)
    read = h.reader.read
    start = h.reader.reads

    def rotate_on_last_read():
        value = read()
        if h.reader.reads == start + 4:
            h.options['review_key_registry'].clear()
        return value

    h.reader.read = rotate_on_last_read
    with pytest.raises(review.LiveGatePreviewError):
        authorize(h)


@pytest.mark.parametrize(
    'attack', ['marker', 'db_authorization', 'db_provider', 'missing_barrier']
)
def test_authorization_checks_actual_authority_peers_not_only_reader_values(
    tmp_path, attack
):
    from sqlalchemy import update

    h = authorization_harness(tmp_path)
    if attack == 'marker':
        h.target.marker_path.write_bytes(h.target.marker_path.read_bytes() + b' ')
    elif attack == 'db_authorization':
        from backend.app.rag.release_schema import (
            build_rag_release_metadata,
            release_tables,
        )

        with h.engine.begin() as connection:
            connection.execute(
                update(release_tables(build_rag_release_metadata()).ledgers).values(
                    generation=2, last_transition_digest='a' * 64
                )
            )
    elif attack == 'db_provider':
        h.reader.value = replace(
            h.reader.value,
            provider_snapshot_bytes=h.reader.value.provider_snapshot_bytes,
        )
        # An in-memory body cannot replace missing authoritative SQL rows.
        from backend.app.models.rag_runtime import RagProviderSafetyAuthority

        with h.engine.begin() as connection:
            RagProviderSafetyAuthority.__table__.drop(connection, checkfirst=True)
    else:
        h.reader.locked_approved = None
    with pytest.raises(review.LiveGatePreviewError):
        authorize(h)


@pytest.mark.parametrize(
    'attack', ['reviewer', 'key_during_read', 'marker_during_read']
)
def test_source_rechecks_current_reviewer_key_and_marker_after_reader_callbacks(
    tmp_path, attack
):
    from backend.app.rag.release_ledger import RagReleaseLedgerError

    h = runtime_source_harness(tmp_path)
    if attack == 'reviewer':
        h.rh.users['reviewer_b'].external_id = 'new-subject'
    else:
        read = h.reader.read

        def change_during_read():
            value = read()
            if attack == 'key_during_read':
                h.options['review_key_registry'].clear()
            else:
                h.target.marker_path.write_bytes(
                    h.target.marker_path.read_bytes() + b' '
                )
            return value

        h.reader.read = change_during_read
    with pytest.raises(RagReleaseLedgerError):
        require_source(h)


@pytest.mark.parametrize(
    'argv',
    [
        ['authorization-bootstrap', '--code', 'secret'],
        ['authorization-bootstrap', '--state', 'secret'],
        ['authorization-bootstrap', '--token', 'secret'],
        ['approval-sign'],
        ['approve'],
        ['sign'],
    ],
)
def test_cli_has_no_secret_args_or_self_signing(argv):
    outcome = cli._run_cli(
        argv,
        stdin=io.BytesIO(),
        service_factory=lambda: pytest.fail('unexpected service'),
    )
    assert outcome.exit_code == 2 and outcome.payload == {
        'code': 'command_refused',
        'ok': False,
    }


def runtime_source_harness(tmp_path):
    from decimal import Decimal

    from backend.app.rag.release_schema import (
        build_rag_release_metadata,
        release_tables,
    )

    h = authorization_harness(tmp_path)
    h.authorization = authorize(h)
    h.source = h.authorizer.approved_source(h.authorization)
    a = h.authorization
    p = json.loads(h.preview.canonical_bytes)
    metadata = build_rag_release_metadata()
    metadata.create_all(h.engine)
    h.tables = release_tables(metadata)
    h.auth_row = {
        'ledger_uuid': str(a.ledger_uuid),
        'ledger_epoch': a.ledger_epoch,
        'approval_id_hmac': a.approval_id_hmac,
        'approval_hmac': a.approval_hmac,
        'base_generation': a.approval_base_generation,
        'state': 'unused',
        'approved_corpus_snapshot_hmac': a.corpus.corpus_snapshot_hmac,
        'approved_provider_safety_snapshot_hmac': a.approved_provider_safety_snapshot_hmac,
        'provider_safety_envelope_digest': p['provider_safety_envelope_digest'],
        'validation_database_identity_hmac': a.validation_database_identity_hmac,
        'manifest_hmac': a.manifest.manifest_hmac,
        'baseline_hmac': a.baseline_hmac,
        'reviewer_roster_hmac': a.reviewer_roster_hmac,
        'execution_process_instance_hmac': None,
        'execution_runner_fence_hmac': None,
        'case_claim_count': 0,
        'embedding_dispatch_count': 0,
        'generation_dispatch_count': 0,
        'total_dispatch_count': 0,
        'reserved_cost_usd': Decimal('0.000000'),
        'charged_cost_usd': Decimal('0.000000'),
    }
    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        ReleaseRowPrimaryKey,
        _derived_payload_rows,
    )
    from backend.tests.release_ledger_fixtures import (
        observation,
        observe_provider_fixture,
        row_key,
    )
    from backend.tests.test_rag_release_ledger import _bootstrap_payload

    ledger = RagReleaseLedger(authority=h.reader.authority, identity_secret=SECRET)
    payload = _bootstrap_payload(h.reader.value.release)
    payload.update(
        {
            key: h.auth_row[key]
            for key in (
                'approval_id_hmac',
                'approval_hmac',
                'approved_corpus_snapshot_hmac',
                'approved_provider_safety_snapshot_hmac',
            )
        }
    )
    payload['current_corpus_snapshot_hmac'] = h.auth_row[
        'approved_corpus_snapshot_hmac'
    ]
    payload['affected_rows'] = [
        {'row_kind': kind, 'row_identity_hmac': digest}
        for kind, digest in _derived_payload_rows(payload, identity_secret=SECRET)
    ]
    with h.engine.connect() as connection:
        mutations = ledger.mutation_set(connection)
        observe_provider_fixture(connection, mutations)
        payload['observation_set'] = []
        mutations._observation_rows.clear()
        for kind in ('provider_safety_authority', 'provider_readiness'):
            table, _aliases = mutations._table(kind)
            for row in connection.execute(select(table)).mappings():
                key = row_key(kind, dict(row))
                mutations.observe(key)
                payload['observation_set'].append(observation(key, dict(row), SECRET))
        payload['observation_set'].sort(
            key=lambda item: (item['row_kind'], item['row_identity_hmac'])
        )
        mutations.plan(
            insert(h.tables.authorizations).values(**h.auth_row),
            ReleaseRowPrimaryKey(
                'authorization',
                {
                    key: h.auth_row[key]
                    for key in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
                },
            ),
        )
        current = ledger.append(
            connection,
            payload,
            actual_mutations=mutations,
            database_identity=h.reader.database_identity,
        )
    h.reader.value = replace(h.reader.value, release=current)
    h.ledger, h.base = ledger, payload
    h.reader.runtime_lock_count = 0
    return h


@pytest.mark.parametrize(
    'stage',
    [
        'reader_exit',
        'reader_enter',
        'read',
        'read_hard_negative_oracle',
        'read_pgvector_baseline_members',
        'auth_user_reader',
        'clock',
        'property',
        'provider',
        'transport',
    ],
)
@pytest.mark.parametrize(
    'attack', ['payload_scalar', 'payload_tree', 'mutation_images']
)
def test_append_owns_inputs_before_reader_teardown(
    tmp_path, monkeypatch, attack, stage
):
    """A retained input may not inject comparisons that commit unaudited rows."""
    from contextlib import contextmanager

    from backend.app.models.agent_runs import AgentRun
    from backend.app.models.rag_runtime import AgentRunCostComponent
    from backend.app.rag.release_ledger import RagReleaseLedger

    phase = {
        'payload': None,
        'mutations': None,
        'dml': [],
        'calls': [],
        'attacked': 0,
        'late_callbacks': [],
    }
    kept = []
    original_harness = runtime_source_harness
    original_append = RagReleaseLedger.append

    class LateString(str):
        __hash__ = str.__hash__

        def fire(self):
            if phase['dml'] and not phase['calls']:
                phase['calls'].append('commit')
                phase['connection'].commit()

        def __eq__(self, other):
            self.fire()
            return str.__eq__(self, other)

        def __ne__(self, other):
            self.fire()
            return str.__ne__(self, other)

    def poison_tree(value):
        if type(value) is dict:
            for key, item in tuple(value.items()):
                poison_tree(item)
                value[key] = LateString(item) if type(item) is str else None
        elif type(value) is list:
            for item in value:
                poison_tree(item)
            value.clear()

    def callback(kind):
        if phase['dml']:
            phase['late_callbacks'].append(kind)
        if phase['payload'] is not None and kind == stage:
            phase['attacked'] += 1
            if phase['attacked'] == 1:
                if attack == 'payload_scalar':
                    phase['payload']['authorization_state_after'] = LateString(
                        'started'
                    )
                elif attack == 'payload_tree':
                    poison_tree(phase['payload'])
                elif attack in {'schema_processor', 'schema_result_processor'}:
                    from sqlalchemy import Numeric, TypeDecorator

                    class CallbackNumeric(TypeDecorator):
                        impl = Numeric(18, 6)
                        cache_ok = False

                        @property
                        def python_type(self):
                            from decimal import Decimal

                            return Decimal

                        def process_bind_param(self, value, dialect):
                            if (
                                attack == 'schema_processor'
                                and phase['dml']
                                and not phase['calls']
                            ):
                                phase['calls'].append('commit')
                                phase['connection'].commit()
                            return value

                        def process_result_value(self, value, dialect):
                            if (
                                attack == 'schema_result_processor'
                                and phase['dml']
                                and phase['connection'].in_transaction()
                                and not phase['calls']
                            ):
                                phase['calls'].append('commit')
                                phase['connection'].commit()
                            return value

                    plan = next(
                        plan
                        for plan in phase['mutations']._plans
                        if plan.row.row_kind == 'agent_run'
                    )
                    monkeypatch.setattr(
                        plan.statement.table.c.total_charged_cost_usd,
                        'type',
                        CallbackNumeric(),
                    )
                else:
                    for plan in phase['mutations']._frozen_plans or ():
                        if plan.row.row_kind == 'authorization':
                            plan.after['state'] = LateString('started')
                    for plan in phase['mutations']._plans:
                        poison_tree(plan.row.primary_key)
                        for binding in (plan.statement._values or {}).values():
                            binding.value = LateString('started')
                    for row in phase['mutations']._observation_rows:
                        poison_tree(row.primary_key)
                    for field in (
                        '_plans',
                        '_rows',
                        '_observation_rows',
                        '_before_snapshots',
                        '_after_snapshots',
                        '_observation_snapshots',
                    ):
                        getattr(phase['mutations'], field).clear()
                    phase['mutations']._frozen_plans = ()

    def harness(path):
        h = original_harness(path)
        kept.append(h)
        original_locked = h.reader.locked

        @contextmanager
        def locked():
            callback('reader_enter')
            with original_locked() as value:
                yield value
            callback('reader_exit')

        h.reader.locked = locked
        h.rh.callback = callback
        for method in (
            'read',
            'read_hard_negative_oracle',
            'read_pgvector_baseline_members',
        ):
            original_method = getattr(h.reader, method)

            def method_callback(
                *args, _original=original_method, _method=method, **kwargs
            ):
                callback(_method)
                return _original(*args, **kwargs)

            setattr(h.reader, method, method_callback)

        class CurrentUser:
            def __init__(self, user):
                self.values = vars(user).copy()

            def __getattr__(self, key):
                callback('property')
                return self.values[key]

        for role, user in tuple(h.rh.users.items()):
            h.rh.users[role] = CurrentUser(user)
        peer = h.reader.authority._provider_safety_release_peer
        original_peer = peer.revalidate_database_peer

        def peer_callback(*args, **kwargs):
            callback('provider')
            return original_peer(*args, **kwargs)

        peer.revalidate_database_peer = peer_callback
        original_transport = h.reader.authority._authority_transport

        @contextmanager
        def transport(*args, **kwargs):
            callback('transport')
            with original_transport(*args, **kwargs) as value:
                yield value

        h.reader.authority._authority_transport = transport

        def count(connection, _cursor, sql, *_):
            if phase['payload'] is not None and sql.lstrip().upper().startswith(
                ('INSERT', 'UPDATE', 'DELETE')
            ):
                phase['dml'].append(sql)
                phase['connection'] = connection

        event.listen(h.engine, 'before_cursor_execute', count)
        return h

    def append(self, connection, payload, **kwargs):
        if payload['transition_kind'] == 'case_claim':
            phase['payload'] = payload
            phase['mutations'] = kwargs['actual_mutations']
            phase['canonical'] = encoded(payload)
            phase['marker'] = self._authority.marker_path.read_bytes()
        return original_append(self, connection, payload, **kwargs)

    monkeypatch.setattr(
        __import__(__name__, fromlist=['runtime_source_harness']),
        'runtime_source_harness',
        harness,
    )
    monkeypatch.setattr(RagReleaseLedger, 'append', append)
    outcome = None
    try:
        test_genuine_source_case_projection_reuses_append_barrier(
            tmp_path, monkeypatch, False
        )
    except Exception as exc:
        outcome = exc
    h = kept[0]
    with h.engine.connect() as connection:
        auth = dict(
            connection.execute(select(h.tables.authorizations)).mappings().one()
        )
        cases = connection.execute(select(h.tables.cases)).all()
        runs = connection.execute(select(AgentRun.__table__)).all()
        costs = connection.execute(select(AgentRunCostComponent.__table__)).all()
        generations = connection.scalars(select(h.tables.ledgers.c.generation)).all()
        history = connection.execute(select(h.tables.transitions)).mappings().all()
    assert phase['calls'] == [], 'caller comparison committed after mutation DML'
    assert phase['attacked'] and phase['late_callbacks'] == []
    assert outcome is None
    assert auth['state'] == 'started' and auth['case_claim_count'] == 1
    assert len(cases) == len(runs) == 1 and len(costs) == 2
    assert generations == [2] and len(history) == 2
    assert history[-1]['payload_canonical_bytes'] == phase['canonical']
    assert h.target.marker_path.read_bytes() != phase['marker']


def test_append_owns_sql_schema_before_reader_teardown(tmp_path, monkeypatch):
    test_append_owns_inputs_before_reader_teardown(
        tmp_path, monkeypatch, 'schema_processor', 'reader_exit'
    )


def test_append_owns_sql_roster_schema_before_reader_teardown(tmp_path, monkeypatch):
    test_append_owns_inputs_before_reader_teardown(
        tmp_path, monkeypatch, 'schema_result_processor', 'reader_exit'
    )


def require_source(h, **overrides):
    args = {
        'manifest': h.authorization.manifest.executable,
        'source_binding': h.source,
        'authorization': dict(h.auth_row),
        'identity_secret': SECRET,
    }
    args.update(overrides)
    with h.engine.connect() as connection:
        return review.require_approved_case_source(connection, **args)


def test_genuine_approved_source_validates_locked_runtime_snapshots(tmp_path):
    h = runtime_source_harness(tmp_path)
    require_source(h)
    assert h.reader.runtime_lock_count == 1
    with pytest.raises(TypeError):
        copy(h.source)
    with pytest.raises(TypeError):
        deepcopy(h.source)


@pytest.mark.parametrize(
    'drift',
    [
        False,
        True,
        'preflight_exit',
        'final_reader_exit',
        'corpus_exit',
        'provider_exit',
        'source_exit',
        'after_preflight',
        'before_publication',
        'before_commit',
    ],
)
def test_genuine_source_case_projection_reuses_append_barrier(
    tmp_path, monkeypatch, drift, r2_attack=None
):
    from contextlib import contextmanager

    from backend.app.agent_runtime.rag_safety_identity import rag_identity_hmac
    from backend.app.models.agent_runs import AgentRun
    from backend.app.models.rag_runtime import AgentRunCostComponent
    from backend.app.rag.release_authority import RagReleaseAuthorityError
    from backend.app.rag.release_ledger import (
        RagReleaseLedgerError,
        RagReleaseMutationSet,
    )
    from backend.tests.release_ledger_fixtures import ReleaseHarness

    h = runtime_source_harness(tmp_path)
    with h.engine.begin() as connection:
        AgentRun.__table__.create(connection)
        AgentRunCostComponent.__table__.create(connection)
    selected = h.authorization.manifest.executable.cases[0]
    before = h.auth_row
    after = before | {
        'state': 'started',
        'case_claim_count': 1,
        'execution_process_instance_hmac': 'd' * 64,
        'execution_runner_fence_hmac': 'e' * 64,
        'reserved_cost_usd': sum(c.reserved_cost_usd for c in selected.components),
    }
    projection_payload = h.base | {
        'from_generation': h.reader.value.release.generation,
        'case_id_hmac': selected.case_id_hmac,
        'execution_process_instance_hmac': 'd' * 64,
        'execution_runner_fence_hmac': 'e' * 64,
    }
    with h.engine.connect() as connection:
        projection = review.prepare_case_claim_projection(
            connection,
            manifest=h.authorization.manifest.executable,
            source_binding=h.source,
            payload=projection_payload,
            identity_secret=SECRET,
        )
    parent, query, generation = projection.runtime_images
    case = {
        k: before[k] for k in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
    } | {
        'case_id_hmac': selected.case_id_hmac,
        'manifest_ordinal': 0,
        'state': 'claimed',
        'case_projection_hmac': None,
        'runtime_agent_run_id_hmac': rag_identity_hmac(
            {'agent_run_id': parent['id']},
            secret=SECRET,
            schema_version='rag-runtime-agent-run-id:v1',
            policy_version='rag-run:v2',
        ),
        'embedding_reserved_cost_usd': query['reserved_cost_usd'],
        'generation_reserved_cost_usd': generation['reserved_cost_usd'],
        'total_reserved_cost_usd': after['reserved_cost_usd'],
    }
    builder = object.__new__(ReleaseHarness)
    builder.engine, builder.secret, builder.ledger = h.engine, SECRET, h.ledger
    builder.base, builder.snapshot = h.base, h.reader.value.release
    original_barrier = h.reader.authority._authority_barrier
    held = []

    @contextmanager
    def barrier(connection, *, marker):
        assert not held, 'source must reuse the held release/provider barrier'
        with original_barrier(connection, marker=marker) as guard:
            held.append(guard)
            try:
                yield guard
            finally:
                held.pop()

    monkeypatch.setattr(h.reader.authority, '_authority_barrier', barrier)
    original_reader_lock = h.reader.locked_approved
    reader_exits = []
    callbacks_after_dml = []
    phase = {'dml': False, 'transition': False, 'connection': None}
    attack_calls = []

    def attack(effect):
        from sqlalchemy import update

        from backend.app.models.rag_runtime import RagProviderSafetyAuthority
        from backend.app.models.rag_serving import RagServingCorpusGeneration

        attack_calls.append(effect)
        if effect == 'corpus':
            phase['connection'].execute(
                update(RagServingCorpusGeneration).values(corpus_generation=99)
            )
        elif effect == 'provider':
            phase['connection'].execute(
                update(RagProviderSafetyAuthority).values(envelope_digest='f' * 64)
            )
        elif effect == 'key':
            h.options['review_key_registry'].clear()
        elif effect == 'marker':
            h.target.marker_path.write_bytes(b'test-only-invalid-marker')
        elif effect == 'replace_transaction':
            phase['connection'].rollback()
            phase['connection'].begin()
        else:
            getattr(phase['connection'], effect)()

    def callback(kind):
        if phase['dml']:
            callbacks_after_dml.append(kind)
        if r2_attack and ':' in r2_attack and not attack_calls:
            when, selected_kind, effect = r2_attack.split(':')
            armed = phase['transition'] if when == 'post_dml' else bool(reader_exits)
            if armed and kind == selected_kind:
                attack(effect)

    if r2_attack is not None:
        h.rh.callback = callback

        class CurrentUser:
            def __init__(self, user):
                self.values = vars(user).copy()

            def __getattr__(self, key):
                callback('property')
                return self.values[key]

        for role, user in tuple(h.rh.users.items()):
            h.rh.users[role] = CurrentUser(user)

        def mark_dml(connection, _cursor, sql, *_):
            phase['connection'] = connection
            if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                phase['dml'] = True
            if sql.lstrip().upper().startswith('INSERT INTO RAG_LIVE_GATE_TRANSITIONS'):
                phase['transition'] = True

        event.listen(h.engine, 'before_cursor_execute', mark_dml)
        for method in (
            'read',
            'read_hard_negative_oracle',
            'read_pgvector_baseline_members',
        ):
            original = getattr(h.reader, method)

            def counted(*args, _original=original, _method=method, **kwargs):
                callback(_method)
                return _original(*args, **kwargs)

            setattr(h.reader, method, counted)
        peer = h.reader.authority._provider_safety_release_peer
        original_peer_check = peer.revalidate_database_peer

        def counted_peer_check(*args, **kwargs):
            callback('provider_revalidation')
            return original_peer_check(*args, **kwargs)

        peer.revalidate_database_peer = counted_peer_check

    @contextmanager
    def reader_lock(connection, *, barrier_guard=None):
        callback('reader_enter')
        if barrier_guard is None:
            with original_reader_lock(connection) as reader:
                yield reader
        else:
            assert held == [barrier_guard]
            with h.reader.locked():
                h.reader.barrier_guard = barrier_guard
                try:
                    yield h.reader
                finally:
                    h.reader.barrier_guard = None
            reader_exits.append(len(reader_exits) + 1)
            callback('reader_exit')
            if r2_attack and r2_attack.startswith('reader_') and phase['dml']:
                attack(r2_attack.removeprefix('reader_'))
            if (drift == 'preflight_exit' and len(reader_exits) == 1) or (
                drift == 'final_reader_exit' and len(reader_exits) == 1
            ):
                h.options['review_key_registry'].clear()
            if drift == 'corpus_exit' and len(reader_exits) == 1:
                from sqlalchemy import update

                from backend.app.models.rag_serving import RagServingCorpusGeneration

                connection.execute(
                    update(RagServingCorpusGeneration).values(corpus_generation=99)
                )
            if drift == 'provider_exit' and len(reader_exits) == 1:
                from sqlalchemy import update

                from backend.app.models.rag_runtime import RagProviderSafetyAuthority

                connection.execute(
                    update(RagProviderSafetyAuthority).values(envelope_digest='f' * 64)
                )
            if drift == 'source_exit' and len(reader_exits) == 1:
                (h.repo / 'unapproved-reader-exit').write_text('test-only drift')

    h.reader.locked_approved = reader_lock
    if r2_attack == 'publication_hook':
        h.reader.authority._after_marker_replace = lambda: attack_calls.append('hook')
    if drift == 'after_preflight':
        original_preflight = RagReleaseMutationSet._preflight_runtime_mutations

        def preflight(*args, **options):
            result = original_preflight(*args, **options)
            h.options['review_key_registry'].clear()
            return result

        monkeypatch.setattr(
            RagReleaseMutationSet, '_preflight_runtime_mutations', preflight
        )
    if drift == 'before_publication':
        original_wrap = h.reader.authority._wrap

        def wrap(*args, **options):
            result = original_wrap(*args, **options)
            h.options['review_key_registry'].clear()
            return result

        monkeypatch.setattr(h.reader.authority, '_wrap', wrap)
    if drift == 'before_commit':

        def before_commit(_c, _cu, sql, *_):
            if sql.lstrip().upper().startswith('INSERT INTO RAG_LIVE_GATE_TRANSITIONS'):
                h.options['review_key_registry'].clear()

        event.listen(h.engine, 'after_cursor_execute', before_commit)
    if drift is True:
        h.options['review_key_registry'].clear()
    marker_before = h.target.marker_path.read_bytes()
    writes = []
    event.listen(
        h.engine, 'before_cursor_execute', lambda _c, _cu, sql, *_: writes.append(sql)
    )
    with h.engine.connect() as connection:
        payload, mutations = builder.prepare(
            connection,
            'case_claim',
            [
                ('authorization', before, after),
                ('case', None, case),
                ('agent_run', None, parent),
                ('cost_component', None, query),
                ('cost_component', None, generation),
            ],
            case=case,
        )
        refusal = None
        try:
            snapshot = h.ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=h.reader.database_identity,
                approved_case_claim=projection,
            )
        except RagReleaseLedgerError as exc:
            refusal = exc
        if r2_attack is not None:
            from backend.app.models.rag_serving import RagServingCorpusGeneration

            with h.engine.connect() as inspector:
                auth = (
                    inspector.execute(select(h.tables.authorizations)).mappings().one()
                )
                generation_now = inspector.scalar(select(h.tables.ledgers.c.generation))
                corpus_now = inspector.scalar(
                    select(RagServingCorpusGeneration.corpus_generation)
                )
                cases_now = inspector.execute(select(h.tables.cases)).mappings().all()
            assert corpus_now == 7
            if refusal is not None:
                assert dict(auth) == before
                assert generation_now == 1 and cases_now == []
                if r2_attack.endswith(':marker'):
                    assert (
                        h.target.marker_path.read_bytes() == b'test-only-invalid-marker'
                    )
                else:
                    assert h.target.marker_path.read_bytes() == marker_before
            else:
                assert (
                    generation_now == 2
                    and auth['case_claim_count'] == len(cases_now) == 1
                )
            if r2_attack.startswith('pre_dml:'):
                assert refusal is not None and len(attack_calls) == 1
                assert not any(
                    sql.lstrip().upper().startswith('INSERT') for sql in writes
                )
                return
            if r2_attack == 'publication_hook':
                assert refusal is not None and attack_calls == []
                assert reader_exits == [] and callbacks_after_dml == []
                assert not any(
                    sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE'))
                    for sql in writes
                )
                return
            assert attack_calls == []
            assert callbacks_after_dml == []
        if not drift:
            assert refusal is None
            assert snapshot.generation == builder.snapshot.generation + 1
    if drift:
        if drift not in ('corpus_exit', 'provider_exit', 'source_exit'):
            assert h.options['review_key_registry'] == {}
        assert (
            reader_exits
            == {
                True: [],
                'preflight_exit': [1],
                'final_reader_exit': [1],
                'corpus_exit': [1],
                'provider_exit': [1],
                'source_exit': [1],
                'after_preflight': [1],
                'before_publication': [1],
                'before_commit': [1],
            }[drift]
        )
        if drift == 'before_commit':
            # Preserve Task23 marker-first crash evidence; never silently repair it.
            assert h.target.marker_path.read_bytes() != marker_before
            with (
                h.engine.connect() as connection,
                pytest.raises(RagReleaseAuthorityError),
            ):
                h.reader.authority.inspect(
                    connection, database_identity=h.reader.database_identity
                )
        else:
            assert h.target.marker_path.read_bytes() == marker_before
        assert builder.records('authorization') == [before]
        assert builder.records('case') == []
        assert builder.records('agent_run') == []
        assert builder.records('cost_component') == []
        assert builder.records('dispatch') == []

        with h.engine.connect() as connection:
            assert connection.scalar(select(h.tables.ledgers.c.generation)) == 1
            assert len(connection.execute(select(h.tables.transitions)).all()) == 1
            from backend.app.models.rag_serving import RagServingCorpusGeneration

            assert (
                connection.scalar(select(RagServingCorpusGeneration.corpus_generation))
                == 7
            )
        if drift in (True, 'preflight_exit', 'after_preflight'):
            assert not any(
                sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE'))
                for sql in writes
            )
        assert refusal is not None
    else:
        assert len(builder.records('case')) == 1
        assert len(builder.records('cost_component')) == 2
        assert builder.records('dispatch') == []


@pytest.mark.parametrize(
    'attack',
    [
        *[
            f'post_dml:property:{effect}'
            for effect in (
                'corpus',
                'provider',
                'key',
                'marker',
                'commit',
                'rollback',
                'close',
                'begin',
                'replace_transaction',
            )
        ],
        'post_dml:auth_user_reader:corpus',
        'post_dml:clock:provider',
        'reader_commit',
        'reader_rollback',
        'reader_close',
        'reader_begin',
        'reader_replace_transaction',
        'count',
    ],
)
def test_actual_append_never_invokes_external_callbacks_after_first_dml(
    tmp_path, monkeypatch, attack
):
    test_genuine_source_case_projection_reuses_append_barrier(
        tmp_path,
        monkeypatch,
        False,
        r2_attack=attack,
    )


def test_actual_append_refuses_publication_callback_without_calling_it(
    tmp_path, monkeypatch
):
    test_genuine_source_case_projection_reuses_append_barrier(
        tmp_path, monkeypatch, False, r2_attack='publication_hook'
    )


@pytest.mark.parametrize(
    'effect',
    [
        'corpus',
        'provider',
        'key',
        'marker',
        'commit',
        'rollback',
        'close',
        'begin',
        'replace_transaction',
    ],
)
def test_last_reviewer_callback_drift_refuses_before_release_dml(
    tmp_path, monkeypatch, effect
):
    test_genuine_source_case_projection_reuses_append_barrier(
        tmp_path,
        monkeypatch,
        False,
        r2_attack=f'pre_dml:property:{effect}',
    )


@pytest.mark.parametrize(
    'attack',
    [
        'raw_source',
        'preview_source',
        'copy_auth',
        'manifest',
        'source_root',
        'ledger',
        'epoch',
        'host',
        'key',
        'key_registry',
        'provider',
        'corpus',
        'baseline',
        'limits',
        'approval',
        'state',
        'database',
        'generation',
        'dirty',
        'unlocked',
    ],
)
def test_approved_source_drift_and_forgery_refuse_without_mutation(tmp_path, attack):
    from backend.app.rag.release_ledger import RagReleaseLedgerError

    h = runtime_source_harness(tmp_path)
    overrides = {}
    if attack == 'raw_source':
        overrides['source_binding'] = object.__new__(review.ApprovedLiveManifestSource)
    if attack == 'preview_source':
        overrides['source_binding'] = h.preview.source_binding
    if attack == 'copy_auth':
        with pytest.raises(review.LiveGatePreviewError):
            h.authorizer.approved_source(copy(h.authorization))
        return
    if attack in ('manifest', 'source_root'):
        field, value = (
            ('source_manifest_hmac', 'e' * 64)
            if attack == 'source_root'
            else ('cases', ())
        )
        overrides['manifest'] = replace(
            h.authorization.manifest.executable, **{field: value}
        )
    if attack in ('ledger', 'epoch', 'host', 'key', 'generation'):
        key, value = {
            'ledger': ('ledger_uuid', UUID('22222222-2222-4222-8222-222222222222')),
            'epoch': ('ledger_epoch', 2),
            'host': ('designated_host_id_hmac', 'e' * 64),
            'key': ('fingerprint_key_version', 'v2'),
            'generation': ('generation', 0),
        }[attack]
        h.reader.value = replace(
            h.reader.value, release=replace(h.reader.value.release, **{key: value})
        )
    if attack == 'provider':
        h.reader.value = replace(
            h.reader.value, approved_provider_safety_snapshot_hmac='e' * 64
        )
    if attack == 'corpus':
        h.reader.value = replace(
            h.reader.value, corpus=replace(h.reader.value.corpus, corpus_generation=99)
        )
    if attack in ('baseline', 'approval', 'state', 'database'):
        key, value = {
            'baseline': ('baseline_hmac', 'e' * 64),
            'approval': ('approval_hmac', 'e' * 64),
            'state': ('state', 'finished_failed'),
            'database': ('validation_database_identity_hmac', 'e' * 64),
        }[attack]
        overrides['authorization'] = h.auth_row | {key: value}
    if attack == 'limits':
        object.__setattr__(h.authorization.limits, 'total_dispatches', 41)
    if attack == 'key_registry':
        h.options['review_key_registry'].clear()
    if attack == 'dirty':
        (h.repo / 'drift').write_text('dirty')
    if attack == 'unlocked':
        del h.reader.locked_approved
    writes = []

    def capture(_c, _cu, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            writes.append(sql)

    event.listen(h.engine, 'before_cursor_execute', capture)
    with pytest.raises((review.LiveGatePreviewError, RagReleaseLedgerError)):
        require_source(h, **overrides)
    assert writes == []
    with h.engine.connect() as connection:
        assert (
            dict(connection.execute(select(h.tables.authorizations)).mappings().one())
            == h.auth_row
        )


def test_approved_source_cannot_cross_engine_or_authorizer(tmp_path):
    from backend.app.rag.release_ledger import RagReleaseLedgerError

    h = runtime_source_harness(tmp_path)
    other = create_engine('sqlite://')
    with (
        other.connect() as connection,
        pytest.raises((review.LiveGatePreviewError, RagReleaseLedgerError)),
    ):
        review.require_approved_case_source(
            connection,
            manifest=h.authorization.manifest.executable,
            source_binding=h.source,
            authorization=h.auth_row,
            identity_secret=SECRET,
        )
    second = review.RagLiveGateAuthorizer(**h.options)
    with pytest.raises(review.LiveGatePreviewError):
        second.approved_source(h.authorization)
