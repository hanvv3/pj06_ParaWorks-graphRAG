from pathlib import Path
from types import SimpleNamespace

from backend.tests.release_contracts import LeaseLifecycleEvent, build_manifest
from scripts.checks.backend_release_matrix import _coverage_counts, _lease_counts


def test_repaired_historical_slack_failure_is_now_unexpected():
    from backend.tests.release_contracts import SLACK_TEN, ReleaseEvent
    from scripts.checks.backend_release_matrix import _verify_profile

    node = SLACK_TEN[0]
    canonical = SimpleNamespace(
        identity=SimpleNamespace(child_id='full-collection'),
        collection_nodeids=(node,),
        events=(),
    )
    verification = SimpleNamespace(
        identity=SimpleNamespace(child_id='full-backend'),
        collection_nodeids=(node,),
        events=(ReleaseEvent(node, 'call', 'failed', False, 0),),
    )
    _, status, unexpected = _verify_profile(
        'full', (canonical, verification), focused=False
    )
    assert status == 'baseline_mismatch' and unexpected == (node,)
    verification.events = (ReleaseEvent(node, 'call', 'passed', False, 0),)
    _, status, unexpected = _verify_profile(
        'full', (canonical, verification), focused=False
    )
    assert status == 'passed' and unexpected == ()


def test_controller_uses_exact_profile_children_and_argument_vectors():
    manifest = build_manifest()
    # Canonical collection plus nine existing PostgreSQL gates and the
    # validation-only Task 23 release-authority gate.
    assert len(manifest.profiles['postgres'].children) == 11
    assert all(
        isinstance(child.pytest_argv, tuple)
        for profile in manifest.profiles.values()
        for child in profile.children
    )
    assert all(
        ';' not in arg and '&&' not in arg
        for profile in manifest.profiles.values()
        for child in profile.children
        for arg in child.pytest_argv
    )


def test_controller_has_no_application_settings_or_session_import():
    source = (
        Path(__file__).parents[2] / 'scripts' / 'checks' / 'backend_release_matrix.py'
    ).read_text(encoding='utf-8')
    assert 'backend.app.core.config' not in source
    assert 'backend.app.db.session' not in source


def test_controller_accepts_only_exact_created_then_dropped_lease_pairs():
    balanced = SimpleNamespace(
        lease_events=(
            LeaseLifecycleEvent('a' * 64, 'created'),
            LeaseLifecycleEvent('a' * 64, 'dropped'),
        )
    )
    unbalanced = SimpleNamespace(
        lease_events=(LeaseLifecycleEvent('b' * 64, 'created'),)
    )

    assert _lease_counts((balanced,)) == (1, 1, True)
    assert _lease_counts((unbalanced,)) == (1, 0, False)


def test_controller_counts_only_exact_manifest_deselections_as_unexecuted():
    identity = SimpleNamespace(child_id='sample-collection')
    canonical = SimpleNamespace(
        identity=identity,
        collection_nodeids=('a', 'b', 'deferred'),
        events=(),
    )
    verification = SimpleNamespace(
        identity=SimpleNamespace(child_id='sample-verification'),
        collection_nodeids=('a', 'b', 'deferred'),
        events=(SimpleNamespace(nodeid='a'), SimpleNamespace(nodeid='b')),
    )

    assert _coverage_counts(
        (canonical, verification),
        expected_deselected_nodeids=('deferred',),
    ) == (3, 2, 1, True)
    assert _coverage_counts(
        (canonical, verification),
        expected_deselected_nodeids=(),
    ) == (3, 2, 1, False)
