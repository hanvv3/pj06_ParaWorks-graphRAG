from backend.tests.release_contracts import (
    DEFERRED_SLACK_FAILURES,
    POSTGRES_MODULES,
    SLACK_TEN,
    build_manifest,
    invocation_hash,
)


def test_manifest_freezes_exact_profiles_and_deferred_baseline():
    manifest = build_manifest()
    assert tuple(manifest.profiles) == (
        'settings-diagnostic',
        'postgres',
        'compatibility',
        'non-slack',
        'full',
    )
    assert len(SLACK_TEN) == 10
    # Task 23 adds the isolated release-authority PostgreSQL gate. Keeping the
    # count exact prevents the controller from silently dropping this module.
    assert len(POSTGRES_MODULES) == 10
    assert DEFERRED_SLACK_FAILURES == ()
    assert manifest.deferred_slack_failure_nodeids == ()
    assert manifest.profiles['compatibility'].expected_deselected_nodeids == ()
    assert manifest.profiles['non-slack'].expected_deselected_nodeids == ()
    assert all(
        not arg.startswith('--deselect=')
        for profile in manifest.profiles.values()
        for child in profile.children
        for arg in child.pytest_argv
    )
    assert manifest.profiles['full'].expected_deselected_nodeids == ()
    assert len(
        {
            child.child_id
            for profile in manifest.profiles.values()
            for child in profile.children
        }
    ) == sum(len(profile.children) for profile in manifest.profiles.values())


def test_invocation_hash_changes_with_commit_and_selector():
    child = build_manifest().profiles['full'].children[0]
    assert invocation_hash(
        profile='full', child=child, commit_sha='a' * 40
    ) != invocation_hash(profile='full', child=child, commit_sha='b' * 40)


def test_default_slack_runner_keeps_exact_history_and_refuses_empty_selectors(
    monkeypatch,
):
    import pytest

    from backend.tests import release_contracts
    from backend.tests.slack_offline_runner import historical_pytest_args

    assert historical_pytest_args([]) == [*SLACK_TEN, '-q', '--tb=short']
    monkeypatch.setattr(release_contracts, 'SLACK_TEN', ())
    with pytest.raises(ValueError, match='must not be empty'):
        historical_pytest_args([])
