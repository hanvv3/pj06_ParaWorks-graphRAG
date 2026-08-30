from backend.tests.release_contracts import (
    POSTGRES_MODULES,
    SLACK_TEN,
    build_manifest,
    invocation_hash,
)


def test_manifest_freezes_exact_profiles_and_deferred_baseline():
    manifest = build_manifest()
    assert tuple(manifest.profiles) == ('settings-diagnostic', 'postgres', 'compatibility', 'non-slack', 'full')
    assert len(SLACK_TEN) == 10
    assert len(POSTGRES_MODULES) == 9
    assert len(manifest.profiles['compatibility'].expected_deselected_nodeids) == 4
    assert manifest.profiles['non-slack'].expected_deselected_nodeids == SLACK_TEN
    assert manifest.profiles['full'].expected_deselected_nodeids == ()
    assert len({child.child_id for profile in manifest.profiles.values() for child in profile.children}) == sum(len(profile.children) for profile in manifest.profiles.values())


def test_invocation_hash_changes_with_commit_and_selector():
    child = build_manifest().profiles['full'].children[0]
    assert invocation_hash(profile='full', child=child, commit_sha='a' * 40) != invocation_hash(profile='full', child=child, commit_sha='b' * 40)
