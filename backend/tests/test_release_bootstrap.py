import pytest

from backend.tests.release_bootstrap import (
    ReleaseBootstrapRefused,
    probe_effective_state,
)


def test_guarded_probe_reports_only_bounded_disabled_state(monkeypatch, tmp_path):
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{tmp_path / "release.db"}')
    state = probe_effective_state()
    assert state.resolved_database_backend == 'sqlite'
    assert state.auto_review_mode == 'disabled'
    assert state.live_provider_credentials_present is False


def test_guard_refuses_xdist(monkeypatch, tmp_path):
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{tmp_path / "release.db"}')
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    with pytest.raises(ReleaseBootstrapRefused):
        probe_effective_state()
