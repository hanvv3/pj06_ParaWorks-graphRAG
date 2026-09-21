from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_smoke_frontend_uses_isolated_next_cache() -> None:
    smoke_script = (REPO_ROOT / 'scripts' / 'internal' / 'local_runtime.py').read_text()
    next_config = (REPO_ROOT / 'frontend' / 'next.config.ts').read_text()

    assert '.next-smoke' in smoke_script
    assert 'NEXT_DIST_DIR' in smoke_script
    assert 'distDir' in next_config
    assert 'NEXT_DIST_DIR' in next_config
