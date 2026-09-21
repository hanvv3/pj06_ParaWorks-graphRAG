"""Relocated command entrypoints remain executable outside the repository cwd."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('command', [
    'internal/local_runtime.py',
    'checks/check_db_schema.py',
    'checks/check_pgvector_dev.py',
    'checks/backend_release_matrix.py',
    'admin/bootstrap_local_env.py',
    'admin/bootstrap_langgraph_checkpointer.py',
    'admin/prune_langgraph_checkpoints.py',
    'admin/reset_connector_data.py',
    'admin/sync_slack.py',
])
def test_relocated_command_help_from_arbitrary_cwd(command, tmp_path):
    env = dict(os.environ, DATABASE_URL='sqlite://', PARAWORKS_DATABASE_URL='sqlite://')
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / command), '--help'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()
    assert list(tmp_path.iterdir()) == []
