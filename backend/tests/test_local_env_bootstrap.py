import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values, set_key

from backend.app.core.config import Settings
from scripts import bootstrap_local_env as bootstrap_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SCRIPT = REPOSITORY_ROOT / 'scripts' / 'bootstrap_local_env.py'
SECRET_NAMES = (
    'AGENT_RUNTIME_FINGERPRINT_SECRET',
    'AUTH_SESSION_SECRET',
    'GOOGLE_OAUTH_STATE_SECRET',
    'GOOGLE_IDENTITY_STATE_SECRET',
    'SLACK_OAUTH_STATE_SECRET',
)


def _prepare_workspace(workspace: Path) -> None:
    subprocess.run(['git', 'init', '-q'], cwd=workspace, check=True)
    (workspace / '.gitignore').write_text(
        '.env\n.env.*.tmp\n', encoding='utf-8'
    )
    shutil.copyfile(REPOSITORY_ROOT / '.env.example', workspace / '.env.example')


def _run_bootstrap(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BOOTSTRAP_SCRIPT),
            '--workspace',
            str(workspace),
        ],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )


def test_bootstrap_creates_ignored_env_with_durable_local_secrets(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)

    result = _run_bootstrap(tmp_path)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == {
        'created': True,
        'generated_names': list(SECRET_NAMES),
        'path': '.env',
    }
    values = dotenv_values(tmp_path / '.env')
    generated_values = [values[name] for name in SECRET_NAMES]
    assert all(value is not None and len(value) >= 64 for value in generated_values)
    assert len(set(generated_values)) == len(SECRET_NAMES)
    assert all(value not in result.stdout for value in generated_values)
    Settings(_env_file=tmp_path / '.env').require_c5_durable_key_ready()


def test_bootstrap_preserves_existing_provider_key_and_generated_secrets(
    tmp_path: Path,
) -> None:
    _prepare_workspace(tmp_path)
    first = _run_bootstrap(tmp_path)
    assert first.returncode == 0, first.stderr
    env_path = tmp_path / '.env'
    provider_key = 'test-provider-key-that-is-not-live'
    set_key(env_path, 'OPENAI_API_KEY', provider_key, quote_mode='never')
    before = dotenv_values(env_path)

    second = _run_bootstrap(tmp_path)

    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout) == {
        'created': False,
        'generated_names': [],
        'path': '.env',
    }
    after = dotenv_values(env_path)
    assert after['OPENAI_API_KEY'] == provider_key
    assert all(after[name] == before[name] for name in SECRET_NAMES)
    assert provider_key not in second.stdout


def test_bootstrap_preserves_existing_32_byte_signing_secrets(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    env_path = tmp_path / '.env'
    shutil.copyfile(tmp_path / '.env.example', env_path)
    existing_values = {
        name: chr(ord('a') + index) * 32
        for index, name in enumerate(SECRET_NAMES)
    }
    for name, value in existing_values.items():
        set_key(env_path, name, value, quote_mode='never')

    result = _run_bootstrap(tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        'created': False,
        'generated_names': [],
        'path': '.env',
    }
    after = dotenv_values(env_path)
    assert all(after[name] == value for name, value in existing_values.items())
    Settings(_env_file=env_path).require_c5_durable_key_ready()


def test_bootstrap_creates_secret_staging_file_exclusively_with_mode_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_workspace(tmp_path)
    real_open = os.open
    staging_calls: list[tuple[int, int]] = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        candidate = Path(path)
        if candidate.parent == tmp_path and candidate.name.startswith('.env.'):
            staging_calls.append((flags, mode))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(bootstrap_module.os, 'open', recording_open)

    bootstrap_module.bootstrap_local_env(tmp_path)

    assert staging_calls == [
        (os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600),
    ]
