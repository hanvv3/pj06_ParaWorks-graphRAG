from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values, set_key

SECRET_NAMES = (
    'AGENT_RUNTIME_FINGERPRINT_SECRET',
    'AUTH_SESSION_SECRET',
    'GOOGLE_OAUTH_STATE_SECRET',
    'GOOGLE_IDENTITY_STATE_SECRET',
    'SLACK_OAUTH_STATE_SECRET',
)
_GENERATED_SECRET_PATTERN = re.compile(r'^[0-9a-f]{64}$')
_MINIMUM_SECRET_BYTES = 32


class LocalEnvBootstrapError(RuntimeError):
    pass


def _run_git(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', '-C', str(workspace), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _require_ignored_untracked_target(workspace: Path) -> None:
    ignored = _run_git(workspace, 'check-ignore', '-q', '--', '.env')
    if ignored.returncode != 0:
        raise LocalEnvBootstrapError('env_target_not_ignored')
    tracked = _run_git(workspace, 'ls-files', '--', '.env')
    if tracked.returncode != 0 or tracked.stdout.strip():
        raise LocalEnvBootstrapError('env_target_tracked')


def _require_ignored_temporary_path(workspace: Path, name: str) -> None:
    ignored = _run_git(workspace, 'check-ignore', '-q', '--', name)
    if ignored.returncode != 0:
        raise LocalEnvBootstrapError('env_temporary_target_not_ignored')


def _needs_generated_secret(value: str | None) -> bool:
    if not value or not value.strip():
        return True
    normalized = value.strip().lower()
    if normalized.startswith(('local-development-', 'replace-with-', 'your-')):
        return True
    return len(value.encode('utf-8')) < _MINIMUM_SECRET_BYTES


def _valid_secret(value: str | None) -> bool:
    return bool(value) and len(value.encode('utf-8')) >= _MINIMUM_SECRET_BYTES


def bootstrap_local_env(workspace: Path) -> dict[str, object]:
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise LocalEnvBootstrapError('workspace_not_directory')
    env_path = workspace / '.env'
    template_path = workspace / '.env.example'
    if env_path.parent != workspace:
        raise LocalEnvBootstrapError('env_target_outside_workspace')
    if env_path.is_symlink() or template_path.is_symlink():
        raise LocalEnvBootstrapError('env_symlink_refused')
    if not template_path.is_file():
        raise LocalEnvBootstrapError('env_template_missing')
    _require_ignored_untracked_target(workspace)

    created = not env_path.exists()
    if not created and not env_path.is_file():
        raise LocalEnvBootstrapError('env_target_not_regular_file')

    source_path = template_path if created else env_path
    values = dotenv_values(source_path)
    generated_values: dict[str, str] = {}
    reserved_values = {
        value for name in SECRET_NAMES if (value := values.get(name)) is not None
    }
    for name in SECRET_NAMES:
        value = values.get(name)
        if not _needs_generated_secret(value):
            continue
        generated = secrets.token_hex(32)
        while generated in reserved_values:
            generated = secrets.token_hex(32)
        generated_values[name] = generated
        reserved_values.add(generated)

    proposed_values = {
        name: generated_values.get(name, values.get(name)) for name in SECRET_NAMES
    }
    if any(not _valid_secret(value) for value in proposed_values.values()):
        raise LocalEnvBootstrapError('generated_secret_verification_failed')
    if any(
        not _GENERATED_SECRET_PATTERN.fullmatch(value)
        for value in generated_values.values()
    ) or len(set(generated_values.values())) != len(generated_values):
        raise LocalEnvBootstrapError('generated_secret_verification_failed')

    if created or generated_values:
        temporary_name = f'.env.{secrets.token_hex(8)}.tmp'
        _require_ignored_temporary_path(workspace, temporary_name)
        temporary_path = workspace / temporary_name
        try:
            descriptor = os.open(
                temporary_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)
            shutil.copyfile(source_path, temporary_path)
            for name, value in generated_values.items():
                set_key(
                    temporary_path,
                    name,
                    value,
                    quote_mode='never',
                    encoding='utf-8',
                )
            verified = dotenv_values(temporary_path)
            if any(
                verified.get(name) != value
                for name, value in proposed_values.items()
            ):
                raise LocalEnvBootstrapError('generated_secret_verification_failed')
            os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
            os.replace(temporary_path, env_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    os.chmod(env_path, stat.S_IRUSR | stat.S_IWUSR)
    return {
        'created': created,
        'generated_names': list(generated_values),
        'path': '.env',
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Create ParaWorks .env and generate local signing secrets.'
    )
    parser.add_argument(
        '--workspace',
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = bootstrap_local_env(arguments.workspace)
    except (LocalEnvBootstrapError, OSError) as exc:
        error_code = str(exc) if isinstance(exc, LocalEnvBootstrapError) else 'env_io_error'
        print(json.dumps({'error': error_code}, separators=(',', ':')), file=sys.stderr)
        return 1
    print(json.dumps(report, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
