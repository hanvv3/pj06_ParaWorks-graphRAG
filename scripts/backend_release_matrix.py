from __future__ import annotations

# ruff: noqa: E402 - script entrypoint establishes repository import root
import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import psycopg
from psycopg import sql

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.tests.postgres_isolation import lease_postgres_schema
from backend.tests.release_contracts import (
    ReleaseChild,
    ReleaseEvidenceSidecar,
    ReleaseInvocationIdentity,
    build_manifest,
    invocation_hash,
)
from backend.tests.release_evidence_plugin import (
    read_completed_sidecar,
    validate_completed_sidecar,
)

ADMIN_URL = 'postgresql://paraworks:paraworks@127.0.0.1:55432/postgres'
_RUN = re.compile(r'^[0-9a-f]{12}$', re.ASCII)


@dataclass(slots=True)
class OwnedPostgresResources:
    run_id: str
    role_name: str | None = None
    database_name: str | None = None
    sentinel_schema_name: str | None = None
    role_owned: bool = False
    database_owned: bool = False
    sentinel_owned: bool = False


class ReleaseRefused(RuntimeError):  # noqa: N818 - bounded release outcome
    pass


def _bounded_run(argv: Sequence[str], *, env: dict[str, str] | None = None, timeout: int = 30) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(tuple(argv), cwd=REPOSITORY_ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, shell=False, check=False)


def _preflight() -> None:
    inspected = _bounded_run(('docker', 'inspect', 'paraworks-postgres'))
    if inspected.returncode != 0:
        raise ReleaseRefused('preflight_refused')
    try:
        value = json.loads(inspected.stdout)[0]
        bindings = value['NetworkSettings']['Ports']['5432/tcp']
        valid = (
            value['Config']['Image'] == 'pgvector/pgvector:pg17'
            and value['State']['Running'] is True
            and value['State']['Health']['Status'] == 'healthy'
            and value['Config']['Labels']['com.docker.compose.service'] == 'postgres'
            and len(bindings) == 1
            and bindings[0]['HostIp'] == '127.0.0.1'
            and bindings[0]['HostPort'] == '55432'
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        valid = False
    if not valid:
        raise ReleaseRefused('preflight_refused')
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=5) as connection:
            identity = connection.execute('SELECT current_user, current_database()').fetchone()
    except Exception:
        raise ReleaseRefused('preflight_refused') from None
    if identity != ('paraworks', 'postgres'):
        raise ReleaseRefused('preflight_refused')


def _resource_names(run_id: str) -> OwnedPostgresResources:
    if not _RUN.fullmatch(run_id):
        raise ReleaseRefused('preflight_refused')
    prefix = f'paraworks_c5t16_{run_id}'
    resources = OwnedPostgresResources(
        run_id,
        role_name=f'{prefix}_role_test',
        database_name=f'{prefix}_database_test',
        sentinel_schema_name=f'{prefix}_sentinel',
    )
    if any(len(value or '') > 63 for value in (resources.role_name, resources.database_name, resources.sentinel_schema_name)):
        raise ReleaseRefused('preflight_refused')
    return resources


def _create_resources(resources: OwnedPostgresResources, password: str) -> str:
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=5) as connection:
            existing = connection.execute(
                'SELECT (SELECT count(*) FROM pg_roles WHERE rolname=%s), (SELECT count(*) FROM pg_database WHERE datname=%s)',
                (resources.role_name, resources.database_name),
            ).fetchone()
            if existing != (0, 0):
                raise ReleaseRefused('preflight_refused')
            connection.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(sql.Identifier(resources.role_name), sql.Literal(password)))
            resources.role_owned = True
            connection.execute(sql.SQL('CREATE DATABASE {} OWNER {}').format(sql.Identifier(resources.database_name), sql.Identifier(resources.role_name)))
            resources.database_owned = True
        admin_database_url = f'postgresql://paraworks:paraworks@127.0.0.1:55432/{resources.database_name}'
        with psycopg.connect(admin_database_url, autocommit=True, connect_timeout=5) as connection:
            connection.execute('CREATE EXTENSION IF NOT EXISTS vector')
            connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(resources.sentinel_schema_name)))
            resources.sentinel_owned = True
        return f'postgresql+psycopg://{resources.role_name}:{quote(password)}@127.0.0.1:55432/{resources.database_name}'
    except ReleaseRefused:
        raise
    except Exception:
        raise ReleaseRefused('preflight_refused') from None


def _cleanup(resources: OwnedPostgresResources) -> bool:
    failed = False
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=5) as connection:
            if resources.database_owned:
                connection.execute('SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()', (resources.database_name,))
                connection.execute(sql.SQL('DROP DATABASE IF EXISTS {}').format(sql.Identifier(resources.database_name)))
            if resources.role_owned:
                connection.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(resources.role_name)))
            remaining = connection.execute(
                'SELECT (SELECT count(*) FROM pg_database WHERE datname=%s), (SELECT count(*) FROM pg_roles WHERE rolname=%s)',
                (resources.database_name, resources.role_name),
            ).fetchone()
            if remaining != (0, 0):
                failed = True
    except Exception:
        failed = True
    return not failed


def _child_environment(*, base_url: str, resources: OwnedPostgresResources, profile: str, child: ReleaseChild, identity_hash: str, sidecar: Path, sqlite_path: Path) -> dict[str, str]:
    allowed_parent = ('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL')
    env = {name: os.environ[name] for name in allowed_parent if name in os.environ}
    env.update({
        'DATABASE_URL': f'sqlite:///{sqlite_path.as_posix()}',
        'PARAWORKS_TEST_POSTGRES_URL': base_url,
        'PARAWORKS_PGVECTOR_TEST_DATABASE_URL': base_url,
        'PARAWORKS_DEMO_MODE': 'true',
        'PARAWORKS_ENV': 'local',
        'AUTO_REVIEW_MODE': 'disabled',
        'AGENT_LLM_ENABLED': 'false',
        'PYTEST_ADDOPTS': '',
        'PYTEST_PLUGINS': '',
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        'PARAWORKS_RELEASE_RUN_ID': resources.run_id,
        'PARAWORKS_RELEASE_PROFILE': profile,
        'PARAWORKS_RELEASE_CHILD_ID': child.child_id,
        'PARAWORKS_RELEASE_INVOCATION_HASH': identity_hash,
        'PARAWORKS_RELEASE_SIDECAR_PATH': str(sidecar),
        'PARAWORKS_RELEASE_EXPECTED_DATABASE': resources.database_name or '',
        'PARAWORKS_RELEASE_EXPECTED_ROLE': resources.role_name or '',
    })
    return env


def _run_child(*, profile: str, child: ReleaseChild, commit_sha: str, base_url: str, resources: OwnedPostgresResources, artifact_dir: Path) -> tuple[ReleaseEvidenceSidecar, int]:
    identity_hash = invocation_hash(profile=profile, child=child, commit_sha=commit_sha)
    identity = ReleaseInvocationIdentity(resources.run_id, profile, child.child_id, identity_hash)
    sidecar_path = artifact_dir / f'{child.child_id}.json'
    sqlite_path = artifact_dir / f'{child.child_id}.sqlite3'
    env = _child_environment(base_url=base_url, resources=resources, profile=profile, child=child, identity_hash=identity_hash, sidecar=sidecar_path, sqlite_path=sqlite_path)
    command = (sys.executable, '-m', 'pytest', *child.pytest_argv, '-p', 'backend.tests.release_evidence_plugin')
    result = _bounded_run(command, env=env, timeout=child.timeout_seconds)
    if result.returncode not in {*child.allowed_pytest_exit_codes, 1}:
        raise ReleaseRefused('verification_failed')
    sidecar = read_completed_sidecar(sidecar_path)
    validate_completed_sidecar(sidecar, expected_identity=identity, native_exit_code=result.returncode)
    return sidecar, result.returncode


def _event_counts(sidecars: Sequence[ReleaseEvidenceSidecar]) -> tuple[dict[str, int], tuple[str, ...]]:
    final: dict[str, Any] = {}
    for sidecar in sidecars:
        for event in sidecar.events:
            prior = final.get(event.nodeid)
            if prior is None or event.outcome == 'failed' or (event.outcome == 'skipped' and prior.outcome == 'passed'):
                final[event.nodeid] = event
    counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'xfailed': 0, 'errors': 0}
    failures = []
    for nodeid, event in final.items():
        if event.wasxfail:
            counts['xfailed'] += 1
        elif event.outcome == 'failed' and event.phase != 'call':
            counts['errors'] += 1
        elif event.outcome in counts:
            counts[event.outcome] += 1
        if event.outcome == 'failed':
            failures.append(nodeid)
    return counts, tuple(sorted(failures))


def _lease_counts(
    sidecars: Sequence[ReleaseEvidenceSidecar],
) -> tuple[int, int, bool]:
    lifecycle: dict[str, list[str]] = {}
    for sidecar in sidecars:
        for event in sidecar.lease_events:
            lifecycle.setdefault(event.lease_id_sha256, []).append(event.state)
    created = sum(states.count('created') for states in lifecycle.values())
    dropped = sum(states.count('dropped') for states in lifecycle.values())
    balanced = all(states == ['created', 'dropped'] for states in lifecycle.values())
    return created, dropped, balanced


def _coverage_counts(
    sidecars: Sequence[ReleaseEvidenceSidecar],
    *,
    expected_deselected_nodeids: Sequence[str],
) -> tuple[int, int, int, bool]:
    canonical = next(
        (
            sidecar
            for sidecar in sidecars
            if sidecar.identity.child_id.endswith('collection')
        ),
        None,
    )
    if canonical is None:
        return 0, 0, 0, False
    collected = set(canonical.collection_nodeids)
    executed = {
        event.nodeid
        for sidecar in sidecars
        if sidecar is not canonical
        for event in sidecar.events
    }
    missing = collected - executed
    valid = (
        not (executed - collected)
        and missing == set(expected_deselected_nodeids)
        and len(collected) == len(canonical.collection_nodeids)
    )
    return len(collected), len(executed), len(missing), valid


def _verify_profile(profile: str, sidecars: Sequence[ReleaseEvidenceSidecar], *, focused: bool) -> tuple[dict[str, int], str, tuple[str, ...]]:
    counts, failures = _event_counts(sidecars)
    if focused:
        return counts, 'passed' if not failures and counts['skipped'] == 0 and counts['xfailed'] == 0 else 'verification_failed', failures
    canonical = next(sidecar for sidecar in sidecars if sidecar.identity.child_id.endswith('collection'))
    verification = [sidecar for sidecar in sidecars if sidecar is not canonical]
    selected = tuple(dict.fromkeys(node for sidecar in verification for node in sidecar.collection_nodeids))
    if len(selected) != sum(len(sidecar.collection_nodeids) for sidecar in verification) or set(selected) != set(canonical.collection_nodeids):
        return counts, 'collection_mismatch', failures
    if counts['skipped'] or counts['xfailed']:
        return counts, 'verification_failed', failures
    if profile == 'full':
        # Historical Slack selectors are not an allowed-failure list. The active
        # deferred set can be empty; then every full-suite failure is unexpected.
        expected = tuple(sorted(build_manifest().deferred_slack_failure_nodeids))
        return counts, 'passed' if failures == expected else 'baseline_mismatch', tuple(node for node in failures if node not in expected)
    return counts, 'passed' if not failures else 'verification_failed', failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, choices=tuple(build_manifest().profiles))
    parser.add_argument('--child-id')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest()
    profile = manifest.profiles[args.profile]
    children = profile.children
    focused = args.child_id is not None
    if focused:
        children = tuple(child for child in children if child.child_id == args.child_id)
        if len(children) != 1:
            print(json.dumps({'schema_version': 1, 'profile': args.profile, 'outcome': 'collection_mismatch', 'release_proof': False}, sort_keys=True))
            return 2
    run_id = secrets.token_hex(6)
    resources = _resource_names(run_id)
    outcome = 'preflight_refused'
    counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'xfailed': 0, 'errors': 0}
    cleanup_ok = True
    sidecars: list[ReleaseEvidenceSidecar] = []
    unexpected_nodeids: tuple[str, ...] = ()
    lease_created_count = 0
    lease_dropped_count = 0
    collected_count = 0
    selected_count = 0
    deselected_count = 0
    try:
        _preflight()
        base_url = _create_resources(resources, secrets.token_urlsafe(24))
        commit = _bounded_run(('git', 'rev-parse', 'HEAD')).stdout.decode('ascii').strip()
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ReleaseRefused('preflight_refused')
        with tempfile.TemporaryDirectory(prefix=f'paraworks-c5t16-{run_id}-') as directory:
            artifact_dir = Path(directory)
            for child in children:
                if child.kind == 'canonical_collection':
                    sidecar, _ = _run_child(profile=args.profile, child=child, commit_sha=commit, base_url=base_url, resources=resources, artifact_dir=artifact_dir)
                else:
                    with lease_postgres_schema(base_url, run_id=run_id, scope_name=child.child_id) as lease:
                        sidecar, _ = _run_child(profile=args.profile, child=child, commit_sha=commit, base_url=lease.database_url, resources=resources, artifact_dir=artifact_dir)
                sidecars.append(sidecar)
        counts, outcome, unexpected_nodeids = _verify_profile(args.profile, sidecars, focused=focused)
        lease_created_count, lease_dropped_count, leases_balanced = _lease_counts(sidecars)
        if not leases_balanced:
            outcome = 'evidence_refused'
        if not focused:
            (
                collected_count,
                selected_count,
                deselected_count,
                coverage_valid,
            ) = _coverage_counts(
                sidecars,
                expected_deselected_nodeids=profile.expected_deselected_nodeids,
            )
            if not coverage_valid:
                outcome = 'collection_mismatch'
    except (ReleaseRefused, ValueError, subprocess.TimeoutExpired) as exc:
        outcome = str(exc) if str(exc) in {'preflight_refused', 'verification_failed', 'collection_mismatch', 'evidence_refused'} else 'evidence_refused'
    except Exception:
        outcome = 'evidence_refused'
    finally:
        cleanup_ok = _cleanup(resources)
        if not cleanup_ok:
            outcome = 'cleanup_failed'
    if focused:
        selected_count = sum(counts.values())
    report = {
        'schema_version': 1,
        'profile': args.profile,
        'outcome': outcome,
        'release_proof': outcome == 'passed' and not focused,
        'collected_count': collected_count,
        'selected_count': selected_count,
        'deselected_count': deselected_count,
        'passed_count': counts['passed'],
        'failed_count': counts['failed'],
        'error_count': counts['errors'],
        'skipped_count': counts['skipped'],
        'xfailed_count': counts['xfailed'],
        'expected_sidecar_count': len(children),
        'observed_sidecar_count': len(sidecars),
        'unexpected_nodeids': unexpected_nodeids,
        'lease_created_count': lease_created_count,
        'lease_dropped_count': lease_dropped_count,
        'database_owned': resources.database_owned,
        'role_owned': resources.role_owned,
        'remaining_database_count': 0 if cleanup_ok else 1,
        'remaining_role_count': 0 if cleanup_ok else 1,
        'no_live_provider': True,
    }
    print(json.dumps(report, sort_keys=True, separators=(',', ':')))
    return 0 if outcome == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
