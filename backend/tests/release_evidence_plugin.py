from __future__ import annotations

import json
import os
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest

from backend.tests.release_bootstrap import apply_test_environment_guard
from backend.tests.release_contracts import (
    ReleaseEvent,
    ReleaseEvidenceSidecar,
    ReleaseInvocationIdentity,
    canonical_sha256,
    sidecar_from_json,
)

SIDECAR_SCHEMA_VERSION = 1
_nodeids: tuple[str, ...] = ()
_events: list[ReleaseEvent] = []
_lease_events: list[dict[str, str]] = []
_module_lease: object | None = None
_module_key: str | None = None
_module_base_url: str | None = None


def pytest_load_initial_conftests() -> None:
    apply_test_environment_guard()


def validate_safe_nodeid(nodeid: str) -> None:
    if not nodeid.startswith('backend/tests/') or any(char in nodeid for char in ('\n', '\r', '\x00')):
        raise ValueError('unsafe_nodeid')


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    global _nodeids
    values = tuple(item.nodeid.replace('\\', '/') for item in items)
    for nodeid in values:
        validate_safe_nodeid(nodeid)
    _nodeids = values


def _close_module_lease() -> None:
    global _module_lease, _module_key, _module_base_url
    if _module_lease is None:
        return
    base_url = _module_base_url
    try:
        _module_lease.__exit__(None, None, None)
    finally:
        if base_url is not None:
            os.environ['PARAWORKS_TEST_POSTGRES_URL'] = base_url
            os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL'] = base_url
        _module_lease = None
        _module_key = None
        _module_base_url = None


def _open_module_lease(item: pytest.Item) -> None:
    global _module_lease, _module_key, _module_base_url
    key = item.nodeid.replace('\\', '/').split('::', 1)[0]
    if key == _module_key:
        return
    _close_module_lease()
    base_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    run_id = os.getenv('PARAWORKS_RELEASE_RUN_ID')
    if not base_url or not run_id:
        return
    from backend.tests.postgres_isolation import lease_postgres_schema

    context = lease_postgres_schema(base_url, run_id=run_id, scope_name=Path(key).stem)
    lease = context.__enter__()
    _module_lease = context
    _module_key = key
    _module_base_url = base_url
    os.environ['PARAWORKS_TEST_POSTGRES_URL'] = lease.database_url
    os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL'] = lease.database_url


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item):
    _open_module_lease(item)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when == 'call' or (report.when in {'setup', 'teardown'} and report.outcome != 'passed'):
        nodeid = item.nodeid.replace('\\', '/')
        validate_safe_nodeid(nodeid)
        _events.append(ReleaseEvent(nodeid, report.when, report.outcome, bool(getattr(report, 'wasxfail', False)), _nodeids.index(nodeid), None))


def record_lease_event(*, lease_identity: str, state: str) -> None:
    if not os.getenv('PARAWORKS_RELEASE_SIDECAR_PATH'):
        return
    digest = sha256(lease_identity.encode()).hexdigest()
    _lease_events.append({'lease_id_sha256': digest, 'state': state})


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _close_module_lease()
    target_value = os.getenv('PARAWORKS_RELEASE_SIDECAR_PATH')
    if not target_value:
        return
    target = Path(target_value)
    if target.exists():
        session.exitstatus = 3
        return
    identity = ReleaseInvocationIdentity(
        run_id=os.environ['PARAWORKS_RELEASE_RUN_ID'],
        profile=os.environ['PARAWORKS_RELEASE_PROFILE'],
        child_id=os.environ['PARAWORKS_RELEASE_CHILD_ID'],
        invocation_hash=os.environ['PARAWORKS_RELEASE_INVOCATION_HASH'],
    )
    event_payload = tuple(json.dumps(asdict(event), sort_keys=True, separators=(',', ':')) for event in _events)
    sidecar = ReleaseEvidenceSidecar(
        schema_version=SIDECAR_SCHEMA_VERSION,
        identity=identity,
        collection_nodeids=_nodeids,
        collection_sha256=canonical_sha256(_nodeids),
        events=tuple(_events),
        lease_events=(),
        pytest_exitstatus=int(exitstatus),
        events_sha256=canonical_sha256(event_payload),
        complete=True,
    )
    value = asdict(sidecar)
    value['lease_events'] = list(_lease_events)
    partial = target.with_suffix(target.suffix + '.partial')
    partial.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')), encoding='utf-8')
    partial.replace(target)


def read_completed_sidecar(path: Path) -> ReleaseEvidenceSidecar:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        sidecar = sidecar_from_json(value)
    except Exception:
        raise ValueError('sidecar_refused') from None
    if sidecar.schema_version != SIDECAR_SCHEMA_VERSION or sidecar.complete is not True:
        raise ValueError('sidecar_refused')
    if sidecar.collection_sha256 != canonical_sha256(sidecar.collection_nodeids):
        raise ValueError('sidecar_refused')
    event_payload = tuple(
        json.dumps(asdict(event), sort_keys=True, separators=(',', ':'))
        for event in sidecar.events
    )
    if sidecar.events_sha256 != canonical_sha256(event_payload):
        raise ValueError('sidecar_refused')
    return sidecar


def validate_completed_sidecar(sidecar: ReleaseEvidenceSidecar, *, expected_identity: ReleaseInvocationIdentity, native_exit_code: int) -> None:
    if sidecar.identity != expected_identity or sidecar.pytest_exitstatus != native_exit_code:
        raise ValueError('sidecar_refused')
