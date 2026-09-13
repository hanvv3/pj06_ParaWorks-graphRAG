from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Iterator
from pathlib import Path
from threading import Barrier, Thread
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from backend.app.admin.rag_provider_safety import (
    ProviderSafetyAdminTarget,
    RagProviderSafetyAdminService,
    review_key_material_verifier,
)
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
    load_registered_advisory_capability,
    register_advisory_identity_db,
)
from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService
from backend.app.core.config import get_settings
from backend.tests.test_rag_v2_costs import _snapshot

_REVIEW_KEY = b'task22-external-review-authority-key'
_RUNTIME_KEY = b'task22-runtime-provider-safety-key'


@pytest.fixture
def postgres_admin_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip('PARAWORKS_TEST_POSTGRES_URL is unavailable for Task 22')
    parsed = make_url(database_url)
    if parsed.host != '127.0.0.1' or parsed.port != 55432:
        pytest.fail('Task 22 PostgreSQL checks require disposable 127.0.0.1:55432')
    admin = create_engine(database_url)
    schema = f'rag_task22_{uuid4().hex}'
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA {schema}'))
    query = dict(parsed.query)
    query['options'] = f'-csearch_path={schema},public'
    isolated_url = parsed.set(query=query).render_as_string(hide_password=False)
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', isolated_url)
    get_settings.cache_clear()
    command.upgrade(Config('alembic.ini'), 'head')
    engine = create_engine(isolated_url)
    try:
        yield engine
    finally:
        engine.dispose()
        get_settings.cache_clear()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        admin.dispose()


def _review(target, operation, context):
    payload = {
        'actor_subject_hmac': '4' * 64,
        'expected_context': context,
        'historical_block_acknowledged': False,
        'implementation_plan_reference_hmac': '9' * 64,
        'nonce': str(uuid4()),
        'operation': operation,
        'review_authority_key_id': 'provider-safety-review-v1',
        'schema_version': 'rag-provider-safety-admin-review:v1',
        'successor': None,
        'target': target.review_identity,
    }
    return canonical_json_bytes(
        {
            'hmac_sha256': hmac.new(
                _REVIEW_KEY,
                b'paraworks:provider-safety-admin-review:v1\x00'
                + canonical_json_bytes(payload),
                hashlib.sha256,
            ).hexdigest(),
            'signed_payload': payload,
        }
    )


def _context(value):
    return {
        'authority_uuid': value.authority_uuid,
        'component': value.component,
        'current_family_identity': list(value.family_identity),
        'current_state': value.state,
        'current_state_version': value.state_version,
        'designated_environment_id': value.designated_environment_id,
        'global_safety_generation': value.global_safety_generation,
        'has_historical_blocker': value.has_historical_blocker,
    }


def _admin(engine: Engine, tmp_path: Path):
    with engine.begin() as connection:
        register_advisory_identity_db(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
    with engine.connect() as connection:
        capability = load_registered_advisory_capability(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
    target = ProviderSafetyAdminTarget.build(
        kind='live_validation',
        database_url=engine.url.render_as_string(hide_password=False),
        latch_path=tmp_path / 'live-validation.json',
        designated_environment_id='task22-live-validation',
        review_secret=_REVIEW_KEY,
    )
    runtime = RagProviderSafetyService(
        latch_path=target.latch_path,
        identity_secret=_RUNTIME_KEY,
        designated_environment_id=target.designated_environment_id,
        advisory_capability=capability,
    )
    admin = RagProviderSafetyAdminService(
        target=target,
        connection_factory=engine.connect,
        provider_safety=runtime,
        snapshots=(_snapshot('query_embedding'), _snapshot('answer_generation')),
        runtime_identity_secret=_RUNTIME_KEY,
        review_secret=_REVIEW_KEY,
        review_key_id='provider-safety-review-v1',
        implementation_plan_reference_hmac='9' * 64,
        successor_registry={},
        review_key_registry={
            'provider-safety-review-v1': review_key_material_verifier(_REVIEW_KEY)
        },
    )
    return target, runtime, admin


def test_postgresql_admin_serializes_reviewed_cas_and_appends_one_generation(
    postgres_admin_db: Engine,
    tmp_path: Path,
) -> None:
    target, runtime, admin = _admin(postgres_admin_db, tmp_path)
    admin.initialize(_review(target, 'provider-safety-init', None))
    with postgres_admin_db.connect() as connection:
        context = runtime.review_context(connection, 'answer_generation')
    raw = _review(
        target,
        'provider-safety-mark-rebind-required',
        _context(context),
    )
    start = Barrier(3)
    outcomes: list[str] = []

    def mutate() -> None:
        start.wait()
        try:
            admin.mark_rebind_required(raw)
            outcomes.append('committed')
        except Exception:
            outcomes.append('refused')

    workers = [Thread(target=mutate), Thread(target=mutate)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(10)
        assert not worker.is_alive()
    assert sorted(outcomes) == ['committed', 'refused']
    with postgres_admin_db.connect() as connection:
        generations = connection.execute(
            text(
                'SELECT global_safety_generation FROM '
                'rag_provider_safety_transitions ORDER BY global_safety_generation'
            )
        ).scalars().all()
    assert generations == [0, 1]
