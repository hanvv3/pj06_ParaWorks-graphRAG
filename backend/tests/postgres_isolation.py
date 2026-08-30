from __future__ import annotations

import os
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from backend.tests.release_evidence_plugin import record_lease_event

_SAFE = re.compile(r'^paraworks_c5t16_[0-9a-f]{12}_schema_[a-z0-9_]+_[0-9a-f]{8}$', re.ASCII)


@dataclass(frozen=True, slots=True)
class PostgresSchemaLease:
    base_url: str
    database_url: str
    schema_name: str


@contextmanager
def lease_postgres_schema(base_url: str, *, run_id: str, scope_name: str) -> Iterator[PostgresSchemaLease]:
    url = make_url(base_url)
    expected_database = os.getenv('PARAWORKS_RELEASE_EXPECTED_DATABASE')
    expected_role = os.getenv('PARAWORKS_RELEASE_EXPECTED_ROLE')
    if url.get_backend_name() != 'postgresql' or not url.database or not url.database.endswith('_test'):
        raise ValueError('postgres_locator_refused')
    if expected_database and url.database != expected_database:
        raise ValueError('postgres_locator_refused')
    scope = re.sub(r'[^a-z0-9]+', '_', scope_name.lower()).strip('_')[:16] or 'test'
    schema = f'paraworks_c5t16_{run_id}_schema_{scope}_{secrets.token_hex(4)}'
    if len(schema) > 63 or not _SAFE.fullmatch(schema):
        raise ValueError('schema_identity_refused')
    engine = create_engine(base_url)
    quoted = engine.dialect.identifier_preparer.quote(schema)
    lease_identity = f'{run_id}:{schema}'
    created = False
    try:
        with engine.begin() as connection:
            current_database, current_role = connection.execute(text('SELECT current_database(), current_user')).one()
            if current_database != url.database or (expected_role and current_role != expected_role):
                raise ValueError('postgres_locator_refused')
            connection.execute(text(f'CREATE SCHEMA {quoted}'))
            created = True
        record_lease_event(lease_identity=lease_identity, state='created')
        leased = url.update_query_dict({'options': f'-csearch_path={schema},public'})
        yield PostgresSchemaLease(base_url, leased.render_as_string(hide_password=False), schema)
    finally:
        if created:
            with engine.begin() as connection:
                connection.execute(text("SET LOCAL lock_timeout = '5s'"))
                connection.execute(text("SET LOCAL statement_timeout = '30s'"))
                connection.execute(text(f'DROP SCHEMA {quoted} CASCADE'))
                if connection.scalar(text('SELECT count(*) FROM pg_namespace WHERE nspname=:name'), {'name': schema}) != 0:
                    raise RuntimeError('schema_cleanup_failed')
            record_lease_event(lease_identity=lease_identity, state='dropped')
        engine.dispose()


@contextmanager
def leased_database_environment(database_url: str) -> Iterator[None]:
    names = ('PARAWORKS_DEMO_MODE', 'PARAWORKS_DATABASE_URL', 'DATABASE_URL')
    prior = {name: os.environ.get(name) for name in names}
    try:
        os.environ['PARAWORKS_DEMO_MODE'] = 'false'
        os.environ['PARAWORKS_DATABASE_URL'] = database_url
        os.environ['DATABASE_URL'] = database_url
        from backend.app.core.config import get_settings
        get_settings.cache_clear()
        yield
    finally:
        from backend.app.core.config import get_settings
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()
