import os

import pytest
from sqlalchemy import create_engine, text

from backend.tests.postgres_isolation import lease_postgres_schema


def test_lease_refuses_non_postgres_locator():
    with (
        pytest.raises(ValueError, match='locator'),
        lease_postgres_schema('sqlite://', run_id='a' * 12, scope_name='unit'),
    ):
        pass


def test_live_schema_lease_is_private_and_removed(monkeypatch):
    url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('release PostgreSQL locator required')
    monkeypatch.setenv('PARAWORKS_RELEASE_RUN_ID', 'a' * 12)
    with lease_postgres_schema(url, run_id='a' * 12, scope_name='live') as lease:
        engine = create_engine(lease.database_url)
        try:
            with engine.begin() as connection:
                connection.execute(text('CREATE TABLE lease_marker (id integer)'))
                assert connection.scalar(text("SELECT current_schema()")) == lease.schema_name
        finally:
            engine.dispose()
        schema_name = lease.schema_name
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            assert connection.scalar(text('SELECT count(*) FROM pg_namespace WHERE nspname=:name'), {'name': schema_name}) == 0
    finally:
        engine.dispose()
