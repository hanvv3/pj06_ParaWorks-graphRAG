"""Explicit two-database local S3 harness; never loads credentials from files."""

import os

from backend.tests.slack_offline_runner import main

if __name__ == '__main__':
    required = [
        'PARAWORKS_TEST_POSTGRES_URL',
        *(f'PARAWORKS_TEST_NEO4J_{name}' for name in ('URI', 'USER', 'PASSWORD')),
    ]
    if not all(os.environ.get(name) for name in required):
        raise SystemExit('explicit disposable PG and Neo4j process settings required')
    raise SystemExit(
        main(
            postgres_url=os.environ[required[0]],
            neo4j={
                name: os.environ[f'PARAWORKS_TEST_NEO4J_{name}']
                for name in ('URI', 'USER', 'PASSWORD')
            },
        )
    )
