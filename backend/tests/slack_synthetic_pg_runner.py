"""S2 isolated PostgreSQL run; credentials are supplied only by the process."""

import os

from backend.tests.slack_offline_runner import main

if __name__ == '__main__':
    locator = os.environ.get('PARAWORKS_TEST_POSTGRES_URL')
    if not locator:
        raise SystemExit(
            'PARAWORKS_TEST_POSTGRES_URL is required; no implicit database'
        )
    raise SystemExit(main(postgres_url=locator))
