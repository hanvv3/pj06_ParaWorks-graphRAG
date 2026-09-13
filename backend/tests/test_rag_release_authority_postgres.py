from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv('PARAWORKS_TEST_POSTGRES_URL'),
    reason='PARAWORKS_TEST_POSTGRES_URL is required for physical release-authority verification',
)


def test_release_authority_postgres_contract_is_environment_gated() -> None:
    """Task 23 registers this module before destructive disposable-DB coverage is enabled."""
    assert os.environ['PARAWORKS_TEST_POSTGRES_URL'].startswith('postgresql')

