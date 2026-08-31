from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    not os.environ.get('PARAWORKS_TEST_POSTGRES_URL'),
    reason='disposable PostgreSQL gate is not configured',
)
def test_two_argument_postgres_advisory_lock_gate():
    """Reserved for the explicitly configured disposable PostgreSQL gate."""
    pytest.skip('Task 12 never invents or opens PostgreSQL credentials')
