from __future__ import annotations

import pytest
from sqlalchemy import event

from backend.app.rag import release_review as review
from backend.app.rag.release_ledger import RagReleaseLedgerError
from backend.tests.test_rag_release_ledger_review_q import (
    _release_test_seam,  # noqa: F401
)
from backend.tests.test_rag_release_ledger_round4 import harness as harness


def test_raw_self_attested_manifest_cannot_authorize_a_case_claim(harness):
    harness.source_verifier = None
    before = harness.records('authorization')
    writes = []

    def record(_connection, _cursor, sql, _params, _context, _many):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            writes.append(sql)

    event.listen(harness.engine, 'before_cursor_execute', record)
    try:
        with pytest.raises(RagReleaseLedgerError):
            harness.claim()
    finally:
        event.remove(harness.engine, 'before_cursor_execute', record)
    assert writes == []
    assert harness.records('authorization') == before
    assert harness.records('agent_run') == []
    assert harness.records('cost_component') == []


def test_default_approved_source_verifier_is_unavailable_before_task24_b():
    with pytest.raises(RagReleaseLedgerError):
        review.require_approved_case_source(
            None,
            manifest=None,
            source_binding=None,
            authorization={},
            identity_secret=b'test-only-secret-material-32-bytes',
        )
