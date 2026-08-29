from backend.app.admin.auto_review_source_reconciliation import (
    command_exit_code,
    format_aggregate_result,
)
from backend.app.review.auto_review_source_reconciliation import (
    SourceReconciliationResult,
)


def test_source_reconciliation_admin_has_fixed_actor_aggregate_output_and_exit_codes() -> None:
    clean = SourceReconciliationResult(readiness=True)
    retained = SourceReconciliationResult(
        stale_count=1,
        remaining_count=1,
        readiness=False,
    )

    output = format_aggregate_result(retained)

    assert 'system:local-auto-review-source-reconciler' in output
    assert 'stale=1' in output
    assert 'remaining=1' in output
    assert 'source_id' not in output
    assert command_exit_code(clean) == 0
    assert command_exit_code(retained) == 3
