import pytest
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_call_recovery import (
    AutoReviewCallRecoveryResult,
    AutoReviewCallRecoveryService,
    format_result,
)


def test_empty_call_ledger_is_ready_and_replay_safe(db_session: Session) -> None:
    factory = sessionmaker(bind=db_session.get_bind(), expire_on_commit=False)
    service = AutoReviewCallRecoveryService(session_factory=factory)

    first = service.recover(limit=100)
    second = service.recover(limit=100)

    assert first == second == AutoReviewCallRecoveryResult()
    assert first.ready is True


def test_call_recovery_output_is_aggregate_only() -> None:
    output = format_result(AutoReviewCallRecoveryResult(
        extraction_pending=2,
        extraction_recovered=1,
        extraction_remaining=1,
        validation_pending=3,
        validation_recovered=2,
        validation_remaining=1,
        failure_count=0,
    ))

    assert output == (
        'extraction_pending=2 extraction_recovered=1 extraction_remaining=1 '
        'validation_pending=3 validation_recovered=2 validation_remaining=1 '
        'failures=0'
    )
    assert 'call_id' not in output
    assert 'output' not in output


@pytest.mark.parametrize('limit', (0, -1, 1001, True))
def test_call_recovery_refuses_unbounded_limits(
    db_session: Session, limit: int
) -> None:
    factory = sessionmaker(bind=db_session.get_bind(), expire_on_commit=False)
    service = AutoReviewCallRecoveryService(session_factory=factory)

    with pytest.raises(ValueError, match='limit'):
        service.status(limit=limit)
