from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.models import (
    AgentRun,
    AutoReviewExtractionCall,
    AutoReviewValidation,
    AutoReviewValidationCall,
)


@dataclass(frozen=True, slots=True)
class AutoReviewCallRecoveryResult:
    extraction_pending: int = 0
    extraction_recovered: int = 0
    validation_pending: int = 0
    validation_recovered: int = 0
    extraction_remaining: int = 0
    validation_remaining: int = 0
    failure_count: int = 0

    @property
    def ready(self) -> bool:
        return (
            self.extraction_remaining == 0
            and self.validation_remaining == 0
            and self.failure_count == 0
        )


class AutoReviewCallRecoveryService:
    """Finalizes expired provider ledgers without retrying either provider."""

    def __init__(self, *, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def status(self, *, limit: int = 100) -> AutoReviewCallRecoveryResult:
        self._validate_limit(limit)
        extraction, validation = self._counts()
        return AutoReviewCallRecoveryResult(
            extraction_pending=min(extraction, limit),
            validation_pending=min(validation, limit),
            extraction_remaining=extraction,
            validation_remaining=validation,
        )

    def recover(self, *, limit: int = 100) -> AutoReviewCallRecoveryResult:
        self._validate_limit(limit)
        now = datetime.now(UTC)
        extraction_ids, validation_ids = self._expired_ids(now=now, limit=limit)
        extraction_recovered = 0
        validation_recovered = 0
        failures = 0
        for call_id in extraction_ids:
            try:
                extraction_recovered += self._recover_extraction(
                    call_id=call_id, now=now
                )
            except SQLAlchemyError:
                failures += 1
        for call_id in validation_ids:
            try:
                validation_recovered += self._recover_validation(
                    call_id=call_id, now=now
                )
            except SQLAlchemyError:
                failures += 1
        extraction_remaining, validation_remaining = self._counts()
        return AutoReviewCallRecoveryResult(
            extraction_pending=len(extraction_ids),
            extraction_recovered=extraction_recovered,
            validation_pending=len(validation_ids),
            validation_recovered=validation_recovered,
            extraction_remaining=extraction_remaining,
            validation_remaining=validation_remaining,
            failure_count=failures,
        )

    def cancel_workflow(self, *, workflow_thread_id: str) -> None:
        if not workflow_thread_id.strip():
            raise ValueError('workflow_thread_id is required')
        now = datetime.now(UTC)
        with self._session_factory() as db:
            extraction_ids = tuple(db.scalars(
                select(AutoReviewExtractionCall.id).where(
                    AutoReviewExtractionCall.workflow_thread_id
                    == workflow_thread_id,
                    AutoReviewExtractionCall.status == 'claimed',
                ).order_by(AutoReviewExtractionCall.id)
            ).all())
            validation_ids = tuple(db.scalars(
                select(AutoReviewValidationCall.id).where(
                    AutoReviewValidationCall.workflow_thread_id
                    == workflow_thread_id,
                    AutoReviewValidationCall.status == 'claimed',
                ).order_by(AutoReviewValidationCall.id)
            ).all())
            db.rollback()
        for call_id in extraction_ids:
            self._recover_extraction(call_id=call_id, now=now, force=True)
        for call_id in validation_ids:
            self._recover_validation(call_id=call_id, now=now, force=True)

    def _expired_ids(
        self, *, now: datetime, limit: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        with self._session_factory() as db:
            extraction = tuple(db.scalars(
                select(AutoReviewExtractionCall.id).where(
                    AutoReviewExtractionCall.status == 'claimed',
                    AutoReviewExtractionCall.lease_expires_at.is_not(None),
                    AutoReviewExtractionCall.lease_expires_at <= now,
                ).order_by(AutoReviewExtractionCall.id).limit(limit)
            ).all())
            validation = tuple(db.scalars(
                select(AutoReviewValidationCall.id).where(
                    AutoReviewValidationCall.status == 'claimed',
                    AutoReviewValidationCall.lease_expires_at.is_not(None),
                    AutoReviewValidationCall.lease_expires_at <= now,
                ).order_by(AutoReviewValidationCall.id).limit(limit)
            ).all())
            db.rollback()
        return extraction, validation

    def _recover_extraction(
        self, *, call_id: int, now: datetime, force: bool = False
    ) -> int:
        with self._session_factory() as db, db.begin():
            call = db.scalar(select(AutoReviewExtractionCall).where(
                AutoReviewExtractionCall.id == call_id
            ).with_for_update())
            if (
                call is None
                or call.status != 'claimed'
                or call.lease_expires_at is None
                or (not force and call.lease_expires_at > now)
            ):
                return 0
            attempted = call.provider_attempt_count == 1
            call.status = 'failed'
            call.lease_token = None
            call.terminal_at = now
            call.charged_input_tokens = call.reserved_input_tokens if attempted else 0
            call.charged_output_tokens = call.reserved_output_tokens if attempted else 0
            call.charged_cost_usd = call.reserved_cost_usd if attempted else Decimal('0')
            call.result_kind = None
            call.result_candidate_count = None
            call.result_candidate_set_hmac = None
            run = db.scalar(select(AgentRun).where(
                AgentRun.id == call.agent_run_id
            ).with_for_update())
            if run is not None:
                run.status = 'failed'
                run.input_tokens = call.charged_input_tokens
                run.output_tokens = call.charged_output_tokens
                run.total_tokens = run.input_tokens + run.output_tokens
                run.estimated_cost_usd = float(call.charged_cost_usd)
                run.completed_at = now
                run.metadata_ = {
                    **(run.metadata_ or {}),
                    'failure_reason_code': 'lease_expired',
                }
        return 1

    def _recover_validation(
        self, *, call_id: int, now: datetime, force: bool = False
    ) -> int:
        with self._session_factory() as db, db.begin():
            call = db.scalar(select(AutoReviewValidationCall).where(
                AutoReviewValidationCall.id == call_id
            ).with_for_update())
            if (
                call is None
                or call.status != 'claimed'
                or call.lease_expires_at is None
                or (not force and call.lease_expires_at > now)
            ):
                return 0
            attempted = call.provider_attempt_count == 1
            call.status = 'failed'
            call.lease_token = None
            call.terminal_at = now
            call.charged_input_tokens = call.reserved_input_tokens if attempted else 0
            call.charged_output_tokens = call.reserved_output_tokens if attempted else 0
            call.charged_cost_usd = call.reserved_cost_usd if attempted else Decimal('0')
            children = tuple(db.scalars(select(AutoReviewValidation).where(
                AutoReviewValidation.validation_call_id == call.id,
                AutoReviewValidation.workflow_thread_id == call.workflow_thread_id,
            ).with_for_update()).all())
            for child in children:
                if child.status == 'claimed':
                    child.status = 'failed'
                    child.policy_reason_codes = ['validator_unavailable']
                    child.input_tokens = 0
                    child.output_tokens = 0
                    child.estimated_cost_usd = Decimal('0')
                    child.completed_at = now
        return 1

    def _counts(self) -> tuple[int, int]:
        with self._session_factory() as db:
            extraction = int(db.scalar(select(func.count()).select_from(
                AutoReviewExtractionCall
            ).where(AutoReviewExtractionCall.status == 'claimed')) or 0)
            validation = int(db.scalar(select(func.count()).select_from(
                AutoReviewValidationCall
            ).where(AutoReviewValidationCall.status == 'claimed')) or 0)
            db.rollback()
        return extraction, validation

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('limit must be between 1 and 1000')


def format_result(result: AutoReviewCallRecoveryResult) -> str:
    return (
        f'extraction_pending={result.extraction_pending} '
        f'extraction_recovered={result.extraction_recovered} '
        f'extraction_remaining={result.extraction_remaining} '
        f'validation_pending={result.validation_pending} '
        f'validation_recovered={result.validation_recovered} '
        f'validation_remaining={result.validation_remaining} '
        f'failures={result.failure_count}'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('status', 'recover'))
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        if settings.resolved_database_url().split(':', 1)[0] != 'postgresql+psycopg':
            return 2
        service = AutoReviewCallRecoveryService(session_factory=SessionLocal)
        result = (
            service.status(limit=args.limit)
            if args.command == 'status'
            else service.recover(limit=args.limit)
        )
    except (RuntimeError, ValueError):
        return 2
    except SQLAlchemyError:
        result = AutoReviewCallRecoveryResult(
            extraction_remaining=1,
            validation_remaining=1,
            failure_count=1,
        )
    print(format_result(result))
    return 0 if result.ready else 3


if __name__ == '__main__':
    raise SystemExit(main())
