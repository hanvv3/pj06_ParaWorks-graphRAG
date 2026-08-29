from __future__ import annotations

import argparse

from backend.app.core.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.review.auto_review_source_reconciliation import (
    AutoReviewSourceReconciliationService,
    SourceReconciliationResult,
)

SYSTEM_ACTOR = 'system:local-auto-review-source-reconciler'


def format_aggregate_result(result: SourceReconciliationResult) -> str:
    return (
        f'actor={SYSTEM_ACTOR} stale={result.stale_count} '
        f'reconciled={result.reconciled_count} repaired={result.repaired_count} '
        f'ambiguous={result.ambiguous_count} remaining={result.remaining_count} '
        f'failures={result.failure_count} readiness={str(result.readiness).lower()}'
    )


def command_exit_code(result: SourceReconciliationResult) -> int:
    return 0 if result.readiness and result.remaining_count == 0 else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        choices=('status', 'recover', 'repair-current-document-versions'),
    )
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        with SessionLocal() as db:
            service = AutoReviewSourceReconciliationService(
                db, settings=settings
            )
            if args.command == 'status':
                result = service.status(limit=args.limit)
            elif args.command == 'recover':
                result = service.recover_stale_sources(limit=args.limit)
            else:
                result = service.repair_current_document_versions(
                    limit=args.limit
                )
    except (RuntimeError, ValueError):
        return 2
    print(format_aggregate_result(result))
    return command_exit_code(result)


if __name__ == '__main__':
    raise SystemExit(main())
