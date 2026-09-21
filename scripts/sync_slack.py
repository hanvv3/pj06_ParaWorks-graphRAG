"""Explicit Slack ingestion utility; failures never print provider payloads."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_sync():
    from scripts.local_runtime import load_environment

    os.environ.update(load_environment(ROOT))
    os.chdir(ROOT)
    from backend.app.connectors.factory import get_sync_connector
    from backend.app.core.config import get_settings
    from backend.app.db.session import SessionLocal
    from backend.app.ingestion.sync import sync_connector_events

    with SessionLocal() as db:
        settings = get_settings()
        connector = get_sync_connector('slack', settings, db=db)
        result = sync_connector_events(
            db=db,
            connector=connector,
            # The shared ingestion boundary skips automatic paid reindexing
            # without an embedding key. Keep the live connector credentials;
            # do not enqueue a fake job that can never complete.
            settings=settings.model_copy(update={'openai_api_key': None}),
        )
        return {
            'status': result.status,
            'fetched_count': result.fetched_events,
            'created_count': result.created_review_items,
            'skipped_count': result.skipped_events,
            'analysis': 'use_application_review_workflow',
            'embedding': 'not_dispatched',
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Explicit Slack ingestion; no analysis or paid embedding dispatch.'
    )
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            'No writes: use --execute only for an authorized Slack source; use the application for review and indexing.'
        )
        return 2
    try:
        report = run_sync()
    except Exception:
        print(json.dumps({'error': 'slack_sync_or_analysis_failed'}), file=sys.stderr)
        return 1
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
