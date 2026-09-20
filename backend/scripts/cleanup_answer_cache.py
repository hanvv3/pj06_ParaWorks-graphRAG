"""Bounded operator cleanup, including while answer-cache rollout is disabled.

Run with deployment settings: python -m backend.scripts.cleanup_answer_cache --limit 100
Schedule independently of traffic; repeat until deleted_count is zero.
"""

import argparse

from backend.app.core.config import get_settings
from backend.app.rag.answer_cache_store import create_answer_cache


def main():
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error('--limit must be between 1 and 1000')
    settings = get_settings()
    engine = create_engine(settings.resolved_database_url())
    try:
        cache = create_answer_cache(
            engine=engine, settings=settings, validator=None, enabled=True
        )
        print(f'deleted_count={cache.cleanup(limit=args.limit)}')
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
