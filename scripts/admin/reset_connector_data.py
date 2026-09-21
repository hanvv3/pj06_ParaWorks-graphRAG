from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    from backend.app.admin.data_reset import reset_connector_derived_data
    from backend.app.core.config import get_settings
    from backend.app.db.session import SessionLocal

    parser = argparse.ArgumentParser(description='Reset ParaWorks connector-derived local/dev data.')
    parser.add_argument('--execute', action='store_true', help='Delete data instead of only printing counts.')
    parser.add_argument('--confirm', action='store_true', help='Required with --execute.')
    args = parser.parse_args()

    settings = get_settings()
    try:
        with SessionLocal() as db:
            result = reset_connector_derived_data(
                db,
                settings=settings,
                dry_run=not args.execute,
                confirm=args.confirm,
            )
    except SQLAlchemyError as exc:
        raise SystemExit(f'Could not connect to the configured database: {exc}') from exc

    mode = 'dry-run' if result.dry_run else 'executed'
    print(f'connector-derived reset {mode}')
    print('preserved:', ', '.join(result.preserved_tables))
    for table_name, count in sorted(result.deleted_counts.items()):
        print(f'{table_name}: {count}')


if __name__ == '__main__':
    main()
