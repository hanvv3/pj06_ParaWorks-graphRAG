from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.agent_runtime.bootstrap import (  # noqa: E402
    CheckpointBootstrapError,
    bootstrap_langgraph_checkpointer,
)
from backend.app.core.config import get_settings  # noqa: E402


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f'{self.prog}: invalid arguments\n')


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeArgumentParser(
        description='Bootstrap the ParaWorks LangGraph PostgreSQL checkpointer.',
    )
    parser.add_argument('--confirm-backup', action='store_true')
    args = parser.parse_args(argv)
    if not args.confirm_backup:
        parser.error('--confirm-backup is required')

    try:
        result = bootstrap_langgraph_checkpointer(
            get_settings(),
            backup_confirmed=True,
        )
    except CheckpointBootstrapError:
        print('checkpoint bootstrap failed', file=sys.stderr)
        return 1

    print(f'component: {result.component}')
    print(f'package version: {result.package_version}')
    print(f'schema revision: {result.schema_revision}')
    print(f'applied at: {result.applied_at.isoformat()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
