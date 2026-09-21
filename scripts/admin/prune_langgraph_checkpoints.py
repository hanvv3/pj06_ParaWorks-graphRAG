from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.agent_runtime.retention import (  # noqa: E402
    prune_expired_checkpoints,
)
from backend.app.core.config import get_settings  # noqa: E402


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        self.exit(2, f'{self.prog}: invalid arguments\n')


def _prune_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('invalid limit') from None
    if limit < 1 or limit > 1000:
        raise argparse.ArgumentTypeError('invalid limit')
    return limit


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeArgumentParser(
        description='Prune expired ParaWorks LangGraph checkpoints.',
        allow_abbrev=False,
    )
    parser.add_argument('--limit', type=_prune_limit, default=100)
    args = parser.parse_args(argv)

    try:
        result = prune_expired_checkpoints(get_settings(), limit=args.limit)
    except Exception:
        print('checkpoint prune failed', file=sys.stderr)
        return 1

    print(f'cutoff: {result.cutoff.isoformat()}')
    print(f'selected threads: {result.selected_thread_count}')
    print(f'deleted writes: {result.deleted_write_count}')
    print(f'deleted blobs: {result.deleted_blob_count}')
    print(f'deleted checkpoints: {result.deleted_checkpoint_count}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
