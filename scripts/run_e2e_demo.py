"""Retired unsafe legacy demo. Use the isolated, offline S-3 demo workflow."""

import sys


def main() -> int:
    print(
        'run_e2e_demo.py is retired: it used stale APIs and could mutate the '
        'configured database. Use scripts/start-smoke.ps1 for the UI, or the '
        'documented S-3 isolated synthetic integration tests.',
        file=sys.stderr,
    )
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
