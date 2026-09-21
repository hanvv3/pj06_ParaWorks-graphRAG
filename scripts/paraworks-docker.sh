#!/usr/bin/env bash
set -eu
printf '%s\n' 'This legacy launcher is retired. Use scripts/start.ps1, stop.ps1 and status.ps1 on Windows.' >&2
printf '%s\n' 'Cross-platform process ownership is not implemented; no processes or Docker resources were changed.' >&2
exit 2
