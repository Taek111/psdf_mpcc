#!/usr/bin/env bash
# DoN 0.70:0.01:0.87 x four controllers x 50 matched seeds, sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/rectangle_benchmark_suite.py" don_sweep "$@"
