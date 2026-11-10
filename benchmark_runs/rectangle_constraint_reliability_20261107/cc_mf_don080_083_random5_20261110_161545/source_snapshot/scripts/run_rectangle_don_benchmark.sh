#!/usr/bin/env bash
# 0.075 / 0.1125 m x four controllers x 100 matched seeds, sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/rectangle_benchmark_suite.py" don "$@"
