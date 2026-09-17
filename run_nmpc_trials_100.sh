#!/usr/bin/env bash
# Run 100 trials per combination: 2 maps x 3 methods = 600 trials.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_nmpc_trials_25.sh [RESULTS_DIR]

Maps: maze, oblique_maze; methods: psdf, dcbf, obca.
Kinematics: differential_drive (DD); footprint: rectangle; planner: A*.
Runs sequentially, with 100 trials per combination and start seeds 0-99.

RESULTS_DIR defaults to a new benchmark_runs/maze_oblique_100_<timestamp>_<pid>
directory. Relative paths are resolved from the project root.
Existing trial CSVs are never overwritten; resuming a batch is unsupported.

Activate the Python environment containing the project dependencies first.
Set PYTHON_BIN to select an interpreter (default: python).
EOF
}

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && ( -z "$1" || "$1" == -* ) ) ]]; then
    usage >&2
    exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

CONFIG_PATH="$PROJECT_DIR/config/config_nmpc_trials_maze_oblique.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    printf 'Configuration not found: %s\n' "$CONFIG_PATH" >&2
    exit 1
fi

RESULTS_DIR="${1:-benchmark_runs/maze_oblique_100_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "$RESULTS_DIR" != /* ]]; then
    RESULTS_DIR="$PROJECT_DIR/$RESULTS_DIR"
fi

# Keep plotting imports usable on a machine without a display.
export MPLBACKEND=Agg
printf 'Running 6 configurations x 100 trials = 600 trials sequentially.\n'
printf 'Results: %s\nSummary: %s/summary.csv\n' "$RESULTS_DIR" "$RESULTS_DIR"

# The CLI count explicitly selects 100 trials per configuration. exec keeps Ctrl+C
# handling in test_nmpc.py, which saves the interrupted trial before stopping.
exec "$PYTHON_BIN" -u "$PROJECT_DIR/test_nmpc.py" \
    --config "$CONFIG_PATH" \
    --trials 100 \
    --results-dir "$RESULTS_DIR" \
    --no-animation \
    --no-plots
