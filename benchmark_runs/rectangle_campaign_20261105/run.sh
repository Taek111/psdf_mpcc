#!/usr/bin/env bash
set -euo pipefail
cd /home/taekwon/projects/psdf_mpcc
export PYTHON_BIN=/home/taekwon/projects/psdf_mpcc/.venv/bin/python
export BENCHMARK_NUM_THREADS=6
bash scripts/run_rectangle_don_benchmark.sh --results-dir benchmark_runs/rectangle_don_100_20261105_current_sf
bash scripts/run_rectangle_don_sweep_benchmark.sh --results-dir benchmark_runs/rectangle_don_sweep_50_20261105_current_sf --reuse-results benchmark_runs/rectangle_don_100_20261105_current_sf
printf 'experiments_complete\n' > benchmark_runs/rectangle_campaign_20261105/experiments_complete
