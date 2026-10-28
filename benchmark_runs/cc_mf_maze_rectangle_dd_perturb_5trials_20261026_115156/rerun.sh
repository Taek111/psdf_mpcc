#!/usr/bin/env bash
set -euo pipefail
cd /home/taek111/projects/psdf_mpcc
if [[ $# != 1 ]]; then
    printf 'Usage: bash rerun.sh NEW_RESULTS_DIR\n' >&2
    exit 2
fi
export ACADOS_SOURCE_DIR=/home/taek111/projects/acados
export LD_LIBRARY_PATH=/home/taek111/projects/acados/lib
export MPLBACKEND=Agg
exec /home/taek111/anaconda3/envs/pp/bin/python -u test_nmpc.py \
    --config /home/taek111/projects/psdf_mpcc/benchmark_runs/cc_mf_maze_rectangle_dd_perturb_5trials_20261026_115156/config.yaml \
    --trials 5 --results-dir "$1"
