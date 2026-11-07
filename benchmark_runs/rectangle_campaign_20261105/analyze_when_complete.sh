#!/usr/bin/env bash
set -euo pipefail
cd /home/taekwon/projects/psdf_mpcc
while [ ! -f benchmark_runs/rectangle_campaign_20261105/experiments_complete ]; do
  sleep 30
done
codex exec -m gpt-6-astra -c 'model_reasoning_effort="ultra"' -s danger-full-access -C /home/taekwon/projects/psdf_mpcc -o benchmark_runs/rectangle_campaign_20261105/analysis_final.txt - < benchmark_runs/rectangle_campaign_20261105/analysis_prompt.txt > benchmark_runs/rectangle_campaign_20261105/analysis_cli.log 2>&1
printf 'analysis_complete\n' > benchmark_runs/rectangle_campaign_20261105/analysis_complete
