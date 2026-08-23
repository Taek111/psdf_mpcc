# Benchmark report source notes

## Reporting job

- Question: compare the real-time performance and driving stability of four PSDF controllers, and select an effective `covariance_growth_scale` for PSDF-MPCC-CC-MF.
- Audience: technical.
- Scope: `maze / rectangle / differential_drive / A* / 60 s / localization error disabled`.
- Decision rule: retain only successful, collision-free MF runs, then rank by q05 clearance (descending), solver failures, safe-stops, 100 ms deadline-miss rate, controller p95 time, and driving time (all ascending after q05).

## Source inventory

- `benchmark_runs/maze_rectangle_nominal/scale_sweep.csv`: one reviewed aggregate row for each MF scale from 0.1 through 1.0.
- `benchmark_runs/maze_rectangle_nominal/optimizer_comparison.csv`: the three baseline optimizers plus the selected MF result.
- `benchmark_runs/maze_rectangle_nominal/<run_id>/data/trial_history_*.csv`: raw per-step controller, solver, command, and solver-success observations.
- `benchmark_runs/maze_rectangle_nominal/<run_id>/data/pose_sdf_*.csv`: raw per-pose signed-distance observations.
- `benchmark_runs/maze_rectangle_nominal/<run_id>/data/covariance_trajectory.csv`: raw MF covariance trajectories.
- `run_optimizer_benchmark.py`: configuration, metric definitions, selection rule, and process-isolation implementation.

## Metric definitions

- Controller time: wall-clock duration of the full `generate_control_input()` call, measured with `time.perf_counter()`.
- Deadline miss: controller time strictly greater than 0.1 s; rate denominator is the number of controller calls logged in that run.
- Driving time: simulated elapsed time only for a run whose outcome is `success`.
- q05/min clearance: 5th percentile and minimum of the raw `sdf_value` sequence; a negative minimum is marked as collision.
- RMS delta v / omega: square root of the mean squared first difference of the commanded linear/angular velocity sequence.
- Solver failure: a logged `solver_success=False` step.
- Safe-stop: the controller runtime counter for fallback zero/safe control plans.

## Validation and robustness checks

- All 13 planned runs have a non-empty raw trial log, a non-empty pose/SDF log, and a per-run summary row.
- Controller p95 time, q05 clearance, and solver-failure count were independently recomputed from every run's raw logs; the maximum discrepancy from the summary CSV was 0.
- Each MF scale was executed in a fresh Python process. This prevents a regenerated acados shared library with a repeated filename from being reused across scale settings.
- The MF covariance-trajectory maxima change across the scale sweep, confirming that the scale-dependent coefficients reached separate solver builds.
- The report is descriptive: there is one deterministic run per controller/scale, so it does not estimate run-to-run variance or statistical confidence intervals.

## Chart map

- `scale_q05`: ordered scale effect on q05 clearance; single-series bar chart; fields `scale_label`, `q05_mm`; scale summary source; blue single-root palette.
- `optimizer_p95`: category comparison of full-controller p95 time; single-series horizontal bar chart; fields `method`, `p95_ms`; optimizer comparison source; blue single-root palette.

## Audience-structure map

- Title: `PSDF 제어기 벤치마크와 MF 공분산 스케일 분석`.
- Technical summary: `Technical summary`.
- Key findings with visual evidence: `Scale 0.7 gave the best collision-free lower-tail clearance` and `MF improved clearance but did not meet the 100 ms real-time target`.
- Scope, data, and metric definitions: `Scope and metric definitions`.
- Methodology: `Reproducible execution and validation`.
- Limitations, uncertainty, and robustness checks: `What this single-run benchmark can and cannot establish`.
- Recommended next steps: `Recommended next steps`.
- Further questions: `Further questions`.
