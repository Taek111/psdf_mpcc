# Rectangle benchmark: 실행 및 후속 LLM 분석 context

이 문서는 앞으로 실행할 두 실험의 설계, 저장 데이터의 의미, 분석 절차를 정의한다. 결과 보고서가 아니며, 파일이 생성됐다는 이유만으로 실험이 완료됐다고 판단하지 않는다. 분석을 맡은 Agent는 이 문서와 실제 실행 결과 디렉토리를 함께 읽고, 주행 실험을 새로 실행하지 않는다.

기준 실험은 저장소의 `benchmark_results_old.md` 중 **「Rectangle 너비별 주행 benchmark (0.005 m 간격)」, 실행일 2026-11-0:3 (KST)** 절이다. 같은 파일의 다른 날짜·제목에 있는 60초 trial, 4초 정체 창, 일부 너비의 초기 clearance 10 mm 조건을 섞지 않는다. 현재 실험은 기준 실험의 주행 조건을 따르지만, computation time 정의와 진단 수집 방식은 아래와 같이 변경했다.

## 1. 비교할 실험과 완료 기준

| 실험 | Rectangle 너비 W (m) | Controller | 반복 | 예상 trial 수 |
|---|---|---|---|---:|
| A: DoN 두 지점의 주행 정량 비교 | 0.075, 0.1125 | MPCC, CC-ECS, CC-SF, CC-MF | 각 너비·controller당 seed 0–99, 100회 | 800 |
| B: DoN sweep 성공률 | DoN 0.70–0.87, 양 끝 포함, 간격 0.01; W=DoN×0.150 m | 같은 네 방법 | 각 DoN·controller당 seed 0–49, 50회 | 3,600 |

여기서 nominal **DoN = robot width / 직선 통로 너비 0.150 m**로 표시한다. A의 0.075 m는 DoN 0.5, 0.1125 m는 DoN 0.75다. 요청에 적힌 첫 너비 `0.75`는 이 비율에 따라 `0.075 m`로 해석했다. Robot 길이도 0.150 m이지만 DoN의 분모는 통로 너비이며, 회전 구간에는 직선 통로와 다른 기하학적 여유가 있다.

B는 다음 18개 지점을 실행한다. 너비를 1 mm 단위로 반올림하거나 이전 요청의 비균일 너비 목록으로 바꾸지 않는다.

| DoN | W (m) | DoN | W (m) | DoN | W (m) |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.1050 | 0.76 | 0.1140 | 0.82 | 0.1230 |
| 0.71 | 0.1065 | 0.77 | 0.1155 | 0.83 | 0.1245 |
| 0.72 | 0.1080 | 0.78 | 0.1170 | 0.84 | 0.1260 |
| 0.73 | 0.1095 | 0.79 | 0.1185 | 0.85 | 0.1275 |
| 0.74 | 0.1110 | 0.80 | 0.1200 | 0.86 | 0.1290 |
| 0.75 | 0.1125 | 0.81 | 0.1215 | 0.87 | 0.1305 |

A와 B는 **DoN 0.75 / W=0.1125 m**에서 겹친다. A는 너비당 400건(방법당 100건), B는 DoN당 200건(방법당 50건)이며 분석 표의 계획 행 수는 총 **4,400건**이다. B의 겹치는 지점에는 A의 seed 0–49만 선택하여 200건을 재사용할 수 있다. 이때 신규 실제 주행은 A 800 + B 3,400 = **4,200회**이고, 재사용하지 않으면 4,400회다. B에 A의 seed 50–99를 추가해서 방법당 100건으로 늘리지 않는다. 중단되거나 아직 시작하지 않은 trial을 실패로 만들어 계획한 100건 또는 50건을 채우지 않는다.

## 2. 고정 주행 조건

| 항목 | 조건 |
|---|---|
| 환경 / 동역학 / planner | `maze` / `differential_drive` / A* (`astar`) |
| Footprint | Rectangle, 길이 0.150 m, rear_dist=0.0, 너비만 실험별 변경 |
| 제어 주기 / 예측 구간 | dt=0.1 s, horizon=20, tf=2.0 s |
| Trial 제한 | 시뮬레이션 시간 80 s |
| 성공 | 목표 위치의 L1 오차 `abs(dx)+abs(dy) <= 0.02 m`; heading 조건 없음 |
| 정체 실패 | 10초 구간 동안 시작점 대비 위치 이동량 ≤0.03 m |
| 연속 실패 | solver 실패 또는 unsuccessful control step 연속 20회; 성공 시 횟수 초기화 |
| 시작 교란 | x/y 각각 Gaussian 표준편차 0.02 m, heading 표준편차 0.05 rad, 각 성분 ±3σ 제한 |
| 초기 위치 수락 | 실제 footprint의 초기 clearance ≥0.001 m, 최대 1,000회 rejection sampling |
| Seed | A는 각 너비·controller에서 0–99; B는 각 DoN·controller에서 0–49; 같은 너비·seed의 네 controller는 동일 시작 pose |
| Localization error | 비활성화 |
| Debug / animation / plot | 모두 비활성화 (`mpcc.use_debug: false`) |
| 실행 | 같은 PC·동일 Python/라이브러리/CPU thread 설정에서 한 번에 한 프로세스로 순차 실행; thread 기본 6개 (`BENCHMARK_NUM_THREADS`) |

기준 실험에서 보고한 대표 파라미터는 `d_min=0.0001 m`, `a_max=0.3 m/s²`, `r_delta_omega=0.1`이며 CC 초기 표준편차는 `sigma_f0=0.0002 m`, `sigma_l0=0.00025 m`, `sigma_psi0=0.00025 rad`다. 상세 값의 권위 있는 원본은 **해당 실행의 source snapshot, 생성 YAML, 저장된 resolved optimizer parameters**다. 분석 시 현재 checkout의 기본값으로 당시 설정을 추정하지 않는다. CC-ECS의 process noise=0 등 방법 고유의 수식 차이는 실험 오염으로 취급하지 않는다.

`scripts/rectangle_benchmark_baseline.json`에는 2026-11-03 원본에서 추출한 네 방법의 전체 파라미터와 원본 경로·SHA-256을 포함했다. 따라서 다른 PC에 과거 결과 디렉토리 전체를 가져갈 필요는 없다. 실제 실행 전 preflight는 solver를 생성하지 않고 param class를 읽어 `optimizer_parameters.json`과 `baseline_validation.json`을 저장한다. `use_debug`, `debug*`, `constraint_debug_tolerance`, `first_interval_substeps`를 제외한 파라미터가 기준값과 다르면 **주행 시작 전에 중단**한다. `baseline_validation.json`의 `matched: true`와 빈 `differences`를 확인한다. 이는 파라미터 일치 검사이며, 소스 수식·환경까지 과거와 같다는 증명은 아니다.

시작 pose 매칭은 같은 너비 내부에서 확인한다. 너비가 달라지면 clearance rejection 조건 때문에 같은 seed라도 최종 수락 pose가 달라질 수 있다. 너비 간 seed가 같다는 이유로 시작 pose까지 동일하다고 쓰지 않는다.

접촉·겹침 자체는 주행 loop의 별도 실패 조건이 아니다. 따라서 목표 도달 성공률과 접촉 여부는 별개의 지표다.

## 3. 실행 방법과 산출물

저장소 루트에서 각 실험을 차례로 실행한다. 아래 명령은 사용자가 실행할 예시이며, 분석 Agent가 자동 실행할 명령이 아니다.

```bash
bash scripts/run_rectangle_don_benchmark.sh --results-dir benchmark_runs/rectangle_don_100
bash scripts/run_rectangle_don_sweep_benchmark.sh \
  --results-dir benchmark_runs/rectangle_don_sweep_50 \
  --source-snapshot benchmark_runs/rectangle_don_100/source_snapshot
```

각 shell은 `scripts/rectangle_benchmark_suite.py`를 통해 너비별 설정을 만들고, 기존 `test_nmpc.py`를 순차 호출한다. `--source-snapshot`은 앞 실험의 **동일한 코드**를 사용하려는 옵션이다. 완료 데이터를 재사용하거나 결과 파일을 복사하는 옵션이 아니다. `--results-dir`는 매번 새 경로를 지정한다. 기존 결과를 덮어쓰거나 resume하는 기능으로 해석하지 않는다. 기본 Python 대신 특정 환경을 쓰려면 `PYTHON_BIN=/path/to/python`을 지정한다.

DoN 0.75의 완료 데이터를 재사용하려면 아래와 같이 실행한다. `--reuse-results`는 A의 source snapshot도 자동 선택하므로 `--source-snapshot`을 중복 지정할 필요가 없다. 재사용 없는 위 명령은 B의 3,600회를 새로 실행한다.

```bash
bash scripts/run_rectangle_don_sweep_benchmark.sh \
  --results-dir benchmark_runs/rectangle_don_sweep_50 \
  --reuse-results benchmark_runs/rectangle_don_100
```

재사용에는 A suite 완료, source hash·resolved 파라미터·환경의 정확한 일치, 선택한 seed 0–49의 완료 및 정상 trace 검증이 필요하다. B는 검증된 200건의 raw NPZ/metadata를 **자체 canonical 경로로 복사**하고, trial/step CSV를 해당 seed로 필터링하여 방법당 50건의 summary를 다시 계산한다. A 원본은 보존한다. B에는 외부 A raw-trace 경로 의존성이 없지만 재사용 provenance는 남는다. `--reuse-results`와 단순한 `--source-snapshot`을 혼동하지 않는다. 기존 `run_rectangle_success_benchmark.sh` 이름이 남아 있으면 새 B schedule을 가리키는 호환 alias이며, 이전 11개 너비·100회 계획을 실행하는 명령이 아니다.

실제 생성된 `suite_manifest.json`의 경로와 `suite_state.json`의 진행 상태를 우선 읽는다. `--plan-only`는 계획·snapshot·설정만 만들며 주행을 실행하지 않는다. 이 경우 `environment.json`, `optimizer_parameters.json`, `baseline_validation.json`의 실제 실행 정보가 없을 수 있고 결과 완료로 세지 않는다. 주요 구조는 다음과 같다.

`suite_state.status == complete`는 계획한 모든 numerical outcome 행이 기록됐다는 뜻이다. Raw trace까지 모두 정상이라는 뜻은 아니다. 별도 `trace_data_complete`와 너비별 오류 건수 map `trace_export_error_trials`를 확인한다. Trace export 오류가 있어도 목표 도달 결과는 바꾸지 않으며, 최종 분석에는 오프라인 coverage audit가 필요하다.

```text
SUITE_ROOT/
  suite_manifest.json
  suite_state.json
  environment.json
  optimizer_parameters.json
  baseline_validation.json
  source_hashes.json
  source_snapshot/
  logs/
    width_075mm.log
  configs/
    width_075mm.yaml            # A: 75 mm
    width_112p5mm.yaml          # A: 112.5 mm, 정수 mm로 반올림 금지
  widths/
    width_075mm/
      run_manifest.yaml
      trial_results.csv
      summary.csv
      computation_times_*.csv
      runs/
        config01/trial_0001/
          data/
            raw_trace.npz
            raw_trace_metadata.json
```

B는 `width_105mm`, `width_106p5mm`, …, `width_130p5mm`의 18개 너비를 같은 방식으로 저장한다. `config01` 같은 숫자를 controller 이름으로 간주하지 말고 각 너비의 `run_manifest.yaml`과 CSV의 `optimizer_type`으로 매핑한다. 파일명에 쓰인 너비보다 생성 YAML과 metadata의 실제 geometry 값을 우선 확인한다. 재사용 지점의 raw trace도 B 자체 `runs/.../data/`에 복사된다.

상세 console 출력은 live terminal 대신 `logs/width_*.log`에 저장된다. 진행 중인 너비가 75 mm인 경우 다른 terminal에서 `tail -f benchmark_runs/rectangle_don_100/logs/width_075mm.log`로 확인할 수 있다. 이 명령은 로그를 읽기만 한다.

| 파일 | 용도 |
|---|---|
| `suite_manifest.json`, `suite_state.json` | 실험 유형, 계획한 너비·방법·trial·경로 및 실제 진행 상태 |
| `source_snapshot/`, `source_hashes.json` | 실행에 사용한 실제 소스와 파일 해시; dirty checkout의 변경도 포함 |
| `environment.json` | Python·라이브러리·하드웨어·thread 등 실행 환경 확인 |
| `optimizer_parameters.json`, `configs/*.yaml` | 당시 resolved 파라미터와 너비별 명시 설정 |
| `source_snapshot/scripts/rectangle_benchmark_baseline.json`, `baseline_validation.json` | 기준 파라미터·그 원본 해시와 실제 preflight 일치/차이 결과 |
| `widths/*/run_manifest.yaml` | 각 너비의 configuration ID, 방법, seed 설정, timing 범위 |
| `widths/*/trial_results.csv` | trial별 목표 도달 여부, 상태·실패 사유, 시작 pose, 종료 simulation time |
| `widths/*/summary.csv` | 기존 recorder의 성공률 및 성공 solver step 시간 통계; 재집계와 대조할 값 |
| `widths/*/computation_times_*.csv` | 성공 solver step의 `solve_nlp()` 시간 원본, 단위 초 |
| `raw_trace.npz` | `poses`(초기+매 적분 후 Nx3), `pose_times_s`, `applied_inputs`, `solve_nlp_times_s` |
| `raw_trace_metadata.json` | `schema_version`, `geometry`의 bounds·footprint·obstacle polygons, `robot`, `outcome`, `solver_status_infos` 및 배열 개수 |
| `logs/width_*.log` | 너비별 실행 console log; 결과 CSV·metadata와 대조하는 보조 근거 |

`suite_manifest.json`의 `methods` 순서는 `mpcc`, `cc_ecs`, `cc_sf`, `cc_mf`다. `widths`에는 실제 `width_m`, `nominal_don`, `width_id`, `config_path`, `results_dir`가 저장된다. 파일 구조를 추측하지 말고 이 경로를 이용한다. `trial_results.csv`에는 `raw_trace_path`, `raw_trace_metadata_path`, `raw_trace_error`, `safe_stop_count`, `backup_feasible_qp_success_count`, `plant_input_apply_count`, `last_solve_mode`도 남는다. Trace export 오류는 원래 목표 도달 outcome을 바꾸지 않으므로 `raw_trace_error`를 별도로 확인한다.

B의 `experiment`는 `don_sweep`, `trials_per_configuration`은 50이고 A는 `don`, 100이다. 재사용된 너비에는 `widths[].reuse`의 `source_suite_root`, `source_width_id`, `source_results_dir`, `seeds`, `source_aggregate_sha256`, `source_trial_results_sha256`가 기록된다. Trial CSV의 `reused_from_suite`, `reused_from_trial_results`, `reused_from_trial_number`로 원본까지 추적한다. 단순 복사된 trace를 새로운 독립 관측으로 세지 않는다.

Raw trace는 이미 메모리에 쌓인 기록을 **trial 종료 처리에서 optimizer cleanup 전에 저장**한다. 주행 중 clearance를 추가 평가하거나 진단용 예측 궤적을 다시 읽지 않는다. 초기 pose의 clearance 검사와 제어에 필요한 PSDF 평가는 정상 주행 계산으로 유지한다. `use_debug=False`에서 상세 pose-SDF·제약·공분산·cycle CSV가 없다는 것은 누락이 아니다.

## 4. 데이터 유효성 확인 순서

1. 실험 root와 provenance를 먼저 고정한다. Source hash, 환경, resolved 파라미터, timing scope와 `baseline_validation.json`의 일치 결과를 확인한다. 공통 조건은 A·B 및 방법 사이에 맞아야 하지만 방법 고유의 파라미터 차이는 유지한다. 실행일은 실제 manifest의 날짜를 사용한다. 이 문서 작성일이나 이전 보고서 날짜로 대체하지 않는다.
2. 실험별 너비·controller·seed 조합을 세어 A 800 / B 3,600 / 표 전체 4,400건과 비교한다. 재사용했다면 200건은 A와 B 양쪽 cohort에 속하는 같은 원자료이고 신규 실제 주행은 4,200회다. 각 조합의 중복, 미실행, `interrupted`, 실패 원인별 건수를 따로 센다. `summary.csv`의 행 개수만으로 완료를 선언하지 않는다.
3. 같은 너비·seed의 `initial_x`, `initial_y`, `initial_theta`가 네 방법에서 일치하는지 확인한다. 초기 clearance 수락 기준도 동일해야 한다. 매칭 실패나 수락 과정 오류가 있으면 숨기지 말고 해당 pair를 표시한다.
4. 각 trace의 schema, NPZ array shape, 상태·입력·시간 표본 수, finite 값을 확인한다. 초기 pose 1개 + 실제 plant 적분 N개는 상태 N+1개에 대응한다. 마지막 solver 예외처럼 적분되지 않은 solve 기록이 있을 수 있으므로 시간·상태 배열을 길이만 보고 강제로 zip하거나 잘라 맞추지 않는다.
5. `trial_results.csv`, trace metadata의 outcome, `summary.csv`, step CSV를 대조한다. 파일이 없거나 손상된 trial의 clearance를 0으로 채우지 않는다. 결과 상태는 알려져 있으나 trace만 없으면 성공률 집계와 clearance 가용성은 별개로 보고한다.
6. 결과 root가 외부 PC에서 복사됐다면 CSV 안의 절대 `output_root_dir`가 옛 경로일 수 있다. 현재 root 아래 실제 파일 구조와 manifest를 기준으로 경로를 해석하고, 같은 이름의 현재 checkout 파일로 대체하지 않는다.

재사용하려면 최소한 너비, 방법, seed, 시작 pose, geometry·종료 기준·noise 설정, resolved 파라미터, 소스 해시, timing scope, 실행 환경이 일치하는지 검증한다. 재사용 trial의 원본 경로·hash와 seed 범위를 보존한다. **실험별 표**에서는 A의 DoN 0.75는 100건, B의 DoN 0.75는 선택한 50건으로 각각 집계한다. 둘을 합쳐 같은 방법의 독립적인 150건으로 해석하지 않는다. 실험 통합 고유 cohort를 만들 때만 `(width, controller, seed, experiment fingerprint)`와 원본 trace로 중복 제거하고 사용한 규칙을 밝힌다. 재사용하지 않고 같은 seed를 다시 실행했다면 별도 실행 반복으로 표시한다. 이전 2026-11-03의 10-trial 결과를 이번 결과에 단순 합산하지 않는다.

## 5. 지표 정의와 분석 규칙

### 5.1 성공률: 두 실험 공통, B의 주지표

- 분자: `status == success`인 목표 도달 trial 수.
- 분모: `success`, `failure`, `timeout`, `error`로 끝난 완료 trial 수. 사용자 중단인 `interrupted`는 제외한다.
- `successes / completed`와 백분율을 함께 제시한다. A의 계획 n=100, B의 계획 n=50이며 실제 완료 수가 적으면 `x/100` 또는 `x/50`으로 꾸미지 않는다. 성공 0건도 누락값이 아닌 관측된 0이며, 완료 0건은 계산 불가다.
- 각 너비·방법에 이항 비율의 **Wilson 95% 신뢰구간**을 제시한다. `p ± 1.96*sqrt(p*(1-p)/n)` 근사는 0/100·100/100·0/50·50/50에서 부적절하므로 사용하지 않는다.
- Wilson 계산은 `z=1.959963984540054`, `d=1+z²/n`, 중심 `(p+z²/(2n))/d`, 반폭 `z/d*sqrt(p*(1-p)/n+z²/(4n²))`다. `n=0`이면 구간을 비워둔다.
- 같은 너비·seed의 방법끼리는 paired 비교다. 필요하면 공통 완료 seed에 대해 일치 성공/일치 실패/방법별 불일치 건수를 먼저 제시한다. 유의성 검정을 추가하면 비교를 사전에 정한 것인지, 여러 너비·방법의 탐색적 비교인지와 다중비교 처리를 명시한다.

성공률 plot의 주 x축은 **DoN 0.70–0.87, 간격 0.01**로 두고 방법별 선·점, 95% 구간, 각 지점의 실제 n을 표시한다. 대응하는 너비 0.1050–0.1305 m, 간격 0.0015 m를 보조 축이나 표에 제시할 수 있다. A와 B의 DoN 0.75 표본 수가 다름을 표시하고 같은 곡선의 추가 독립 표본으로 겹쳐 합산하지 않는다.

### 5.2 Computation time: A의 주행 정량 지표

표준 값은 `computation_times_*.csv`의 `solver_computation_time_s`다. 네 controller 모두 같은 decorator가 `time.perf_counter()`로 **`solve_nlp()` 진입부터 반환 또는 예외까지 전체 wall time**을 잰다.

포함하는 작업은 함수 안의 parameter 갱신·warm-start 준비·주 solver·backup solver·복구·safe-stop 생성이다. 외부 `setup()`, local/global planning, plant 적분, controller 로깅, trial 종료 후 원자료 저장, offline clearance 계산은 제외한다. Debug를 켜면 함수 안의 진단 비용도 포함되므로 이번 실험은 반드시 `use_debug=False`여야 한다.

주 통계는 **solver가 성공한 step만** 사용하며, 첫 step을 포함한다. 최종적으로 실패·timeout·error가 된 trial의 성공 step도 포함한다. Backup으로 성공한 step은 해당 전체 호출 시간을 포함한다. Solver를 호출하지 않은 stage-0 stop, 실패 solver, 성공 여부 불명확·유효하지 않은 시간 표본은 성공-step 주 통계에서 제외한다. Raw trace에 남아 있는 실패·stop 호출 시간은 필요 시 보조 분석으로 별도 표시하고 주 통계와 섞지 않는다.

같은 너비·방법의 모든 유효 step 시간을 한 벡터로 모아 mean/P95를 계산한다. **trial mean의 평균이나 trial P95의 평균이 아니다.** 초 단위를 유지해 계산하고 보고서 표시 때만 1,000을 곱해 ms/step으로 바꾼다. 유효 step 수, 포함한 trial 수, 제외 이유를 함께 적는다. 유효 표본이 없으면 `—`로 표시한다.

기준 보고서의 시간은 local reference부터 controller 전체 cycle을 재고, trial 첫 cycle을 제외하며, 실패·fallback도 포함하고 CUDA 동기화를 추가했다. 그보다 앞선 내부 `solver.solve()` 시간 역시 현재 정의와 다르다. **이전 보고서의 수치와 현재 ms/step을 같은 측정량처럼 비교하지 않는다.** 이번 방식의 wall-clock 값을 별도 CUDA-event kernel time이나 명시 동기화 계측으로 바꾸어 재해석하지 않는다.

순차 실행이 모든 환경 변동을 제거하는 것은 아니다. 온도·전력·다른 프로세스·method 실행 순서 차이가 있으면 남은 한계를 적는다. 저장된 해시와 환경 정보가 다른 실행을 동일한 timing cohort로 합치지 않는다.

### 5.3 Clearance: 주행 종료 후에만 계산

Raw trace에 저장된 **실제 초기 pose와 매 plant 적분 후 pose**에서 계산한다. 예측 궤적, current PSDF 로그, affinity row, chance margin을 기하학적 clearance의 대체값으로 사용하지 않는다.

각 pose에서 local rectangle 꼭짓점을 실제 pose로 world 변환하고, 모든 obstacle polygon과의 정확한 Euclidean polygon 거리 및 지도 bounds margin 중 최솟값을 취한다. 구현 정의는 `sim/start_perturbation.py`의 `pose_clearance()`와 `convex_polygon_clearance()`다. 접촉·obstacle 겹침은 0, 지도 경계 침범은 음수다. 따라서 음수를 obstacle penetration depth라고 설명하지 않는다.

같은 너비·방법에서 `success`·`failure`·`timeout`·`error` 완료 trial의 가용한 모든 pose 표본을 합쳐 mean/P05를 계산한다. `interrupted`는 주 cohort에서 제외하고 별도 건수를 보고한다. Trial별 mean/P05를 다시 평균하지 않는다. 표본 수, trace 가용 trial 수, missing·invalid trial 수를 같이 기록한다. 오래 주행한 trial이 더 많은 표본을 제공하는 **step 가중 통계**라는 점을 밝힌다. 필요하면 trial 균등 가중이나 성공-only clearance를 보조 표로 따로 제시한다.

오프라인 helper는 다음과 같이 실행한다. 이 명령은 저장된 원자료만 읽고 solver나 simulation을 호출하지 않는다.

```bash
python scripts/analyze_rectangle_clearance.py \
  benchmark_runs/rectangle_don_100 \
  benchmark_runs/rectangle_don_sweep_50 \
  --output-dir benchmark_runs/rectangle_clearance_analysis \
  --strict
```

기본 출력은 첫 번째 root의 새 `offline_clearance/`이며 `--output-dir`로 다른 새 경로를 지정할 수 있다. `clearance_trials.csv`, `clearance_summary.csv`, `analysis_manifest.json`을 생성한다. 원본 trace나 성공/실패 결과는 변경하지 않는다. 실제 helper의 `--help`와 metadata schema를 먼저 확인하고, 다른 버전의 파일 구조를 추정해서 처리하지 않는다.

`clearance_summary.csv`의 `suite_root`, `experiment`, `expected_trials`, `recorded_trials`, `unrecorded_trials`, `completed_trials`, `interrupted_trials`, `available_completed_trials`, `unavailable_completed_trials`, `pooled_pose_samples`, `coverage_complete`로 coverage를 확인한다. `--strict`는 분석·audit 파일을 쓴 뒤 미완료 또는 누락·손상 trace가 있으면 종료 코드 2를 반환한다. 부분 결과가 저장됐다고 완전한 cohort로 보지 않는다. Helper는 A와 B를 별도 cohort로 유지하며 DoN 0.75의 A100/B50을 합치지 않는다. 같은 suite 내부의 중복 조합은 거부한다. Suite 간 재사용 원자료는 provenance를 검증하되 실험별 cohort의 통계에는 각각 포함한다.

`clearance_trials.csv`에는 `reused`, `reused_from_suite`, `reused_from_trial_number`가 표시된다. `analysis_manifest.json`의 `cohort_trial_memberships`(계획대로 완료하면 4,400)와 `unique_source_trial_origins`(200건 재사용 시 4,200), `reused_trial_memberships`(재사용 시 200)를 구분해 보고한다. `unique_available_source_trace_records`는 복사 전 원본 NPZ/metadata 쌍의 hash 기준 가용 원자료 수다. 누락·미완료가 있으면 실제 값이 계획값과 다를 수 있다.

접촉 또는 음수 clearance 표본 비율·해당 trial 수를 추가할 수 있다. 부동소수점 임계값을 쓰면 값을 적고, 해당 지표는 **dt=0.1 s에서 관측한 이산 표본**이라는 한계를 명시한다. 표본 사이의 연속 시간 무충돌이나 확률적 안전 보장을 주장하지 않는다. Clearance가 양수인 guard stop은 실제 접촉과 구분한다.

### 5.4 Traversal time과 기타 주행 지표

Traversal time은 `status == success`인 trial의 최종 시뮬레이션 시간 `final_time` 평균이다. 성공 0건이면 `—`다. Wall-clock 실행 시간이나 실패 trial 종료 시간을 성공 주행 시간에 섞지 않는다.

방법마다 성공한 seed 집합이 다르면 성공-only 평균은 서로 다른 cohort의 조건부 통계다. 원래 성공 trial 수를 병기하고, 방법 간 속도 차이를 논할 때는 공통으로 성공한 paired seed 결과도 함께 제시한다. 실패 trial에 80초를 임의 대입한 수치를 traversal time이라고 부르지 않는다.

Raw trace가 충분하면 실제 이동 거리, 선속도·각속도 profile 등 보조 지표를 재구성할 수 있다. 새 지표마다 initial sample 포함 여부, finite 표본 처리, trial 가중 방식과 단위를 먼저 정의한다. 제어 입력 변화량과 물리적인 속도·가속도를 구분한다.

### 5.5 Safe stop, recovery와 실패 원인

기본 safe-stop 지표는 metadata의 `optimizer_runtime_stats.safe_stop_count` 합계다. 이는 zero-input fallback이 적용된 **제어 step 수**다. 연속 20회 stop은 20 step이지 독립적인 위험 사건 20개가 아니다. Stage-0 guard stop과 solver 실패 후 fallback이 구분되는 카운터가 있으면 함께 제시한다.

`backup_feasible_qp_success_count`는 복구 성공 횟수다. 단순히 실제 입력이 0인 step을 safe stop으로 재분류하지 않는다. 정상 최적해도 0 입력일 수 있다. Runtime 카운터가 없는 방법·파일은 0으로 만들지 않는다. MPCC는 해당 fallback 자체가 없다는 source 근거가 확인된 경우에만 구조적 0으로 설명한다.

Episode 수, 최장 연속 stop 길이, 정확한 발생 시점을 구하려면 그 정의를 구별할 per-step solve-mode/stop 정보가 필요하다. Aggregate 카운터나 마지막 `last_solve_mode`만으로 episode를 복원하지 않는다. 이번 최소 trace가 이를 제공하지 않으면 `미계측`으로 보고한다. Debug를 다시 켜서 새로 실행한 결과를 기존 timing cohort에 합치지 않는다.

실패는 `failure_reason`을 원문 보존한 채 정체, 연속 실패, stage-0 stop 반복, timeout, solver/기타 exception으로 분류한다. 목표 미도달과 실제 접촉을 동일시하지 않는다. 상세 내부 제약 로그가 없는 상태에서는 원인을 기전 수준으로 단정하지 않는다.

## 6. 후속 Agent가 만들어야 할 결과

먼저 가용성·완료성·소스/설정 일치 여부를 짧게 보고하고, 유효한 데이터 범위에서 다음을 작성한다.

1. A: DoN 0.5와 0.75 각각에서 네 방법의 성공/완료, Wilson 95% 구간, mean/P05 clearance(mm), mean/P95 `solve_nlp()` time(ms/step), 성공-only 평균 traversal(s), safe-stop step, recovery count 표. 각 지표의 표본 수를 함께 남긴다.
2. B: DoN 0.70–0.87의 18개 지점 × 네 방법의 성공/완료(n=50 계획) 및 95% 구간 표와 DoN 축의 성공률 plot. 어떤 DoN에서 성공률이 변했는지 관측 범위로 설명하고, 미측정 지점의 임계값을 확정하지 않는다. DoN 0.75의 200건 재사용 여부를 명시한다.
3. 완료 수·중단·미실행·중복·trace 누락, 실패 원인 분포, 시작 pose 매칭 결과, paired 비교에 사용한 seed 목록을 별도 CSV 또는 audit JSON에 보존한다.
4. 집계 근거 경로, 코드/설정 hash, 분석 코드와 지표 정의를 포함한 Markdown 보고서를 새 분석 디렉토리에 저장한다. 원본 CSV·trace·이전 보고서를 덮어쓰지 않는다.

실험이 부분 완료라면 계획한 A의 100-trial 또는 B의 50-trial 결론 대신 현재 완료 수와 남은 조합을 먼저 적는다. `error`나 계측 누락을 지워 성공률을 높이지 않는다. 데이터가 뒷받침하지 않는 우열·인과·안전성 주장을 만들지 않는다.

## 7. LLM Agent에 전달할 요청 예시

> `docs/rectangle_benchmark_analysis_context.md`를 기준으로 `<A_RESULT_ROOT>`와 `<B_RESULT_ROOT>`의 이미 저장된 실험 결과를 분석해줘. 주행 실험은 실행하지 말고 provenance·완료 수·시작 pose 매칭부터 검증해줘. Clearance는 저장된 원자료로 오프라인 계산하고, A의 두 DoN 100-trial 주행 정량 비교와 B의 DoN 0.70–0.87 각 50-trial 성공률을 각각 표·그래프로 정리해줘. DoN 0.75 재사용 데이터는 실험별 cohort를 유지하고 독립 표본으로 중복 합산하지 마. `solve_nlp()` 시간의 성공 step cohort, 실패·중단 분모, 성공-only traversal의 선택 편향과 누락 데이터를 명시하고 새 분석 디렉토리에 근거와 Markdown 보고서를 저장해줘.
